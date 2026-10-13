"""Private isolated runtime entry point; stdin/stdout are a JSON RPC channel.

Only local pinned weights are loaded. No Hub code and no canonical text writes.
"""

from __future__ import annotations

import difflib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys
import traceback
import unicodedata

import numpy as np

_RPC = sys.stdout
SR = 16000
BASMALA = ["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ"]


def reply(value):
    _RPC.write(json.dumps(value, ensure_ascii=False) + "\n")
    _RPC.flush()


def plain(text):
    text = unicodedata.normalize("NFC", text).translate(str.maketrans({
        "ٱ": "ا", "ۥ": "و", "ۦ": "ي", "ـ": "", "۩": "", "۞": "",
    }))
    return "".join(c for c in text if unicodedata.category(c) not in ("Mn", "Me", "Cf"))


def softmax_log(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


class Acoustic:
    def __init__(self, model_id, folder):
        import torch

        torch.set_num_threads(4)
        self.torch = torch
        self.model_id = model_id
        self.blank_id = 0
        self.text_kind = model_id
        if model_id == "mms-fa":
            from safetensors.torch import load_file
            from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

            self.model = Wav2Vec2ForCTC(Wav2Vec2Config.from_pretrained(folder, local_files_only=True))
            # This repo is a converted HF state dict, not torchaudio keys.
            weights = load_file(str(folder / "model-fp16.safetensors"))
            self.model.load_state_dict(weights, strict=True)
            self.model.eval()
            # Official MMS_FA classifier order (no word separator, no star).
            from torchaudio.pipelines import MMS_FA
            self.vocab = {t: i for i, t in enumerate(MMS_FA.get_labels(star=None))}
            import uroman

            self.roman = uroman.Uroman()
        elif model_id == "muaalem":
            import onnxruntime as ort
            from transformers import SeamlessM4TFeatureExtractor

            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            self.session = ort.InferenceSession(str(folder / "muaalem/model.fp32.onnx"),
                                               sess_options=options, providers=["CPUExecutionProvider"])
            head = json.loads((folder / "muaalem/tokens.json").read_text())["outputs"]["phonemes"]
            self.vocab = {text: int(i) for i, text in head["id_to_text"].items()}
            self.features = SeamlessM4TFeatureExtractor()
        elif model_id.startswith("fastconformer"):
            from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

            archives = list(folder.rglob("*.nemo"))
            if len(archives) != 1:
                raise ValueError("Ожидается один checkpoint NeMo.")
            self.model = EncDecHybridRNNTCTCBPEModel.restore_from(str(archives[0]), map_location="cpu")
            self.model.eval()
            self.model.preprocessor.featurizer.dither = 0.0
            self.model.preprocessor.featurizer.pad_to = 0
            # Hybrid forward uses selected decoder; explicitly select the trained CTC head.
            self.model.change_decoding_strategy(decoder_type="ctc")
            self.blank_id = len(self.model.ctc_decoder.vocabulary)
            self.vocab = {f"token:{i}": i for i in range(self.blank_id + 1)}
        elif model_id.startswith("whisper-"):
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            self.processor = WhisperProcessor.from_pretrained(folder, local_files_only=True)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                folder, local_files_only=True, attn_implementation="eager").eval()
            self.processor.tokenizer.set_prefix_tokens(language="ar", task="transcribe", predict_timestamps=False)
            self.vocab = {"unused": 0}
            self.text_kind = "whisper"
        elif model_id == "mms-arabic":
            from safetensors import safe_open
            from safetensors.torch import load_file
            from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor

            self.processor = Wav2Vec2Processor.from_pretrained(folder, target_lang="ara", local_files_only=True)
            # HF tie_weights loads the language head before the base state,
            # whose saved head can differ from config.vocab_size. Restore
            # the exact base state first, then replace only the ara adapter.
            config = Wav2Vec2Config.from_pretrained(folder, local_files_only=True)
            with safe_open(str(folder / "model.safetensors"), framework="pt") as weights:
                config.vocab_size = weights.get_slice("lm_head.weight").get_shape()[0]
            self.model = Wav2Vec2ForCTC(config)
            self.model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
            self.model.config._name_or_path = str(folder)
            self.model.load_adapter("ara", local_files_only=True, use_safetensors=True)
            self.model.eval()
            self.vocab = self.processor.tokenizer.get_vocab()
            self.blank_id = self.processor.tokenizer.pad_token_id
            if len(self.vocab) != self.model.config.vocab_size:
                raise ValueError("Арабский адаптер MMS не соответствует словарю.")
        else:
            raise ValueError("Unknown acoustic runtime")

    def targets(self, words, surah, intro, renderings=None, ayahs=None):
        prefix = BASMALA if intro else []
        if self.model_id == "muaalem":
            from quran_transcript import MoshafAttributes, quran_phonetizer

            words = renderings
            # Phonetize whole canonical ayahs with source character mappings:
            # connected pronunciation and word ownership must be preserved.
            attrs = MoshafAttributes(rewaya="hafs", madd_monfasel_len=4,
                                    madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=4)
            grouped = ([prefix] if prefix else []) + [
                [word for _, word in group] for _, group in itertools.groupby(
                    zip(ayahs, words), key=lambda pair: pair[0])]
            targets, spans = [], []
            for group in grouped:
                text = " ".join(group)
                phon = quran_phonetizer(text, attrs, sura_idx=surah, remove_spaces=True)
                base = len(targets)
                targets.extend(self.vocab[c] for c in phon.phonemes)
                char_offset = 0
                for word in group:
                    mappings = phon.mappings[char_offset:char_offset + len(word)]
                    positions = [m.pos for m in mappings if not m.deleted and m.pos[1] > m.pos[0]]
                    if not positions:
                        raise ValueError("Слово не представлено фонемами Muaalem.")
                    spans.append((base + min(p[0] for p in positions), base + max(p[1] for p in positions)))
                    char_offset += len(word) + 1
            return targets, spans[len(prefix):]
        if renderings is not None:
            words = renderings
        if self.model_id.startswith("fastconformer"):
            prefix = ["بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ"] if intro else []
            if self.model_id == "fastconformer-arabic":
                prefix = [plain(word) for word in prefix]
        targets, spans = [], []
        for word in prefix + words:
            if self.model_id == "mms-fa":
                text = self.roman.romanize_string(plain(word), lcode="ara").lower()
                text = "".join(c for c in text if (c.isalpha() or c == "'") and c in self.vocab)
                ids = [self.vocab[c] for c in text]
            elif self.model_id == "mms-arabic":
                text = plain(word)
                ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
                if self.processor.tokenizer.unk_token_id in ids:
                    raise ValueError("Слово не представлено арабским словарём MMS.")
                delimiter = self.processor.tokenizer.word_delimiter_token_id
                if targets:
                    targets.append(delimiter)
            else:
                # SentencePiece consumes diacritized Arabic; remove only Quran
                # annotation signs unsupported by the tokenizer, never harakat.
                text = unicodedata.normalize("NFC", word).replace("ٱ", "ا").replace("ٰ", "").replace("۩", "").replace("۞", "")
                ids = self.model.tokenizer.text_to_ids(text)
                variants = getattr(self, "bpe_variants", {})
                ids = variants.get(text, ids)
                if self.model.tokenizer.tokenizer.unk_id() in ids:
                    print("Unsupported NeMo word:", repr(text), self.model.tokenizer.ids_to_tokens(ids), file=sys.stderr)
                    raise ValueError("Слово не представлено словарём FastConformer.")
            start = len(targets)
            targets.extend(ids)
            if not ids or self.blank_id in ids:
                raise ValueError("Пустое слово или blank в канонической цели.")
            spans.append((start, len(targets)))
        return targets, spans[len(prefix):]

    def emissions(self, pcm):
        torch = self.torch
        if self.model_id == "muaalem":
            features = self.features(pcm, sampling_rate=SR, return_tensors="np")["input_features"][0]
            # Already normalized per utterance, pairs of Kaldi fbank frames.
            return self._window(features, 25, self._muaalem, axis=0)
        if self.model_id in ("mms-fa", "mms-arabic"):
            pcm = (pcm - pcm.mean()) / np.sqrt(pcm.var() + 1e-7)

            def run(audio):
                with torch.inference_mode():
                    return self.model(torch.from_numpy(audio[None])).logits[0].numpy()

            return self._window(pcm, 50, run)
        if self.model_id.startswith("fastconformer"):
            def run(audio):
                with torch.inference_mode():
                    features, feature_lengths = self.model.preprocessor(
                        input_signal=torch.from_numpy(audio[None]), length=torch.tensor([len(audio)]))
                    encoded, lengths = self.model.encoder(
                        audio_signal=features, length=feature_lengths)
                    log_probs = self.model.ctc_decoder(encoder_output=encoded)
                    return log_probs[0, :int(lengths[0])].numpy()

            result = self._window(pcm, 12.5, run)
            # SentencePiece has equivalent segmentations of the same word.
            # Prefer a variant actually emitted by this head only when its
            # decoded spelling equals the fixed target exactly. No ASR words
            # or ASR times are substituted into the reference sequence.
            raw = result.argmax(1)
            ids = [int(v) for i, v in enumerate(raw)
                   if v != self.blank_id and (i == 0 or v != raw[i - 1])]
            groups, current = [], []
            for token in ids:
                piece = self.model.ctc_decoder.vocabulary[token]
                if piece.startswith("▁") and current:
                    groups.append(current)
                    current = []
                current.append(token)
            if current:
                groups.append(current)
            self.bpe_variants = {}
            for i in range(len(groups)):
                for width in (1, 2, 3):
                    selected = sum(groups[i:i + width], [])
                    text = self.model.tokenizer.ids_to_text(selected).strip()
                    if text:
                        self.bpe_variants[text] = selected
            return result
        raise ValueError("Whisper не является CTC-моделью.")

    def _muaalem(self, features):
        return self.session.run(["phonemes"], {"input_features": features[None]})[0][0]

    def _window(self, audio, fps, run, axis=None):
        rate = 50 if axis == 0 else SR  # Muaalem feature rows are 20ms
        window, context = 30 * rate, 2 * rate
        if len(audio) <= window:
            return softmax_log(run(np.ascontiguousarray(audio, dtype=np.float32)))
        result = []
        for start in range(0, len(audio), window):
            lo, hi = max(0, start - context), min(len(audio), start + window + context)
            logits = run(np.ascontiguousarray(audio[lo:hi], dtype=np.float32))
            left = round((start - lo) / rate * fps)
            count = round(min(window, len(audio) - start) / rate * fps)
            result.append(logits[left:left + count])
        return softmax_log(np.concatenate(result))

    def whisper_clip(self, pcm, words, strict_words=True):
        """Teacher-force known words; cross-attention heads -> monotone DTW."""
        from transformers.models.whisper.generation_whisper import _dynamic_time_warping

        torch = self.torch
        tokenizer = self.processor.tokenizer
        tokens, spans = [], []
        for word in words:
            ids = tokenizer.encode(" " + word.replace("ٱ", "ا"), add_special_tokens=False)
            first = 0
            while first < len(ids) and not tokenizer.decode([ids[first]]).strip():
                first += 1
            if first == len(ids):
                raise ValueError("Пустое слово Whisper.")
            spans.append((len(tokens) + first, len(tokens) + len(ids)))
            tokens.extend(ids)
        prefix = tokenizer.prefix_tokens
        if len(prefix) + len(tokens) + 1 > self.model.config.max_target_positions:
            raise ValueError("Слишком много текста для одного окна Whisper.")
        features = self.processor(pcm, sampling_rate=SR, return_tensors="pt").input_features
        decoder = torch.tensor([[*prefix, *tokens, tokenizer.eos_token_id]])
        with torch.inference_mode():
            # Retain decoder cross-attention only. Keeping every encoder
            # attention matrix would add several GB for large-v3.
            encoded = self.model.model.encoder(features, output_attentions=False, return_dict=True)
            out = self.model(encoder_outputs=encoded, decoder_input_ids=decoder,
                             output_attentions=True, use_cache=False)
        heads = self.model.generation_config.alignment_heads
        if not heads:
            raise ValueError("В конфигурации Whisper отсутствуют alignment heads.")
        frames = min(1500, max(1, len(pcm) // 320))
        weights = torch.stack([out.cross_attentions[layer][0, head, :, :frames] for layer, head in heads]).numpy()
        # Same attention normalization/median smoothing as HF token timestamps.
        weights = (weights - weights.mean(axis=-2, keepdims=True)) / np.maximum(weights.std(axis=-2, keepdims=True), 1e-6)
        padded = np.pad(weights, ((0, 0), (0, 0), (3, 3)), mode="reflect")
        weights = np.median(np.lib.stride_tricks.sliding_window_view(padded, 7, axis=-1), axis=-1)
        matrix = weights.mean(axis=0)[len(prefix)-1:-1]
        text_indices, time_indices = _dynamic_time_warping(-matrix)
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        times = time_indices[jumps] * 20
        probs = out.logits[0, len(prefix)-1:len(prefix)-1+len(tokens)].softmax(-1)
        probs = probs[torch.arange(len(tokens)), torch.tensor(tokens)].numpy()
        # DTW row 0 is the no-timestamps prefix, an absorber for leading
        # silence. Text queries start at row 1. A standalone space token is
        # retained in teacher forcing but cannot mark a spoken word onset.
        rows = [(int(times[start + 1]), float(probs[start:end].mean())) for start, end in spans]
        if strict_words and any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
            raise ValueError("Whisper не различил начала канонических слов. Уточните границы записи.")
        for i in range(1, len(rows)):
            if rows[i][0] == rows[i - 1][0]:
                rows[i] = (rows[i][0], 0.0)
        return rows

    def whisper_align(self, pcm, words, strict_words=True):
        if len(pcm) <= 30 * SR:
            return self.whisper_clip(pcm, words, strict_words=strict_words)
        # ASR is only a coarse locator for bounded windows. All final timing
        # rows come from teacher forcing the original canonical word sequence.
        result, cursor, offset = [], 0, 0
        torch = self.torch
        while cursor < len(words):
            clip = pcm[offset:offset + 30 * SR]
            if len(clip) < SR // 10:
                raise ValueError("Недостаточно аудио для оставшегося текста Whisper.")
            features = self.processor(clip, sampling_rate=SR, return_tensors="pt").input_features
            with torch.inference_mode():
                generated = self.model.generate(features, language="ar", task="transcribe",
                                                return_timestamps=True, max_new_tokens=400,
                                                do_sample=False, num_beams=1)
            hypothesis = self.processor.batch_decode(generated, skip_special_tokens=True)[0].split()
            canonical = [plain(w) for w in words[cursor:cursor + 180]]
            observed = [plain(w) for w in hypothesis]
            blocks = difflib.SequenceMatcher(None, canonical, observed, autojunk=False).get_matching_blocks()
            matched = sum(b.size for b in blocks)
            longest = max((b.size for b in blocks), default=0)
            if (not observed or matched < max(2, len(observed) * 0.5)
                    or (len(observed) >= 8 and longest < 3)):
                print("Whisper anchors:", cursor, offset / SR, observed, canonical, matched, file=sys.stderr)
                raise ValueError("Whisper не нашёл достаточных опор в каноническом тексте.")
            end = max(b.a + b.size for b in blocks if b.size)
            segment_words = words[cursor:cursor + end]
            rows = self.whisper_clip(clip, segment_words, strict_words=strict_words)
            final = offset + len(clip) == len(pcm)
            if final and cursor + end != len(words):
                raise ValueError("Whisper не подтвердил весь конец канонического текста.")
            commit = end if final else next((i for i, row in enumerate(rows) if row[0] >= 24000), end - 2)
            if commit <= 0 or (not final and commit < 2):
                raise ValueError("Whisper не продвинулся по каноническому тексту.")
            base_ms = offset * 1000 // SR
            result.extend((base_ms + ms, conf) for ms, conf in rows[:commit])
            cursor += commit
            if cursor < len(words):
                # Re-read one confirmed word as context, and replace that one
                # automatic estimate atomically within this calculation.
                cursor -= 1
                new_offset = result.pop()[0] * SR // 1000
                if new_offset <= offset:
                    raise ValueError("Whisper не продвинулся по аудио.")
                offset = new_offset
        if strict_words and any(b[0] <= a[0] for a, b in zip(result, result[1:])):
            raise ValueError("Whisper дал конфликт порядка слов.")
        return result


def main():
    global _RPC
    _RPC = sys.stdout
    sys.stdout = sys.stderr  # libraries sometimes print during restore/import
    model_id, folder, work = sys.argv[1:]
    work = Path(work)
    try:
        acoustic = Acoustic(model_id, Path(folder))
        versions = {p: importlib.metadata.version(p) for p in
                    ("torch", "transformers", "numpy")}
        for package in ("torchaudio", "quran-transcript", "uroman", "onnxruntime", "nemo-toolkit", "sentencepiece"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        reply({"vocab": acoustic.vocab, "blank_id": acoustic.blank_id,
               "text_kind": acoustic.text_kind, "versions": versions})
    except Exception as exc:
        traceback.print_exc()
        reply({"error": f"Не удалось загрузить модель: {exc}"})
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request["action"]
            if action == "targets":
                targets, spans = acoustic.targets(request["words"], request["surah"], request["intro"],
                                                   request["renderings"], request["ayahs"])
                reply({"targets": targets, "spans": spans})
            elif action == "emissions":
                pcm = np.load(work / "audio.npy", allow_pickle=False)
                np.save(work / "emissions.npy", acoustic.emissions(pcm))
                reply({"ok": True})
            elif action == "align":
                pcm = np.load(work / "audio.npy", allow_pickle=False)
                words = (BASMALA if request["intro"] else []) + request["renderings"]
                rows = acoustic.whisper_align(pcm, words, strict_words=request["strict_words"])
                reply({"rows": rows[len(BASMALA) if request["intro"] else 0:]})
            elif action == "probe":
                # A real encoder/decoder forward; silence has no usable word
                # boundary evidence, so the probe does not run timing validation.
                torch = acoustic.torch
                features = acoustic.processor(np.zeros(SR, np.float32), sampling_rate=SR, return_tensors="pt").input_features
                with torch.inference_mode():
                    out = acoustic.model(features, decoder_input_ids=torch.tensor([acoustic.processor.tokenizer.prefix_tokens]))
                if not torch.isfinite(out.logits).all():
                    raise ValueError("Некорректный выход Whisper.")
                reply({"ok": True})
            else:
                raise ValueError("Unknown runtime action")
        except Exception as exc:
            traceback.print_exc()
            reply({"error": str(exc)})


if __name__ == "__main__":
    main()
