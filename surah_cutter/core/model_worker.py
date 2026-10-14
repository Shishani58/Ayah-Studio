"""Private isolated runtime entry point; stdin/stdout are a JSON RPC channel.

Only local pinned weights are loaded. No Hub code and no canonical text writes.
"""

from __future__ import annotations

import difflib
import hashlib
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
WHISPER_FULL_VERSION = "pause-retry-spelled-letters-v2"
BASMALA = ["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ"]


class WhisperTextOverflow(ValueError):
    """Known target cannot fit in the decoder's positional budget."""


def reply(value):
    _RPC.write(json.dumps(value, ensure_ascii=False) + "\n")
    _RPC.flush()


def plain(text):
    text = unicodedata.normalize("NFC", text).translate(str.maketrans({
        "ٱ": "ا", "ۥ": "و", "ۦ": "ي", "ـ": "", "۩": "", "۞": "",
    }))
    return "".join(c for c in text if unicodedata.category(c) not in ("Mn", "Me", "Cf"))


def whisper_alignment_text(word):
    """Spoken names for Quran opening letters, retaining one canonical span."""
    normalized = plain(word)
    if normalized not in {"الم", "المص", "الر", "المر", "كهيعص", "طه", "طسم", "طس",
                           "يس", "ص", "حم", "عسق", "ق", "ن"}:
        return word
    names = {"ا": "أَلِفْ", "ل": "لَامْ", "م": "مِيمْ", "ص": "صَادْ", "ر": "رَا",
             "ك": "كَافْ", "ه": "هَا", "ي": "يَا", "ع": "عَيْنْ", "ط": "طَا",
             "س": "سِينْ", "ح": "حَا", "ق": "قَافْ", "ن": "نُونْ"}
    return " ".join(names[letter] for letter in normalized)


def whisper_energy_pauses(pcm):
    """Bounded RMS pauses; worker also runs outside the application package."""
    frame = SR // 50
    count = len(pcm) // frame
    if count < 5:
        return []
    rms = np.sqrt(np.mean(pcm[:count * frame].reshape(count, frame) ** 2, axis=1))
    rms = np.convolve(rms, np.ones(5) / 5, mode="same")
    noise, speech = np.percentile(rms, [15, 70])
    cutoff = max(1e-5, min(noise + (speech - noise) * .18, speech * .35))
    transitions = np.diff(np.r_[False, rms <= cutoff, False].astype(np.int8))
    pauses = []
    for lo, hi in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
        if hi - lo < 15 or lo < 5 or hi > count - 5:
            continue
        left = rms[max(0, lo - 18):lo].mean()
        right = rms[hi:min(count, hi + 18)].mean()
        deep_long = hi - lo >= 35 and rms[lo:hi].mean() < min(left, right) / 2.4
        if min(left, right) > cutoff * 2.2 or (deep_long and max(left, right) > cutoff * 2.2):
            pauses.append((float(lo * 20), float(hi * 20)))
    return pauses


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
            from transformers import GenerationConfig, WhisperConfig, WhisperProcessor, WhisperForConditionalGeneration

            self.processor = WhisperProcessor.from_pretrained(folder, local_files_only=True)
            if model_id in ("whisper-tarteel-tiny", "whisper-tarteel-base"):
                # Tarteel publishes only a state dict and no generation config.
                # Never unpickle Hub code; use explicit safe loading and strict keys.
                config = WhisperConfig.from_pretrained(folder, local_files_only=True)
                config._attn_implementation = "eager"
                self.model = WhisperForConditionalGeneration(config)
                weights = torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
                # Old Transformers omitted the tied output projection from checkpoints.
                if "proj_out.weight" not in weights and config.tie_word_embeddings:
                    if self.model.proj_out.weight is not self.model.model.decoder.embed_tokens.weight:
                        raise ValueError("Whisper output projection is not tied to the token embedding.")
                    weights["proj_out.weight"] = weights["model.decoder.embed_tokens.weight"]
                self.model.load_state_dict(weights, strict=True)
                entry = next(m for m in json.loads(Path(__file__).with_name("model_catalog.json").read_text())
                             if m["id"] == model_id)
                # Explicit, pinned upstream metadata, not different model weights.
                # These heads were selected for OpenAI, not calibrated by Tarteel.
                self.model.generation_config = GenerationConfig.from_dict(entry["generation_config"])
                self.model.eval()
            else:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    folder, local_files_only=True, attn_implementation="eager").eval()
            self.processor.tokenizer.set_prefix_tokens(language="ar", task="transcribe", predict_timestamps=False)
            heads = self.model.generation_config.alignment_heads
            if not heads or any(not (0 <= layer < self.model.config.decoder_layers
                                     and 0 <= head < self.model.config.decoder_attention_heads)
                                for layer, head in heads):
                raise ValueError("Некорректные alignment heads Whisper.")
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
            ids = tokenizer.encode(" " + whisper_alignment_text(word).replace("ٱ", "ا"), add_special_tokens=False)
            first = 0
            while first < len(ids) and not tokenizer.decode([ids[first]]).strip():
                first += 1
            if first == len(ids):
                raise ValueError("Пустое слово Whisper.")
            spans.append((len(tokens) + first, len(tokens) + len(ids)))
            tokens.extend(ids)
        prefix = tokenizer.prefix_tokens
        if len(prefix) + len(tokens) + 1 > self.model.config.max_target_positions:
            raise WhisperTextOverflow("Слишком много текста для одного окна Whisper.")
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

    def whisper_local(self, pcm, total_samples):
        """A bounded ASR phrase only; never call the FULL/tiled aligner here."""
        if (self.model_id not in ("whisper-tarteel-tiny", "whisper-tarteel-base") or pcm.ndim != 1
                or not SR * .3 <= len(pcm) <= SR * 6 or len(pcm) >= total_samples):
            raise ValueError("FAST Whisper требует локальный участок до 6 секунд.")
        torch = self.torch
        processed = self.processor(pcm, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            generated = self.model.generate(
                processed.input_features, attention_mask=processed.attention_mask,
                language="ar", task="transcribe", return_timestamps=False,
                max_new_tokens=64, do_sample=False, num_beams=1, use_cache=True,
                return_dict_in_generate=True, output_scores=True,
            )
        sequence = generated.sequences[0]
        scores = generated.scores
        selected = sequence[-len(scores):]
        log_probs = [score[0].log_softmax(-1)[token] for score, token in zip(scores, selected)]
        confidence = float(torch.stack(log_probs).mean().exp()) if log_probs else 0.0
        text = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
        minimum = min((float(p.exp()) for token, p in zip(selected, log_probs)
                       if int(token) not in self.processor.tokenizer.all_special_ids), default=0.0)
        tokenizer = self.processor.tokenizer
        # Retain the exact selected BPE bytes, including tokens split inside a
        # UTF-8 character or spanning a space. Word attribution is performed by
        # the local adapter; no expected Quran text participates in this decode.
        from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
        byte_decoder = {character: value for value, character in bytes_to_unicode().items()}
        selected_tokens = []
        for token, probability in zip(selected, log_probs):
            token_id = int(token)
            if token_id in tokenizer.all_special_ids:
                continue
            encoded = tokenizer.convert_ids_to_tokens(token_id)
            selected_tokens.append({"id": token_id, "log_probability": float(probability),
                                    "bytes": list(bytes(byte_decoder[c] for c in encoded))})
        return {"text": text, "token_confidence": confidence,
                "min_token_confidence": minimum,
                "selected_tokens": selected_tokens,
                "truncated": int(sequence[-1]) != self.processor.tokenizer.eos_token_id}

    def whisper_locate(self, pcm, words):
        """ASR locates a known prefix; it never supplies the timing text."""
        processed = self.processor(pcm, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
        with self.torch.inference_mode():
            generated = self.model.generate(
                processed.input_features, attention_mask=processed.attention_mask,
                language="ar", task="transcribe", return_timestamps=True,
                max_new_tokens=400, do_sample=False, num_beams=1,
            )
        hypothesis = self.processor.batch_decode(generated, skip_special_tokens=True)[0].split()
        observed = [plain(w) for w in hypothesis]
        # A repeated phrase hundreds of words ahead is not evidence that those
        # intervening words were read in this short window. Match a local prefix.
        canonical, owners = [], []
        for index, word in enumerate(words[:min(180, max(12, len(observed) * 2))]):
            units = plain(word).split()
            canonical.extend(units)
            owners.extend([index] * len(units))
        blocks = difflib.SequenceMatcher(None, canonical, observed, autojunk=False).get_matching_blocks()
        matched = sum(b.size for b in blocks)
        longest = max((b.size for b in blocks), default=0)
        if (not observed or matched < max(min(2, len(words)), len(observed) * 0.5)
                or (len(observed) >= 8 and longest < 3)):
            return None
        last = max(b.a + b.size for b in blocks if b.size) - 1
        end = owners[last] + 1
        # Do not consume a canonical word when ASR only matched its first unit.
        if last + 1 < len(owners) and owners[last + 1] == owners[last]:
            return None
        prefix_words = max((owners[b.a + b.size - 1] + 1 for b in blocks if b.a == 0 and b.size), default=0)
        return end, prefix_words

    def whisper_pause_retry(self, pcm, words, rows, offset, window_samples, strict_words):
        """Confirm both sides of a DSP pause before consuming an early ASR end."""
        clip = pcm[offset:offset + window_samples]
        candidates = [(start, end) for start, end in whisper_energy_pauses(clip)
                      if rows[-1][0] + 200 < start and end < len(clip) * 1000 / SR - 200][:3]
        for start, finish in candidates:
            split = round((start + finish) * SR / 2000)
            located = self.whisper_locate(clip[:split], words)
            if located is None or located[0] != len(rows):
                continue
            end = located[0]
            if end == len(words):
                # An early EOS cannot claim that the rest of the recording is silence.
                continue
            next_offset = offset + split
            following = self.whisper_locate(pcm[next_offset:next_offset + window_samples], words[end:])
            if following is None or following[1] < min(2, len(words) - end):
                continue
            confirmed = self.whisper_clip(clip[:split], words[:end], strict_words=strict_words)
            if len(confirmed) != end or any(ms >= start for ms, _ in confirmed):
                continue
            print(f"Whisper FULL pause retry: {offset / SR:.3f}-{next_offset / SR:.3f}; "
                  f"confirmed={end}; next-prefix={following[0]}", file=sys.stderr)
            return confirmed, end, next_offset
        return None

    def whisper_align(self, pcm, words, strict_words=True):
        window_s = 30
        if len(pcm) <= window_s * SR:
            return self.whisper_clip(pcm, words, strict_words=strict_words)
        # ASR is only a coarse locator for bounded windows. All final timing
        # rows come from teacher forcing the original canonical word sequence.
        result, cursor, offset = [], 0, 0
        while cursor < len(words):
            clip = pcm[offset:offset + window_s * SR]
            if len(clip) < SR // 10:
                raise ValueError("Недостаточно аудио для оставшегося текста Whisper.")
            located = self.whisper_locate(clip, words[cursor:])
            if located is None:
                if window_s > 10:
                    window_s -= 10
                    continue
                raise ValueError("Whisper не нашёл достаточных опор в каноническом тексте.")
            end = located[0]
            segment_words = words[cursor:cursor + end]
            try:
                rows = self.whisper_clip(clip, segment_words, strict_words=strict_words)
            except WhisperTextOverflow:
                # Re-locate the same known sequence in a shorter audio window.
                # Never truncate the target words or bypass anchor validation.
                if window_s <= 10:
                    raise
                window_s -= 10
                continue
            final = offset + len(clip) == len(pcm)
            # The locator may emit EOS at an internal pause. Validate the short
            # phrase and the next known prefix, rather than losing words to the
            # usual overlap or pretending that ASR covered the entire window.
            if rows and rows[-1][0] < (len(clip) * 1000 / SR - 6000):
                retry = self.whisper_pause_retry(
                    pcm, words[cursor:], rows, offset, window_s * SR, strict_words)
                if retry is not None:
                    confirmed, consumed, next_offset = retry
                    base_ms = offset * 1000 // SR
                    result.extend((base_ms + ms, conf) for ms, conf in confirmed)
                    cursor += consumed
                    offset = next_offset
                    window_s = 30
                    continue
            if final and cursor + end != len(words):
                if window_s > 10 and len(clip) > 10 * SR:
                    window_s = 10
                    continue
                raise ValueError("Whisper не подтвердил весь конец канонического текста.")
            commit = end if final else next((i for i, row in enumerate(rows)
                                            if row[0] >= (window_s - 6) * 1000), end - 2)
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
            window_s = 30
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
        folder = Path(folder)
        # Freeze assets before loading, then reject mutation during loading.
        assets = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix in
                  {".bin", ".safetensors", ".onnx", ".json", ".txt", ".model"}}
        acoustic = Acoustic(model_id, folder)
        if any(hashlib.sha256((folder / p).read_bytes()).hexdigest() != sha for p, sha in assets.items()):
            raise ValueError("Model assets changed during loading")
        versions = {p: importlib.metadata.version(p) for p in
                    ("torch", "transformers", "numpy")}
        versions.update(python=sys.version, unicode=unicodedata.unidata_version)
        for package in ("torchaudio", "quran-transcript", "uroman", "onnxruntime", "nemo-toolkit", "sentencepiece"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        identity = {"model_id": model_id, "assets": assets, "versions": versions,
                    "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "vocab": acoustic.vocab, "blank_id": acoustic.blank_id}
        if acoustic.text_kind == "whisper":
            identity.update(config=acoustic.model.config.to_dict(),
                            generation_config=acoustic.model.generation_config.to_dict(),
                            tokenizer_vocab_sha256=hashlib.sha256(json.dumps(
                                acoustic.processor.tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
                            tokenizer_special=acoustic.processor.tokenizer.special_tokens_map,
                            tokenizer_prefix=acoustic.processor.tokenizer.prefix_tokens,
                            feature_extractor=acoustic.processor.feature_extractor.to_dict(),
                            local_generation={"language": "ar", "task": "transcribe", "return_timestamps": False,
                                              "max_new_tokens": 64, "do_sample": False, "num_beams": 1})
        reply({"vocab": acoustic.vocab, "blank_id": acoustic.blank_id,
               "text_kind": acoustic.text_kind, "versions": versions, "runtime_identity": identity})
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
            elif action == "recognize_local":
                pcm = np.load(work / "audio.npy", allow_pickle=False)
                reply(acoustic.whisper_local(pcm, request["total_samples"]))
            elif action == "probe":
                # A real encoder/decoder forward; silence has no usable word
                # boundary evidence, so the probe does not run timing validation.
                torch = acoustic.torch
                features = acoustic.processor(np.zeros(SR, np.float32), sampling_rate=SR, return_tensors="pt").input_features
                with torch.inference_mode():
                    encoded = acoustic.model.model.encoder(features, output_attentions=False, return_dict=True)
                    out = acoustic.model(encoder_outputs=encoded,
                                         decoder_input_ids=torch.tensor([acoustic.processor.tokenizer.prefix_tokens]),
                                         output_attentions=True, use_cache=False)
                heads = acoustic.model.generation_config.alignment_heads
                if (not torch.isfinite(out.logits).all()
                        or any(not torch.isfinite(out.cross_attentions[layer][:, head]).all()
                               for layer, head in heads)):
                    raise ValueError("Некорректный выход Whisper.")
                reply({"ok": True})
            else:
                raise ValueError("Unknown runtime action")
        except Exception as exc:
            traceback.print_exc()
            reply({"error": str(exc)})


if __name__ == "__main__":
    main()
