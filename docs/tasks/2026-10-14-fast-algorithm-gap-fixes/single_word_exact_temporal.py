"""Fresh local timing experiments. No production capability/settings change."""

import hashlib
import json
from pathlib import Path
from time import monotonic
import numpy as np
from surah_cutter.core import model_store
from surah_cutter.core.model_worker import Acoustic, SR, whisper_alignment_text
from surah_cutter.core.fast_reads import LocalAcousticReads
from surah_cutter.core.fast_adapters import arabic_units, attributed_asr_units
from surah_cutter.data import quran_text
from transformers.models.whisper.generation_whisper import _dynamic_time_warping

ROOT = Path(__file__).parent
original = np.load("/tmp/fast-gap-timing-108.npy", allow_pickle=False)
source = original[5720 * 16 :]
words = [w.text for w in quran_text.surah_words(108)[:1]]
output = []
for model_id in ("whisper-tarteel-tiny", "whisper-tarteel-base"):
    acoustic = Acoustic(model_id, model_store.installed_path(model_store.spec(model_id)))
    torch, tokenizer = acoustic.torch, acoustic.processor.tokenizer
    runtime = {
        "model": model_id,
        "weights_sha256": hashlib.sha256(
            (
                model_store.installed_path(model_store.spec(model_id)) / "pytorch_model.bin"
            ).read_bytes()
        ).hexdigest(),
        "heads": acoustic.model.generation_config.alignment_heads,
    }
    for silence in (0, 200):
        pcm = np.r_[np.zeros(silence * 16, np.float32), source]
        reads = LocalAcousticReads(pcm, 16000, None)
        records = []

        def charged(fn, phase):
            s0, s1 = reads._window(0, 3000 + silence)
            permit = reads._permit(s0, s1)
            assert permit is not None
            attempt = reads._begin_read(s0, s1, permit, phase=phase)
            local = np.ascontiguousarray(pcm[s0:s1], np.float32)
            records.append(
                {
                    "phase": phase,
                    "samples": len(local),
                    "sha256": hashlib.sha256(memoryview(local)).hexdigest(),
                }
            )
            begin = monotonic()
            try:
                result = fn(local)
                attempt["outcome"] = "completed"
                return result
            except BaseException:
                attempt["outcome"] = "failed"
                raise
            finally:
                attempt["seconds"] = monotonic() - begin

        asr = charged(lambda local: acoustic.whisper_local(local, len(pcm)), "independent_asr")
        candidates = {}

        def forward_candidates(local, target):
            tokens = []
            spans = []
            if target is None:
                tokens = [t["id"] for t in asr["selected_tokens"]]
                _, _, attribution = attributed_asr_units(asr)
                attribution = attribution[:1]
                tokens = tokens[: max(attribution[0]["selected_indices"]) + 1]
                spans = [
                    (min(a["selected_indices"]), max(a["selected_indices"]) + 1)
                    for a in attribution
                ]
            for word in target or ():
                ids = tokenizer.encode(
                    " " + whisper_alignment_text(word).replace("ٱ", "ا"), add_special_tokens=False
                )
                first = 0
                while first < len(ids) and not tokenizer.decode([ids[first]]).strip():
                    first += 1
                spans.append((len(tokens) + first, len(tokens) + len(ids)))
                tokens.extend(ids)
            prefix = tokenizer.prefix_tokens
            features = acoustic.processor(
                local, sampling_rate=SR, return_tensors="pt"
            ).input_features
            with torch.inference_mode():
                encoded = acoustic.model.model.encoder(
                    features, output_attentions=False, return_dict=True
                )
                out = acoustic.model(
                    encoder_outputs=encoded,
                    decoder_input_ids=torch.tensor([[*prefix, *tokens, tokenizer.eos_token_id]]),
                    output_attentions=True,
                    use_cache=False,
                )
            frames = len(local) // 320
            heads = acoustic.model.generation_config.alignment_heads
            raw = torch.stack(
                [out.cross_attentions[layer][0, head, :, :frames] for layer, head in heads]
            ).numpy()
            results = {}
            for norm in ("token_axis", "time_axis"):
                axis = -2 if norm == "token_axis" else -1
                weights = (raw - raw.mean(axis=axis, keepdims=True)) / np.maximum(
                    raw.std(axis=axis, keepdims=True), 1e-6
                )
                padded = np.pad(weights, ((0, 0), (0, 0), (3, 3)), mode="reflect")
                weights = np.median(
                    np.lib.stride_tricks.sliding_window_view(padded, 7, axis=-1), axis=-1
                )
                # Test both prefix absorber and text-only query conventions.
                for absorber in (True, False):
                    offset = len(prefix) - 1 if absorber else len(prefix)
                    matrix = weights.mean(axis=0)[offset:-1]
                    ti, ai = _dynamic_time_warping(-matrix)
                    jumps = np.pad(np.diff(ti), (1, 0), constant_values=1).astype(bool)
                    times = ai[jumps] * 20
                    shift = 1 if absorber else 0
                    results[f"{norm}_absorber{absorber}"] = [
                        int(times[min(start + shift, len(times) - 1)]) for start, end in spans
                    ]
            # Text query peaks are another candidate, not a production estimator.
            results["attention_peak"] = [
                int(raw.mean(axis=0)[len(prefix) + start].argmax()) * 20 for start, end in spans
            ]
            return results

        candidates["heard_target"] = charged(
            lambda local: forward_candidates(local, words), "candidate_alignment_same_local_crop"
        )
        candidates["exact_selected_tokens"] = charged(
            lambda local: forward_candidates(local, None),
            "exact_independently_selected_BPE_alignment",
        )
        # This negative deliberately bypasses ownership, documenting why forced
        # timestamps alone cannot be exposed as evidence. Runtime stays False.
        candidates["wrong_target"] = charged(
            lambda local: forward_candidates(local, ["قُلْ", "هُوَ"]), "wrong_text_negative"
        )
        output.append(
            {
                "runtime": runtime,
                "silence_ms": silence,
                "asr": asr,
                "expected_words": words,
                "independently_heard": arabic_units(asr["text"])[:1]
                == tuple(arabic_units(" ".join(words))),
                "candidates": candidates,
                "ledger": reads.diagnostics(),
                "actual_dispatches": records,
            }
        )
        print(model_id, silence, candidates, flush=True)
(ROOT / "single-word-exact-temporal.json").write_text(
    json.dumps(
        {
            "results": output,
            "known_word_onsets": "No independent listening annotations available; shift controls do not establish accuracy.",
            "capability_enabled": False,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
