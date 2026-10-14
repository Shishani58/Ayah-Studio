"""Read-only first-window diagnosis using the installed Whisper runtime."""
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from surah_cutter.core import model_store
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.core.model_worker import Acoustic, BASMALA, SR, plain
from surah_cutter.data.quran_text import surah_words


def raw_probe(acoustic, clip):
    features = acoustic.processor(clip, sampling_rate=SR, return_tensors="pt").input_features
    with acoustic.torch.inference_mode():
        generated = acoustic.model.generate(
            features, language="ar", task="transcribe", return_timestamps=False,
            max_new_tokens=400, do_sample=False, num_beams=1,
            force_unique_generate_call=True, return_dict_in_generate=True,
        )
    return {"token_ids": generated.sequences[0].tolist(),
            "text_with_special_tokens": acoustic.processor.batch_decode(
                generated.sequences, skip_special_tokens=False)[0]}


def main():
    pcm = np.load(sys.argv[1], allow_pickle=False)
    output = Path(sys.argv[2])
    model_id = "whisper-tarteel-tiny"
    words = BASMALA + inference_words(surah_words(2), model_id)
    acoustic = Acoustic(model_id, model_store.installed_path(model_store.spec(model_id)))
    report = {
        "model": model_id, "revision": model_store.spec(model_id)["revision"],
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "input_seconds": len(pcm) / SR,
        "windows": [],
    }

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    original_clip = acoustic.whisper_clip

    def diagnostic_clip(clip, segment_words, strict_words=True):
        parent = sys._getframe(1)
        if parent.f_code is Acoustic.whisper_align.__code__:
            state = parent.f_locals
            if len(report["windows"]) >= 6:
                raise RuntimeError("Diagnostic stop: six FULL windows inspected")
            item = {k: state[k] for k in ("cursor", "offset", "window_s", "hypothesis", "matched", "end") if k in state}
            item["segment_words"] = segment_words
            rows = original_clip(clip, segment_words, strict_words=strict_words)
            item["rows"] = rows
            item["commit_nonfinal"] = next((i for i, row in enumerate(rows)
                                            if row[0] >= (state["window_s"] - 6) * 1000), state["end"] - 2)
            report["windows"].append(item)
            save()
            print(json.dumps(item, ensure_ascii=False), flush=True)
            return rows
        return original_clip(clip, segment_words, strict_words=strict_words)

    acoustic.whisper_clip = diagnostic_clip

    def trace(frame, event, arg):
        if frame.f_code is Acoustic.whisper_align.__code__ and event == "exception":
            state = frame.f_locals
            report["failure_state"] = {
                k: state[k] for k in ("cursor", "offset", "window_s", "hypothesis", "canonical",
                                     "observed", "matched", "longest", "end", "segment_words", "rows", "final", "commit")
                if k in state
            }
        return trace

    started = time.monotonic()
    sys.settrace(trace)
    try:
        result = acoustic.whisper_align(pcm, words, strict_words=False)
        report["original_result"] = result
    except Exception as exc:
        report["original_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sys.settrace(None)
        report["original_seconds"] = time.monotonic() - started
        save()
    state = report.get("failure_state", {})
    print(json.dumps({k: v for k, v in report.items() if k not in ("failure_state", "windows")}, ensure_ascii=False), flush=True)
    print(json.dumps(state, ensure_ascii=False), flush=True)

    # Same audio and weights; only the ASR locator's timestamp/mask flags differ.
    experiment_offset = state.get("offset", 0) if len(pcm) > 31 * SR else 0
    experiment_cursor = state.get("cursor", 0) if len(pcm) > 31 * SR else 0
    clip = pcm[experiment_offset:experiment_offset + 30 * SR]
    target_words = words[experiment_cursor:]
    processed = acoustic.processor(clip, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
    experiments = []
    for timestamps, mask in ((True, False), (True, True), (False, True)):
        started = time.monotonic()
        with acoustic.torch.inference_mode():
            generated = acoustic.model.generate(
                processed.input_features, attention_mask=processed.attention_mask if mask else None,
                language="ar", task="transcribe", return_timestamps=timestamps,
                max_new_tokens=400, do_sample=False, num_beams=1,
            )
        text = acoustic.processor.batch_decode(generated, skip_special_tokens=True)[0]
        observed = [plain(w) for w in text.split()]
        canonical = [plain(w) for w in target_words[:180]]
        blocks = difflib.SequenceMatcher(None, canonical, observed, autojunk=False).get_matching_blocks()
        end = max((b.a + b.size for b in blocks if b.size), default=0)
        item = {"timestamps": timestamps, "attention_mask": mask, "text": text,
                "offset_seconds": experiment_offset / SR, "cursor": experiment_cursor,
                "token_ids": generated[0].tolist(),
                "blocks": [(b.a, b.b, b.size) for b in blocks], "end": end,
                "asr_seconds": time.monotonic() - started}
        if end:
            rows = acoustic.whisper_clip(clip, target_words[:end], strict_words=False)
            item.update(rows=rows, segment_words=target_words[:end],
                        commit=next((i for i, row in enumerate(rows) if row[0] >= 24000), end - 2))
        experiments.append(item)
        report["experiments"] = experiments
        save()
        print(json.dumps(item, ensure_ascii=False), flush=True)

    if experiment_offset:
        # Test the observed pause independently, preserving the known word order.
        split_experiments = []
        for start_s, end_s, cursor in ((117.88, 123.9, 81), (123.9, 153.9, 84)):
            clip = pcm[round(start_s * SR):round(end_s * SR)]
            features = acoustic.processor(clip, sampling_rate=SR, return_tensors="pt").input_features
            with acoustic.torch.inference_mode():
                generated = acoustic.model.generate(
                    features, language="ar", task="transcribe", return_timestamps=True,
                    max_new_tokens=400, do_sample=False, num_beams=1,
                )
            text = acoustic.processor.batch_decode(generated, skip_special_tokens=True)[0]
            observed = [plain(w) for w in text.split()]
            canonical = [plain(w) for w in words[cursor:cursor + 180]]
            blocks = difflib.SequenceMatcher(None, canonical, observed, autojunk=False).get_matching_blocks()
            end = max((b.a + b.size for b in blocks if b.size), default=0)
            item = {"interval_seconds": [start_s, end_s], "cursor": cursor,
                    "text": text, "end": end, "blocks": [(b.a, b.b, b.size) for b in blocks]}
            if end:
                item["rows"] = original_clip(clip, words[cursor:cursor + end], strict_words=False)
                item["segment_words"] = words[cursor:cursor + end]
            split_experiments.append(item)
            report["split_experiments"] = split_experiments
            save()
            print(json.dumps(item, ensure_ascii=False), flush=True)

        report["rms_by_second"] = {str(s): float(np.sqrt(np.mean(pcm[s * SR:(s + 1) * SR] ** 2)))
                                   for s in range(118, 148)}
        report["raw_failure_generation"] = raw_probe(
            acoustic, pcm[experiment_offset:experiment_offset + 30 * SR])
        save()


if __name__ == "__main__":
    main()
