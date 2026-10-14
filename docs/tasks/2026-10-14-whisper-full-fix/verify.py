"""Run installed FULL Whisper without writing project/settings databases."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from surah_cutter.core import model_store
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.core.model_worker import Acoustic, BASMALA, SR
from surah_cutter.data.quran_text import surah_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcm", type=Path, required=True)
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--model", default="whisper-tarteel-tiny")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pcm = np.load(args.pcm, allow_pickle=False)
    words = surah_words(args.surah)
    prefix = BASMALA if args.surah not in (1, 9) else []
    rendered = prefix + inference_words(words, args.model)
    model_store.ACTIVE_ID = args.model
    started = time.monotonic()
    acoustic = Acoustic(args.model, model_store.installed_path(model_store.spec(args.model)))
    report = {"surah": args.surah, "source": model_store.source(mode="full"),
              "duration_seconds": len(pcm) / SR,
              "pcm_sha256": hashlib.sha256(memoryview(pcm)).hexdigest(), "reads": []}
    original_clip = acoustic.whisper_clip
    original_locate = acoustic.whisper_locate
    original_decode = acoustic.processor.batch_decode
    last_text = []
    report["asr_reads"] = []

    def capture_decode(*args, **kwargs):
        decoded = original_decode(*args, **kwargs)
        last_text[:] = decoded
        return decoded

    def capture_locate(clip, targets):
        located = original_locate(clip, targets)
        offset = (clip.ctypes.data - pcm.ctypes.data) // pcm.itemsize
        report["asr_reads"].append({"offset_seconds": offset / SR, "duration_seconds": len(clip) / SR,
                                    "text": last_text[0], "end": located[0] if located else None,
                                    "known_prefix": targets[:40]})
        return located

    acoustic.processor.batch_decode = capture_decode
    acoustic.whisper_locate = capture_locate

    def capture(clip, targets, **kwargs):
        parent = sys._getframe(1).f_locals
        offset = parent.get("offset", 0)
        item = {"offset_seconds": offset / SR, "duration_seconds": len(clip) / SR,
                "word_count": len(targets), "first": targets[0], "last": targets[-1]}
        report["reads"].append(item)
        if len(report["reads"]) % 10 == 0:
            print(f"FULL {args.model}: reads={len(report['reads'])}, offset={offset / SR:.2f}s, "
                  f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        return original_clip(clip, targets, **kwargs)

    acoustic.whisper_clip = capture
    try:
        rows = acoustic.whisper_align(pcm, rendered, strict_words=False)
        assert len(rows) == len(rendered)
        rows = rows[len(prefix):]
        ayahs = {}
        for word, row in zip(words, rows):
            ayahs.setdefault(word.ayah, (word.ayah, *row))
        boundaries = list(ayahs.values())
        assert all(b[1] > a[1] for a, b in zip(boundaries, boundaries[1:])), "ayah starts conflict"
        assert all(0 <= row[0] < len(pcm) * 1000 / SR for row in rows)
        report.update(word_rows=rows, ayah_rows=boundaries, ok=True)
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("reads", "word_rows", "ayah_rows", "asr_reads")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
