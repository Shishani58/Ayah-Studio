"""Read-only benchmark of installed acoustic models against a timing reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from surah_cutter.core import align, decode, model_store, word_align
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--surahs", nargs="+", type=int, default=[114, 112])
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = []
    with sqlite3.connect(f"file:{args.reference.resolve()}?mode=ro", uri=True) as reference:
        for model_id in args.models:
            model_store.ACTIVE_ID = model_id
            align._ONNX_CACHE.clear()
            release_runtimes()
            for surah in args.surahs:
                item = {"model": model_id, "surah": surah}
                started = time.monotonic()
                try:
                    audio = args.audio_dir / f"{surah:03d}.mp3"
                    item["audio_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
                    item["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
                    pcm = decode.decode(str(audio))
                    words = quran_text.surah_words(surah)
                    bundle = align._get_onnx_bundle()
                    if hasattr(bundle.emitter, "probe"):
                        bundle.emitter.probe()
                    trace = {}
                    rows = align.align_surah(pcm, 16000, words, model=bundle, trace=trace)
                    assert [a for a, _, _ in rows] == list(range(1, quran_text.ayah_count(surah) + 1))
                    assert all(b[1] > a[1] for a, b in zip(rows, rows[1:]))
                    truth = dict(reference.execute("SELECT ayah,time FROM timings WHERE sura=?", (surah,)))
                    errors = [abs(ms - truth[a]) for a, ms, _ in rows]
                    item.update({"rows": rows, "absolute_errors_ms": errors,
                                 "median_ms": float(np.median(errors)),
                                 "p95_ms": float(np.percentile(errors, 95)),
                                 "max_ms": max(errors), "source": model_store.source(), "trace": trace})
                    end = truth.get(2, len(pcm) // 16)
                    word_rows = word_align.align_ayah_words(pcm[truth[1] * 16:end * 16], 16000,
                                                           [w for w in words if w.ayah == 1])
                    item["first_ayah_words"] = word_rows
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                item["seconds"] = round(time.monotonic() - started, 3)
                report.append(item)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps({k: v for k, v in item.items() if k not in ("trace", "rows")},
                                 ensure_ascii=False), flush=True)
    release_runtimes()


if __name__ == "__main__":
    main()
