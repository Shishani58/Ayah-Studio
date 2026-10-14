"""Read-only FULL experiment on local Shatri recordings.

Example (separate process per case so peak RSS and ONNX pools are comparable):
  .venv/bin/python tools/bench_full_windows.py --suras 78 80 98 112 114
  .venv/bin/python tools/bench_full_windows.py --cases fixed:30:1:0 pause:20:2:2

Case format: kind:window_seconds:parallel_workers:intra_op_threads (0 = ORT default).
Kind strong cuts only at signal-confirmed strong pauses, without fixed fallback.
Results contain all candidate timings, flags, input/source hashes and real reads.
No project DB is opened for writing and no model settings are saved.
Use --reference for another recording and --production-guards for current FULL checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surah_cutter.core import align, decode, emissions, full_windows  # noqa: E402
from surah_cutter.data import quran_text, references  # noqa: E402

DEFAULT_CASES = ["fixed:30:1:0", "fixed:30:1:2"] + [
    f"pause:{window}:{workers}:2" for window in (10, 20, 30) for workers in (1, 2, 4)
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    array = np.abs(values)
    if not len(array):
        return {"n": 0}
    return {"n": len(array), "median_ms": float(np.median(array)),
            "p95_ms": float(np.percentile(array, 95)), "max_ms": float(np.max(array)),
            "over500": int(np.sum(array > 500))}


def score_surahs(items, reference):
    """Score completed predictions; the selected timing reference is never a model input."""
    for item in items:
        surah = item["surah"]
        errors = [ms - reference[(surah, ayah)] for ayah, ms, _ in item["rows"]]
        cuts = [w["end_ms"] for w in item["windows"].get("windows", [])[:-1]]
        item.update(errors_ms=errors, stats=stats(errors), seam_errors_ms=[
            error for (_, ms, _), error in zip(item["rows"], errors)
            if any(abs(ms - cut) <= 2000 for cut in cuts)
        ])


def case_result(args):
    import onnxruntime as ort

    kind, window, workers, threads = args.case.split(":")
    window, workers, threads = int(window), int(workers), int(threads)
    if kind not in ("fixed", "pause", "strong") or not 1 <= workers <= 6 or threads < 0 or window < 1:
        raise ValueError("Invalid benchmark case")
    if kind == "fixed" and workers != 1:
        raise ValueError("Fixed baseline uses the original sequential emitter")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    vocab, normalize = emissions._load_sidecar(args.model, None)
    emitter = emissions.OnnxEmitter(session, vocab, blank_id=emissions._resolve_blank_id(vocab),
                                   do_normalize=normalize)
    load_s = time.perf_counter() - started
    result = {"case": args.case, "load_s": load_s, "surahs": [], "failures": [],
              "reference": args.reference, "production_guards": args.production_guards,
              "onnxruntime": ort.__version__, "model_sha256": sha256(args.model),
              "vocab_sha256": sha256(args.model.with_suffix(args.model.suffix + ".vocab.json")),
              "meta_sha256": sha256(args.model.with_suffix(args.model.suffix + ".meta.json")),
              "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                       "python": sys.version}}
    warmed = False
    for repeat in range(args.repeats):
        for surah in args.suras:
            path = args.audio_dir / f"{surah:03d}.mp3"
            started = time.perf_counter()
            pcm = decode.decode(path)
            decode_s = time.perf_counter() - started
            words = quran_text.surah_words(surah)
            if not warmed:
                emitter.emissions(pcm[:16_000])
                warmed = True
            try:
                window_trace = {}
                started = time.perf_counter()
                if kind in ("pause", "strong"):
                    window_options = ({"max_window_s": full_windows.FULL_MAX_WINDOW_S}
                                      if kind == "strong" and args.production_guards else {})
                    matrix = full_windows.generate_pause_emissions(
                        emitter, pcm, window_s=window, workers=workers, trace=window_trace,
                        strong_pauses=kind == "strong",
                        **window_options,
                    )
                else:
                    matrix = emitter.emissions(pcm, window_s=window)
                emissions_s = time.perf_counter() - started
                # Use the ordinary FULL/oversized-tiled checks on the measured
                # matrix. A proxy prevents recomputing inference during CTC.
                bundle = align._OnnxBundle(SimpleNamespace(
                    vocab=emitter.vocab, blank_id=emitter.blank_id, emissions=lambda _pcm: matrix,
                ))
                alignment_trace = {}
                started = time.perf_counter()
                rows = align.align_surah(pcm, 16_000, words, model=bundle, surah=surah,
                                         trace=alignment_trace)
                raw_review = [c["ayah"] for c in alignment_trace.get("ayahs", [])
                              if not c["reliable"]]
                if kind == "strong" and args.production_guards:
                    window_trace.update(algorithm=full_windows.FULL_POLICY,
                                        review_policy=full_windows.FULL_REVIEW_POLICY)
                    align._review_pause_full_onsets(alignment_trace.get("ayahs", []), window_trace)
                alignment_s = time.perf_counter() - started
                if ([a for a, _, _ in rows] != list(range(1, quran_text.ayah_count(surah) + 1))
                        or any(b[1] <= a[1] for a, b in zip(rows, rows[1:]))):
                    raise ValueError("Canonical ayah sequence/order violation")
                item = {"surah": surah, "repeat": repeat, "audio_sha256": sha256(path),
                        "duration_s": len(pcm) / 16_000, "decode_s": decode_s,
                        "emissions_s": emissions_s, "alignment_s": alignment_s,
                        "full_s": emissions_s + alignment_s,
                        "emissions_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                        "frames": len(matrix), "windows": window_trace,
                        "rows": rows, "raw_review": raw_review,
                        "onset_checks": alignment_trace.get("ayahs", []),
                        "review": [c["ayah"] for c in alignment_trace.get("ayahs", [])
                                   if not c["reliable"]]}
                result["surahs"].append(item)
                print(f"{args.case} surah={surah} repeat={repeat + 1} "
                      f"FULL={item['full_s']:.2f}s ayahs={len(rows)}", flush=True)
            except Exception as exc:
                result["failures"].append({"surah": surah, "repeat": repeat,
                                           "error": f"{type(exc).__name__}: {exc}"})
                print(f"{args.case} surah={surah} FAILED: {exc}", flush=True)
    # Open the selected ground truth only after all predictions in this process.
    reference_path = references._resolve_path(args.reference, None)
    reference = references.load_reference(reference_path)
    score_surahs(result["surahs"], reference)
    result["reference_sha256"] = sha256(reference_path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mib"] = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    result["errors"] = stats([error for item in result["surahs"] for error in item["errors_ms"]])
    result["full_s"] = sum(item["full_s"] for item in result["surahs"])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path,
                        default=Path.home() / "Desktop/Abu_Bakr_Ash-Shatri")
    parser.add_argument("--model", type=Path, default=ROOT / "models/wav2vec2-quran-phonetics.onnx")
    parser.add_argument("--reference", default="abu_bakr_ash_shatri",
                        help="Timing reference name or matching recording's .db path")
    parser.add_argument("--production-guards", action="store_true",
                        help="Use current FULL memory/review guards for strong cases")
    parser.add_argument("--suras", type=int, nargs="+", default=[78, 80, 98, 112, 114])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/tasks/2026-10-14-full-pause-windows/benchmark.json")
    args = parser.parse_args()
    if args.repeats < 1 or any(not 1 <= surah <= 114 for surah in args.suras):
        parser.error("repeats must be positive; suras must be 1..114")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.case:
        case_result(args)
        return
    result = {"cases": [], "settings": {"suras": args.suras, "repeats": args.repeats,
                                          "context_s": 2, "batch_size": 1,
                                          "reference": args.reference,
                                          "production_guards": args.production_guards},
              "sources": {str(p.relative_to(ROOT)): sha256(p) for p in [
                  ROOT / "surah_cutter/core/emissions.py", ROOT / "surah_cutter/core/full_windows.py",
                  ROOT / "surah_cutter/core/align.py", ROOT / "surah_cutter/core/_ctc.py",
                  ROOT / "surah_cutter/core/candidates.py", ROOT / "tools/bench_full_windows.py",
                  ROOT / "data/quran_mushaf.db", ROOT / "data/word_tr_cache.json",
              ]}}
    for index, case in enumerate(args.cases):
        child_output = args.output.with_name(
            args.output.stem + f"-{index + 1}-" + case.replace(":", "-") + ".json",
        )
        command = [sys.executable, str(Path(__file__).resolve()), "--case", case,
                        "--audio-dir", str(args.audio_dir), "--model", str(args.model),
                        "--reference", args.reference,
                        "--suras", *map(str, args.suras), "--repeats", str(args.repeats),
                        "--output", str(child_output)]
        if args.production_guards:
            command.append("--production-guards")
        subprocess.run(command, check=True)
        result["cases"].append(json.loads(child_output.read_text()))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for case in result["cases"]:
        print(f"{case['case']}: FULL={case['full_s']:.2f}s "
              f"RSS={case['peak_rss_mib']:.0f}MiB {case['errors']} failures={case['failures']}")


if __name__ == "__main__":
    main()
