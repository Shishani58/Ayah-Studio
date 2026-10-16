"""Read-only local FAST comparison, with shared DSP/VAD candidates and late evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import sqlite3
import tempfile
import time
from unittest.mock import patch

import numpy as np

from surah_cutter.core import align, align_fast, candidates, decode, model_store
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def dispatch_spy(bundle, pcm):
    """Measure actual local PCM independently of executor-provided diagnostics."""
    calls = []
    def record(local):
        assert 0 < len(local) <= 96000 and len(local) < len(pcm)
        calls.append({"samples": len(local), "sha256": hashlib.sha256(
            memoryview(np.ascontiguousarray(local, dtype=np.float32))).hexdigest()})
    # RuntimeEmitter exposes recognize_local for every model family. Muaalem
    # and other CTC runtimes still call emissions, never the Whisper method.
    if (getattr(bundle.emitter, "model_id", "").startswith("whisper-")
            and hasattr(bundle.emitter, "recognize_local")):
        cls = type(bundle.emitter)
        original = cls.recognize_local
        def recognize(emitter, local, **kwargs):
            record(local)
            return original(emitter, local, **kwargs)
        with patch.object(cls, "recognize_local", recognize):
            yield calls
    else:
        original = align._emissions
        def emissions(selected, local, np_module):
            record(local)
            return original(selected, local, np_module)
        with patch.object(align, "_emissions", emissions):
            yield calls


def verify_dispatch(calls, pcm, trace):
    assert len(calls) == len(trace["acoustic_reads"])
    for actual, reported in zip(calls, trace["acoustic_reads"]):
        left, right = [round(t * 16) for t in reported["window_ms"]]
        assert actual["samples"] == right - left
        assert actual["sha256"] == hashlib.sha256(
            memoryview(np.ascontiguousarray(pcm[left:right], dtype=np.float32))).hexdigest()


def validate(rows, trace, expected):
    assert [a for a, _, _ in rows] == expected
    violations = [{"ayah": a, "start_ms": ms, "reason": "outside_audio"}
                  for a, ms, _ in rows if not 0 <= ms < trace["total_ms"]]
    assert all(b[1] > a[1] for a, b in zip(rows, rows[1:]))
    reads = trace["acoustic_reads"]
    intervals = sorted(r["window_ms"] for r in reads)
    merged = []
    for left, right in intervals:
        assert 0 < right - left <= 6000 and right - left < trace["total_ms"]
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(right, merged[-1][1])
        else:
            merged.append([left, right])
    union = sum(b-a for a, b in merged)
    inputs = sum(b-a for a, b in intervals)
    assert abs(union - trace["covered_audio_ms"]) < .1
    assert abs(inputs - trace["audio_ms"]) < .1
    assert union <= trace["total_ms"] * .65 and inputs <= trace["total_ms"] * 2
    assert trace.get("bounded_ctc_fallback") is None
    confidences = {a: c for a, _, c in rows}
    starts = {a: ms for a, ms, _ in rows}
    for match in trace.get("pause_matches", []):
        if match.get("boundary_evidence") == "unresolved_local":
            assert confidences[match["ayah"]] == 0 and match["ayah"] in trace["review_required_ayahs"]
        if match.get("from_pause") and match.get("boundary_evidence") not in ("exact_fingerprint", "ctc_pre_owner"):
            midpoint = (match["pause_start_ms"] + match["pause_end_ms"]) / 2
            assert abs(starts[match["ayah"]] - midpoint) <= 1
    return violations


def main():
    INPUT_MANIFEST = '/Users/ayub/Developer/surah-cutter/docs/tasks/2026-10-16-fast-addressed-recovery/frozen-input-manifest.json'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental", action="store_true", help="Compatibility alias for the shared FAST executor")
    parser.add_argument("--models", nargs="+", default=["quran-phonetics", "whisper-tarteel-base"])
    parser.add_argument("--surahs", nargs="+", type=int, default=list(range(78, 115)))
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert all(model_store.supports_fast(m) for m in args.models)
    root = Path(__file__).resolve().parents[1]
    paths = [root / "surah_cutter/core" / f for f in (
        "align_fast.py", "fast_reads.py", "candidates.py",
        "fast_executor.py", "fast_adapters.py", "fast_ctc.py", "align.py", "emissions.py", "model_runtime.py", "model_worker.py", "model_store.py",
    )]
    source_sha = {str(p.relative_to(root)): _sha(p) for p in paths}
    paths.extend([Path(__file__).resolve(), root / "tools/bench_fast_models.py", root / "surah_cutter/data/alignment_identity.py"])
    source_sha = {str(p.relative_to(root)): _sha(p) for p in paths}
    canonical_sha = _sha(quran_text.DEFAULT_DB_PATH)
    items, recordings, loads = [], {}, {}
    with tempfile.TemporaryDirectory(prefix="local-fast-bench-") as folder:
        folder = Path(folder)
        manifest = json.loads(Path(INPUT_MANIFEST).read_text())['recordings']
        pcm_paths = {}
        for surah in args.surahs:
            info = manifest[str(surah)]
            audio = args.audio_dir / f"{surah:03d}.mp3"
            assert _sha(audio) == info['audio_sha256']
            cached_path = None
            for cache in Path(tempfile.gettempdir()).glob('local-fast-bench-*'):
                path = cache / f"{surah}.npy"
                if path.exists():
                    pcm = np.load(path, mmap_mode='r', allow_pickle=False)
                    if hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest() == info['pcm_sha256']:
                        cached_path = path
                        break
            if cached_path is None:
                pcm = decode.decode(str(audio))
                assert hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest() == info['pcm_sha256']
                cached_path = folder / f"{surah}.npy"
                np.save(cached_path, pcm)
            del pcm
            pcm_paths[surah] = cached_path
            decisions = [candidates.PauseDecision(**{**d, 'pause': candidates.Pause(**d['pause'])})
                         for d in info['decisions']]
            recordings[surah] = {**info, 'decisions': decisions,
                'decode_seconds': 0., 'dsp_seconds': 0., 'frozen_pcm_dsp_inputs': True,
                'input_manifest': INPUT_MANIFEST}
        for model_id in args.models:
            model_store.ACTIVE_ID = model_id
            align._ONNX_CACHE.clear()
            release_runtimes()
            started = time.monotonic()
            bundle = align._get_onnx_bundle()
            if hasattr(bundle.emitter, "probe"):
                bundle.emitter.probe()
            loads[model_id] = time.monotonic() - started
            for surah in args.surahs:
                info = recordings[surah]
                pcm = np.load(pcm_paths[surah], mmap_mode="r", allow_pickle=False)
                words = quran_text.surah_words(surah)
                trace = {}
                started = time.monotonic()
                executor = align_fast.align_surah_fast
                if args.experimental:
                    from surah_cutter.core.fast_executor import align_surah_sequential
                    executor = align_surah_sequential
                with dispatch_spy(bundle, pcm) as dispatches:
                    rows = executor(
                        pcm, 16000, words, bundle=bundle, surah=surah, trace=trace,
                        pause_decisions=info["decisions"],
                    )
                verify_dispatch(dispatches, pcm, trace)
                elapsed = time.monotonic() - started
                violations = validate(rows, trace, list(range(1, quran_text.ayah_count(surah) + 1)))
                item = {k: v for k, v in info.items() if k != "decisions"}
                item.update(model=model_id, surah=surah, seconds=elapsed,
                            actual_dispatches=dispatches,
                            peak_rss_self=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            peak_rss_children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                            search_seconds=elapsed + info["dsp_seconds"], rows=rows, trace=trace,
                            source=trace["model_source"], sequence_violations=violations)
                items.append(item)
                args.output.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
                print(model_id, surah, round(elapsed, 2), "review", len(trace["review_required_ayahs"]),
                      "coverage", round(trace["coverage"], 3), "calls", len(trace["acoustic_reads"]), flush=True)
        release_runtimes()
    # No reference timing is opened until BOTH models have finished inference.
    reference_sha = _sha(args.reference)
    with sqlite3.connect(f"file:{args.reference.resolve()}?mode=ro", uri=True) as conn:
        for item in items:
            truth = dict(conn.execute("SELECT ayah,time FROM timings WHERE sura=?", (item["surah"],)))
            errors = [abs(ms - truth[a]) for a, ms, _ in item["rows"]]
            item.update(reference_sha256=reference_sha, errors_ms=errors,
                        median_ms=float(np.median(errors)), p95_ms=float(np.percentile(errors, 95)), max_ms=max(errors))
    summary = {}
    for model_id in args.models:
        subset = [i for i in items if i["model"] == model_id]
        errors = np.array([e for i in subset for e in i["errors_ms"]])
        false_confident = [(i["surah"], a, e, c) for i in subset
                           for (a, _, c), e in zip(i["rows"], i["errors_ms"]) if c > 0 and e > 500]
        summary[model_id] = {"surahs": len(subset), "ayahs": len(errors),
                             "median_ms": float(np.median(errors)), "p95_ms": float(np.percentile(errors, 95)),
                             "max_ms": int(errors.max()), "over500": int((errors > 500).sum()),
                             "positive_conf_over500": false_confident,
                             "sequence_violations": [{"surah": i["surah"], **v}
                                                     for i in subset for v in i["sequence_violations"]],
                             "review_ayahs": sum(len(i["trace"]["review_required_ayahs"]) for i in subset),
                             "load_probe_seconds": loads[model_id], "decode_seconds": sum(i["decode_seconds"] for i in subset),
                             "dsp_seconds": sum(i["dsp_seconds"] for i in subset),
                             "inference_match_seconds": sum(i["seconds"] for i in subset),
                             "search_seconds": sum(i["search_seconds"] for i in subset),
                             "audio_ms": sum(i["trace"]["audio_ms"] for i in subset),
                             "covered_audio_ms": sum(i["trace"]["covered_audio_ms"] for i in subset),
                             "total_ms": sum(i["duration_ms"] for i in subset),
                             "model_reads": sum(len(i["trace"]["acoustic_reads"]) for i in subset)}
    assert source_sha == {str(p.relative_to(root)): _sha(p) for p in paths}, "Inference source changed during benchmark"
    assert canonical_sha == _sha(quran_text.DEFAULT_DB_PATH)
    args.output.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    report = {"platform": platform.platform(), "machine": platform.machine(), "source_sha256": source_sha,
              "canonical_sha256": canonical_sha, "reference_sha256": reference_sha,
              "initial_pauses": {s: {**{k: v for k, v in i.items() if k != "decisions"},
                                      "decisions": [asdict(d) for d in i["decisions"]]} for s, i in recordings.items()},
              "summary": summary}
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
