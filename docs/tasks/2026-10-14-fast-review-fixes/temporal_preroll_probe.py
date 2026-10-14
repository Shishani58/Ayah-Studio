"""Isolated local attention diagnostics; neither convention enables FAST timing."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import monotonic
from unittest.mock import patch

import numpy as np


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    worker_path = Path(spec["worker_path"])
    module_spec = importlib.util.spec_from_file_location("diagnostic_worker", worker_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    from transformers.models.whisper import generation_whisper

    acoustic = module.Acoustic(spec["model"], Path(spec["folder"]))
    results = []
    original_dtw = generation_whisper._dynamic_time_warping
    for case in spec["cases"]:
        pcm = np.load(case["path"], allow_pickle=False)
        assert 4800 <= len(pcm) <= 96000 and len(pcm) < case["total_samples"]
        calls = []
        original_asr = acoustic.whisper_local
        original_clip = acoustic.whisper_clip

        def charged(operation, phase):
            started = monotonic()
            calls.append({"method": phase, "samples": len(pcm),
                          "sha256": hashlib.sha256(memoryview(pcm)).hexdigest()})
            try:
                return operation()
            finally:
                calls[-1]["seconds"] = monotonic() - started

        asr = charged(lambda: original_asr(pcm, case["total_samples"]), "independent_asr")
        timings = {}
        for target in ("إِنَّا", "الْحَمْدُ"):
            captured = []

            def capture(matrix):
                text, times = original_dtw(matrix)
                jumps = np.pad(np.diff(text), (1, 0), constant_values=1).astype(bool)
                captured.append((times[jumps] * 20).tolist())
                return text, times

            with patch.object(generation_whisper, "_dynamic_time_warping", capture):
                rows = charged(lambda: original_clip(pcm, [target], strict_words=False),
                               "teacher_forced_attention_" + target)
            ids = acoustic.processor.tokenizer.encode(" " + target, add_special_tokens=False)
            first = 0
            while first < len(ids) and not acoustic.processor.tokenizer.decode([ids[first]]).strip():
                first += 1
            timings[target] = {"existing_query_plus_one": rows,
                               "diagnostic_predicting_query": int(captured[0][first]),
                               "query_times": captured[0], "first_text_index": first,
                               "token_ids": ids}
        results.append({"case": case["name"], "asr": asr, "timings": timings,
                        "actual_dispatches": calls})
    Path(spec["output"]).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def capture_ctc(bundle, pcm):
    from tools.bench_fast_models import dispatch_spy

    with dispatch_spy(bundle, pcm) as calls:
        yield calls


def main():
    from surah_cutter.core import align, decode, model_store
    from surah_cutter.core.fast_adapters import CTCAdapter, attributed_asr_units
    from surah_cutter.core.fast_reads import LocalAcousticReads, LocalEvidenceReads
    from surah_cutter.core.model_runtime import environment, release_runtimes
    from tools.bench_fast_models import verify_dispatch

    original = decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/108.mp3")
    cases = []
    for start in (5720, 6320):
        for silence in (0, 200, 1200):
            pcm = np.r_[np.zeros(silence * 16, np.float32), original[start * 16:]]
            end = 2700 + silence
            cases.append({"name": f"crop{start}_silence{silence}", "pcm": pcm,
                          "crop_start_ms": start, "silence_ms": silence, "end_ms": end})
    cases.append({"name": "basmala_negative", "pcm": original,
                  "crop_start_ms": 0, "silence_ms": 0, "end_ms": 2700})
    release_runtimes()
    output = []
    with tempfile.TemporaryDirectory(prefix="fast-temporal-preroll-") as folder:
        folder = Path(folder)
        for i, case in enumerate(cases):
            np.save(folder / f"{i}.npy", case["pcm"][:case["end_ms"] * 16])
        for model in ("whisper-tarteel-tiny", "whisper-tarteel-base"):
            item = model_store.spec(model)
            spec = {"model": model, "folder": str(model_store.installed_path(item)),
                    "worker_path": str(Path(align.__file__).with_name("model_worker.py")),
                    "output": str(folder / "result.json"), "cases": [
                        {"name": case["name"], "path": str(folder / f"{i}.npy"),
                         "total_samples": len(case["pcm"])} for i, case in enumerate(cases)]}
            (folder / "spec.json").write_text(json.dumps(spec))
            subprocess.run([str(environment(model) / "bin/python"), str(Path(__file__).resolve()),
                            "--worker", str(folder / "spec.json")], check=True)
            results = json.loads((folder / "result.json").read_text())
            for case, result in zip(cases, results):
                reads = LocalAcousticReads(case["pcm"], 16000, None)
                s0, s1 = reads._window(0, case["end_ms"])
                for call in result["actual_dispatches"]:
                    permission = reads._permit(s0, s1)
                    assert permission is not None
                    attempt = reads._begin_read(s0, s1, permission, phase=call["method"])
                    attempt.update(outcome="completed", seconds=call["seconds"])
                trace = reads.diagnostics()
                verify_dispatch(result["actual_dispatches"], case["pcm"], trace)
                units, quality, attribution = attributed_asr_units(result["asr"])
                local = case["pcm"][:s1]
                rms = np.sqrt(np.mean(local[:len(local) // 320 * 320].reshape(-1, 320) ** 2, axis=1))
                thresholds = (.003, .01, float(rms.max() * .08))
                result.update(model=model, crop_start_ms=case["crop_start_ms"],
                              silence_ms=case["silence_ms"], ledger=trace,
                              units=units, unit_quality=quality, attribution=attribution,
                              independently_proved_inna=bool(units and units[0] == "انا" and quality[0] >= .6),
                              rms_onsets_ms={str(t): int(np.flatnonzero(rms >= t)[0] * 20)
                                             for t in thresholds if np.any(rms >= t)})
                output.append(result)
                print(model, case["name"], result["asr"]["text"], result["timings"], flush=True)
    model_store.ACTIVE_ID = "quran-phonetics"
    align._ONNX_CACHE.clear()
    bundle = align.default_bundle()
    adapter = CTCAdapter(bundle)
    ctc = []
    for case in cases:
        reads = LocalEvidenceReads(case["pcm"], 16000, adapter)
        with capture_ctc(bundle, case["pcm"]) as calls:
            obs = reads.read(0, case["end_ms"], reason="temporal_preroll_independent_ctc")
        trace = reads.diagnostics()
        verify_dispatch(calls, case["pcm"], trace)
        ctc.append({"case": case["name"], "units": obs.units, "positions_ms": obs.positions_ms,
                    "unit_quality": obs.unit_quality, "ledger": trace, "actual_dispatches": calls})
    report = {"results": output, "ctc": ctc,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "worker_sha256": hashlib.sha256(Path(align.__file__).with_name("model_worker.py").read_bytes()).hexdigest(),
              "limitation": "DSP activity is not labelled word onset; predicting-query alternative and existing convention are diagnostic only. No timing capability enabled."}
    Path(__file__).with_name("temporal-preroll-probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
