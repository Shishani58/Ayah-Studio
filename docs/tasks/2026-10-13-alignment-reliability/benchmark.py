"""Read-only before/after and candidate-model audit on Shatri 78..114.

Run from the repository root:
  .venv/bin/python docs/tasks/2026-10-13-alignment-reliability/benchmark.py

The candidate ONNX and sidecars live in /tmp/surah-model-comparison; no
user/project DB is opened for writing. Baseline core files must be clean in
the selected git revision (HEAD was the pre-change version for this task).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from surah_cutter.core import _ctc, align, align_fast, candidates, decode, emissions  # noqa: E402
from surah_cutter.core._ctc import forced_align, merge_repeats  # noqa: E402
from surah_cutter.data import quran_text, references, transliteration  # noqa: E402
from tools.bench_fast import pause_contract_violations, sequence_contract_violations  # noqa: E402


def legacy_module(name, relative_path):
    source = subprocess.check_output(["git", "show", f"HEAD:{relative_path}"], cwd=ROOT)
    path = Path("/tmp") / f"{name}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Preserve repository-relative model paths (Silero VAD), even though the
    # historical source itself is evaluated from a temporary file.
    module.__file__ = str(ROOT / relative_path)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ctc_rows(pcm, words, bundle, trs, intro, emis):
    targets, spans = align._build_targets(words, trs, bundle, intro)
    path, scores = forced_align(emis[None], np.array(targets)[None], blank=bundle.blank_id)
    segments = [s for s in merge_repeats(path[0].tolist()) if s.label != bundle.blank_id]
    assert [s.label for s in segments] == targets
    tspans = [(s.start, s.end) for s in segments]
    stride = align._stride_ms(len(pcm), 16_000, len(emis))
    rows = align._ayah_rows(words, spans, tspans, scores[0], stride, np)
    checks = align._check_ayah_onsets(
        words, spans, tspans, scores[0], stride,
        pauses=candidates.detect_energy_pauses(pcm, 16_000, use_ten_vad=False),
    )
    return rows, checks


def cached_bundle(bundle, emis):
    return align._OnnxBundle(SimpleNamespace(
        vocab=bundle.vocab, blank_id=bundle.blank_id, emissions=lambda _pcm: emis,
    ))


def stats(errors):
    values = np.abs(errors)
    return {
        "n": len(errors), "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)), "max_ms": int(max(values)),
        "over500": int(np.sum(values > 500)),
    }


def main():
    task_dir = Path(__file__).parent
    audio = Path.home() / "Desktop/Abu_Bakr_Ash-Shatri"
    reference = references.load_reference("abu_bakr_ash_shatri")
    old_fast = legacy_module("reliability_old_fast", "surah_cutter/core/align_fast.py")
    old_candidates = legacy_module("reliability_old_candidates", "surah_cutter/core/candidates.py")
    old_ctc = legacy_module("reliability_old_ctc", "surah_cutter/core/_ctc.py")
    current_ctc = _ctc.forced_align
    original_decisions = candidates.pause_pipeline_decisions
    current = align._get_onnx_bundle(str(ROOT / "models/wav2vec2-quran-phonetics.onnx"))
    candidate_dir = Path("/tmp/surah-model-comparison")
    vocab = json.loads((candidate_dir / "vocab.json").read_text())
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(candidate_dir / "model_q4.onnx"), sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    candidate = align._OnnxBundle(emissions.OnnxEmitter(session, vocab, blank_id=0))

    def arabic(text):
        # Only tokenizer normalization; canonical Quran words are never changed.
        return unicodedata.normalize("NFKC", text).replace("ۡ", "ْ")

    intro_ar = [arabic(w.text) for w in quran_text.surah_words(1) if w.ayah == 1]
    results = {"baseline_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate_revision": "f90650aac5d9dcff15230a19998c0027f45b33f1",
        "settings": {"sr": 16000, "window_s": 30, "context_s": 2, "lag_ms": 200},
        "suras": [], "summary": {}}
    errors = {name: [] for name in ["full_before", "full_after", "fast_before", "fast_after", "arabic_q4"]}
    selected = [int(a) for a in sys.argv[1:] if a != "--fresh"] or list(range(78, 115))
    output = task_dir / "benchmark.json"
    if output.exists() and "--fresh" not in sys.argv:
        results = json.loads(output.read_text())
        for item in results["suras"]:
            for name, values in item["errors"].items():
                errors[name].extend(values)
    done = {item["surah"] for item in results["suras"]}
    for surah in selected:
        if surah in done:
            continue
        start = time.monotonic()
        pcm = decode.decode(str(audio / f"{surah:03d}.mp3"))
        words = quran_text.surah_words(surah)
        trs = transliteration.surah_word_tr(surah)
        intro = align._basmala_trs(transliteration)
        item = {"surah": surah, "rows": {}, "contracts": {}, "repairs": []}
        for batch in [4, 1]:
            emis = current.emitter.emissions(pcm, batch_size=batch)
            full_rows, checks = ctc_rows(pcm, words, current, trs, intro, emis)
            if batch == 4:
                item["rows"]["full_before"] = full_rows
                candidates.pause_pipeline_decisions = old_candidates.pause_pipeline_decisions
                _ctc.forced_align = old_ctc.forced_align
                try:
                    tr = {}
                    item["rows"]["fast_before"] = old_fast.align_surah_fast(
                        pcm, 16000, words, surah=surah,
                        bundle=cached_bundle(current, emis), trace=tr,
                    )
                finally:
                    candidates.pause_pipeline_decisions = original_decisions
                    _ctc.forced_align = current_ctc
            else:
                item["rows"]["full_after"] = [
                    (a, checks[a]["onset_ms"], checks[a]["conf"]) for a, _, _ in full_rows
                ]
                item["repairs"] = [check for check in checks.values() if check["repaired"]]
                tr = {}
                item["rows"]["fast_after"] = align_fast.align_surah_fast(
                    pcm, 16000, words, surah=surah,
                    bundle=cached_bundle(current, emis), trace=tr,
                )
                item["contracts"]["pause"] = pause_contract_violations(item["rows"]["fast_after"], tr)
                item["contracts"]["sequence"] = sequence_contract_violations(
                    item["rows"]["fast_after"], list(range(1, quran_text.ayah_count(surah) + 1))
                )
            np.save(f"/tmp/reliability-{surah}-{batch}.npy", emis)
        candidate_emis = candidate.emitter.emissions(pcm)
        candidate_rows, candidate_checks = ctc_rows(
            pcm, words, candidate, [arabic(w.text) for w in words], intro_ar, candidate_emis,
        )
        # Compare the unmodified alternative CTC starts, not our pause repair.
        item["rows"]["arabic_q4"] = candidate_rows
        item["candidate_first_tokens"] = [
            {"ayah": a, "first_token_conf": c["first_token_conf"], "conf": c["conf"]}
            for a, c in candidate_checks.items()
        ]
        item["errors"] = {}
        for name, rows in item["rows"].items():
            signed = [ms - reference[(surah, a)] for a, ms, _c in rows]
            errors[name].extend(signed)
            item["errors"][name] = signed
        item["seconds"] = round(time.monotonic() - start, 2)
        results["suras"].append(item)
        results["summary"] = {name: stats(es) for name, es in errors.items()}
        (task_dir / "benchmark.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(surah, "seconds", item["seconds"],
              {name: stats(es)["max_ms"] for name, es in item["errors"].items()},
              "repairs", len(item["repairs"]), "contracts", item["contracts"], flush=True)
    print("SUMMARY", json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
