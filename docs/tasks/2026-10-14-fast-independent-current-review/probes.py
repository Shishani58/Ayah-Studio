"""Independent read-only/fake repros; writes only this task's JSON."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from surah_cutter.core.fast_adapters import CTCAdapter, Observation, TarteelAdapter
from surah_cutter.core.fast_executor import Boundary, plan_phrase, validate_boundaries
from surah_cutter.data import quran_text
from surah_cutter.data.alignment_identity import fast_resume_suras
from surah_cutter.data.alignment_result import write_alignment_result
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.webgui import server

ROOT = Path(__file__).parent
results = {}


def resume_probe():
    with TemporaryDirectory(prefix="fast-independent-review-") as folder:
        path = Path(folder) / "temporary.db"
        with ProjectDB.open(path, "temporary", 1) as db:
            db.write_timings([(32, a, a * 1000) for a in range(1, 31)],
                             "legacy:requested_mode=auto:actual_mode=fast")
        with patch.object(server.STATE, "db_path", path), patch.object(
                server.STATE, "audio_map", {32: Path(folder) / "032.mp3"}):
            selected = {}
            for mode in ("auto", "fast"):
                try:
                    selected[mode] = server._generation_suras(
                        {"suras": [32], "only_missing": True, "mode": mode})
                except server.HTTPException as exc:
                    selected[mode] = {"status": exc.status_code, "detail": exc.detail}
        results["resume"] = {"protected_or_exact_key": sorted(fast_resume_suras(path)),
                             "web_selected": selected}
        assert selected["fast"] == [32]
        assert isinstance(selected["auto"], dict) and selected["auto"]["status"] == 400


def diagnostics_probe():
    observation = Observation(("unique",), None, .99, unit_quality=(.99,))
    trace = {"fast_version": "sequential-fast-v3-sparse-5", "coverage": .1,
             "n_pause": 1, "n_wasl": 0,
             "ayahs": [{"ayah": 1, "conf": .99, "reliable": True},
                       {"ayah": 2, "conf": 0., "reliable": False}],
             "review_required_ayahs": [2],
             "acoustic_reads": [{"phase": "independent-local", "window_ms": [1000, 2500],
                                 "outcome": "completed", "observed": asdict(observation)}],
             "transitions": [{"state": "REVIEW_REQUIRED", "ayah": 2, "reason": "no_local_evidence"}],
             "pause_matches": []}
    summary = server._fast_trace_summary(trace)
    log = server._format_fast_log(112, [], trace, [(1, 1100, .99), (2, 4000, 0.)])
    results["diagnostics"] = {"summary": summary, "log": log,
                              "actual_review_count": len(trace["review_required_ayahs"])}
    assert summary["unresolved"] == 0
    assert "window=1000-2500" not in log


def seam_probe():
    # Only synthetic representations: proves an ownership hole, not an audio error.
    groups = {1: ("alpha", "beta"), 2: ("other", "alpha"), 3: ("beta", "finish")}
    ends = {a: [1, 2] for a in groups}
    plan = plan_phrase(groups, 1, ends)
    heard = Observation(("alpha", "beta", "finish"), None, .99,
                        unit_quality=(.99, .99, .99))
    read = {"window_ms": [1000., 2500.], "outcome": "completed", "observed": asdict(heard)}
    proof = {"phrase": plan["units"], "complete": True, "ambiguous": False,
             "competitors_checked": True, "offset": 0, "pause_ms": [1000., 1200.], "pause_id": 0}
    boundary = Boundary(1, 1100, "pause", .99, read_id=0, proof=proof)
    validate_boundaries([boundary], [1], 10000, [read], groups=groups, word_ends=ends,
                        threshold=.6, pauses=[SimpleNamespace(start_ms=1000., end_ms=1200.)])
    results["cross_ayah_seam_synthetic"] = {"plan": plan, "validator_accepted": True,
                                         "actual_owner": "last word of 2 + first word of 3"}
    assert plan["units"] == ("alpha", "beta") and not plan["ambiguous"]


def canonical_seam_scan():
    findings = []
    # Text-only, all 114 surahs. No model, acoustic reads or canonical writes.
    for kind in ("asr", "ctc"):
        adapter = (TarteelAdapter(SimpleNamespace(emitter=SimpleNamespace(
            model_id="whisper-tarteel-tiny", runtime_identity={"test": True})))
                   if kind == "asr" else CTCAdapter.__new__(CTCAdapter))
        for surah in range(1, 115):
            words = quran_text.surah_words(surah)
            rendered = adapter.represent(words)
            proof_ends = (adapter.proof_word_ends(words, rendered) if kind == "asr"
                          else [(len(u),) for u in rendered])
            groups, ends = defaultdict(list), defaultdict(list)
            for word, units, word_ends in zip(words, rendered, proof_ends):
                base = len(groups[word.ayah])
                groups[word.ayah].extend(units)
                ends[word.ayah].extend(base + end for end in word_ends)
            groups = {a: tuple(u) for a, u in groups.items()}
            for ayah in groups:
                plan = plan_phrase(groups, ayah, ends)
                phrase = plan["units"]
                if plan["ambiguous"]:
                    continue
                for other, units in groups.items():
                    stream = units + groups.get(other + 1, ())
                    for offset in [0] + ends[other][:-1]:
                        if offset < len(units) < offset + len(phrase) and stream[offset:offset + len(phrase)] == phrase:
                            findings.append({"adapter": kind, "surah": surah, "claimed_ayah": ayah,
                                             "actual_seam": [other, other + 1], "offset": offset,
                                             "phrase": phrase})
    results["canonical_cross_ayah_collisions"] = findings


def model_label_probe():
    emitter = SimpleNamespace(model_id="quran-phonetics-fp32",
                              runtime_identity={"weights_sha256": "a" * 64})
    adapter = CTCAdapter(SimpleNamespace(emitter=emitter, vocab={"a": 1}, blank_id=0, lag_ms=200))
    results["ctc_model_label"] = {"selected_emitter": emitter.model_id, "adapter_model": adapter.model_id,
                                  "weights_identity_kept": adapter.revision}
    assert adapter.model_id != emitter.model_id


def auto_history_probe():
    with TemporaryDirectory(prefix="fast-auto-history-") as folder:
        path = Path(folder) / "temporary.db"
        rows = [(112, a, 1000 * a) for a in range(1, 5)]
        trace = {"auto_mode": "fast", "fast_version": "sequential-fast-v3-sparse-5",
                 "model_source": "quran-phonetics:revision=temporary",
                 "align_lag_ms": 200, "total_ms": 20000,
                 "generated_rows": rows,
                 "ayahs": [{"ayah": a, "conf": 0.} for a in range(1, 5)]}
        with ProjectDB.open(path, "temporary", 1) as db:
            first = write_alignment_result(db, rows, trace=trace, requested_mode="auto", db_path=path)
            trace["marker"] = "second-run"
            second = write_alignment_result(db, rows, trace=trace, requested_mode="auto", db_path=path)
        logs = path.with_name("temporary.logs")
        results["auto_fast_history"] = {"first_paths": first.log_paths, "second_paths": second.log_paths,
                                        "history_runs": [str(p) for p in logs.glob("runs/*.json")],
                                        "latest_replaced": json.loads(Path(second.log_paths["json"]).read_text())["trace"]["marker"]}
        assert "run" not in first.log_paths and not list(logs.glob("runs/*.json"))


if __name__ == "__main__":
    resume_probe()
    diagnostics_probe()
    seam_probe()
    canonical_seam_scan()
    model_label_probe()
    auto_history_probe()
    results["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path("surah_cutter/core/fast_executor.py"), Path("surah_cutter/core/fast_adapters.py"),
                  Path("surah_cutter/webgui/server.py"), Path("surah_cutter/gui/dashboard_view.py")]}
    (ROOT / "probes.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "source_sha256"}, ensure_ascii=False))
