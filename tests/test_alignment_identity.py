import copy
import hashlib
import json

import pytest

from surah_cutter.data.alignment_identity import fast_resume_suras, fast_run_key, trace_run_key
from surah_cutter.data.alignment_result import result_source, write_alignment_result
from surah_cutter.data.project_db import ProjectDB


@pytest.fixture
def identity():
    return {
        "algorithm": "sequential-fast-v2-experimental-1", "adapter": "ctc-test-v1",
        "model": "captured-model", "revision": "captured-revision",
        "policy": {"max_candidates": 8, "quality": .25, "cache_bytes": 1024},
        "lag_ms": 20, "text_sha256": "a" * 64, "pcm_sha256": "b" * 64,
        "sample_rate": 16000, "dsp_version": "pause-pipeline-v1",
        "decode": "ffmpeg-mono-float32-16000", "normalization": "ctc-test-v1",
        "limits": {"max_read_ms": 6000, "coverage": .65, "input_ratio": 2.},
    }


def test_key_is_stable_matches_executor_encoding_and_ignores_active_selection(identity, monkeypatch):
    from surah_cutter.core import model_store

    expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    monkeypatch.setattr(model_store, "ACTIVE_ID", "other-model")
    assert fast_run_key(identity) == expected
    assert fast_run_key(dict(reversed(list(identity.items())))) == expected
    assert trace_run_key({"run_snapshot": identity, "fast_run_key": expected}) == expected


def test_writer_source_uses_captured_lag_not_global_selection(identity, monkeypatch):
    from surah_cutter.core import model_store

    def global_lag():
        pytest.fail("Captured run must not consult the current global model lag")

    monkeypatch.setattr(model_store, "lag_ms", global_lag)
    source = result_source({"model_source": "captured-model:revision=captured-revision",
                            "run_snapshot": identity, "fast_run_key": fast_run_key(identity)}, "fast")
    assert "align_lag_ms=20" in source


@pytest.mark.parametrize("field", ["algorithm", "adapter", "model", "revision", "policy",
                                     "lag_ms", "text_sha256", "pcm_sha256", "sample_rate",
                                     "dsp_version", "decode", "normalization", "limits"])
def test_every_execution_input_changes_key(identity, field):
    original = fast_run_key(identity)
    changed = copy.deepcopy(identity)
    value = changed[field]
    if isinstance(value, dict):
        value["effective_env_change"] = 1
    elif field.endswith("sha256"):
        changed[field] = "c" * 64
    elif isinstance(value, str):
        changed[field] += "-different"
    else:
        changed[field] += 1
    assert fast_run_key(changed) != original


@pytest.mark.parametrize("field,value", [("revision", "unknown"), ("text_sha256", "missing"),
                                        ("pcm_sha256", None), ("sample_rate", True),
                                        ("lag_ms", float("nan")), ("policy", {}),
                                        ("limits", {"coverage": float("inf")})])
def test_incomplete_or_nonfinite_snapshot_cannot_claim_a_key(identity, field, value):
    identity[field] = value
    with pytest.raises(ValueError):
        fast_run_key(identity)


def test_legacy_and_mismatched_trace_cannot_claim_current_identity(identity):
    assert trace_run_key({"model_source": "legacy", "requested_mode": "fast"}) is None
    assert trace_run_key({"run_snapshot": identity}) is None
    with pytest.raises(ValueError, match="disagrees"):
        trace_run_key({"run_snapshot": identity, "fast_run_key": "0" * 64})
    with pytest.raises(ValueError):
        fast_run_key({"algorithm": "partial"})


@pytest.mark.parametrize("source", ["legacy", "test:requested_mode=fast", "test:requested_mode=full",
                                   "test:requested_mode=fast:fast_run_key=" + "0" * 64])
def test_complete_legacy_auto_rows_do_not_skip_new_execution(tmp_path, identity, source):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(114, a, a * 1000) for a in range(1, 7)], source)
    before = path.read_bytes()
    assert fast_resume_suras(path, {114: fast_run_key(identity)}) == set()
    assert path.read_bytes() == before


def test_exact_key_skips_only_complete_matching_auto_or_fully_protected(tmp_path, identity):
    path = tmp_path / "p.db"
    key = fast_run_key(identity)
    source = f"test:requested_mode=fast:fast_run_key={key}"
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(114, a, a * 1000) for a in range(1, 7)], source, confidence=.9)
        assert fast_resume_suras(path, {114: key}) == {114}
        assert fast_resume_suras(path) == set()
        db.write_timings([(114, 6, 6000)], "legacy:requested_mode=fast")
        assert fast_resume_suras(path, {114: key}) == set()
        db.set_status(114, 6, "verified")
        assert fast_resume_suras(path, {114: key}) == {114}
        for ayah in range(1, 6):
            db.set_status(114, ayah, "verified")
        assert fast_resume_suras(path) == {114}
        assert fast_resume_suras(path, {114: "0" * 64}) == {114}
        db._conn.execute("DELETE FROM timings WHERE sura=114 AND ayah=6")
        db._conn.commit()
        assert fast_resume_suras(path, {114: key}) == set()


def test_writer_persists_validated_key_in_same_transaction_as_auto_rows(tmp_path, identity):
    path = tmp_path / "p.db"
    key = fast_run_key(identity)
    trace = {"model_source": "reference:test", "run_snapshot": identity, "fast_run_key": key,
             "ayahs": [{"ayah": a, "conf": .9} for a in range(1, 7)]}
    with ProjectDB.open(str(path), "test", 1) as db:
        write_alignment_result(db, [(114, a, a * 1000) for a in range(1, 7)],
                               trace=trace, requested_mode="fast", db_path=path)
        assert fast_resume_suras(path, {114: key}) == {114}
        before = db.get_timings()
        trace["fast_run_key"] = "0" * 64
        with pytest.raises(ValueError, match="disagrees"):
            write_alignment_result(db, [(114, 1, 1200)], trace=trace,
                                   requested_mode="fast", db_path=path)
        assert db.get_timings() == before


def test_web_fast_resume_recomputes_legacy_auto_but_skips_fully_protected(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException
    from surah_cutter.webgui import server

    path = tmp_path / "p.db"
    monkeypatch.setattr(server, "STATE", SimpleNamespace(db_path=path, audio_map={114: "114.mp3"}))
    request = {"mode": "fast", "only_missing": True}
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(114, a, a * 1000) for a in range(1, 7)], "legacy:requested_mode=fast")
        assert server._generation_modes() == {114: {"fast"}}
        assert server._generation_suras(request) == [114]
        for ayah in range(1, 7):
            db.set_status(114, ayah, "verified")
        assert server._generation_modes() == {114: set()}
        with pytest.raises(HTTPException, match="all done"):
            server._generation_suras(request)


@pytest.mark.parametrize("name", ["project#one.db", "project?two.db"])
def test_resume_readonly_uri_supports_desktop_filename_characters(tmp_path, identity, name):
    path = tmp_path / name
    key = fast_run_key(identity)
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(114, a, a * 1000) for a in range(1, 7)],
                         f"test:requested_mode=fast:fast_run_key={key}", confidence=.9)
    before = path.read_bytes()
    assert fast_resume_suras(path, {114: key}) == {114}
    assert path.read_bytes() == before


def test_immutable_history_retains_each_execution_identity_and_evidence(tmp_path, identity):
    from surah_cutter.data.alignment_runs import list_runs, load_run

    path = tmp_path / "p.db"
    identities = []
    with ProjectDB.open(str(path), "test", 1) as db:
        for candidate_limit in (4, 8):
            captured = copy.deepcopy(identity)
            captured["policy"]["max_candidates"] = candidate_limit
            key = fast_run_key(captured)
            identities.append(key)
            trace = {"model_source": "captured-model:revision=captured-revision",
                     "run_snapshot": captured, "fast_run_key": key,
                     "transitions": [{"state": "READ_LOCAL", "reason": str(candidate_limit)}],
                     "acoustic_reads": [{"window_ms": [0, candidate_limit * 100], "outcome": "completed"}],
                     "boundaries": [{"ayah": 1, "evidence_kind": "pause", "read_id": 0}],
                     "generated_rows": [(114, a, a * 1000) for a in range(1, 7)]}
            write_alignment_result(db, trace["generated_rows"], trace=trace,
                                   requested_mode="fast", db_path=path)
    folder = tmp_path / "p.logs"
    runs = list_runs(folder, 114)
    assert len(runs) == 2
    preserved = []
    for run in runs:
        loaded = load_run(folder, 114, run["id"])
        assert len(loaded["ayahs"]) == 6  # Existing reader accepts the v2 JSON.
        snapshot = json.loads((folder / "runs" / f"{run['id']}.json").read_text())
        assert snapshot["version"] == 2
        trace = snapshot["trace"]
        assert trace_run_key(trace) == trace["fast_run_key"]
        limit = trace["run_snapshot"]["policy"]["max_candidates"]
        assert trace["transitions"][0]["reason"] == str(limit)
        assert trace["acoustic_reads"][0]["window_ms"] == [0, limit * 100]
        assert trace["boundaries"][0]["read_id"] == 0
        preserved.append(trace["fast_run_key"])
    assert sorted(preserved) == sorted(identities)
    latest = json.loads((folder / "sura_114_fast.json").read_text())
    assert latest["trace"]["fast_run_key"] == identities[-1]
