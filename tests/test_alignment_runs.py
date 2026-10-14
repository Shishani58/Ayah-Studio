import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from surah_cutter.data.alignment_result import write_alignment_result
from surah_cutter.data.alignment_runs import list_runs, load_run
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.webgui import server
from surah_cutter.webgui.state import AppState


def result(db, path, mode, time):
    return write_alignment_result(
        db, [(114, a, time + a * 1000) for a in range(1, 7)], requested_mode=mode,
        trace={"model_source": "onnx-ctc:test", "mode": mode,
               "generated_rows": [(114, a, time + a * 1000) for a in range(1, 7)],
               "ayahs": [{"ayah": a, "conf": 0 if a == 1 else .8} for a in range(1, 7)]},
        db_path=path,
    )


def test_every_run_survives_reopen_and_latest_replacement(tmp_path):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        for mode, time in [("fast", 0), ("full", 90), ("fast", 20)]:
            assert result(db, path, mode, time)[0] == 6
    runs = list_runs(tmp_path / "p.logs", 114)
    assert len(runs) == 3
    assert len({r["id"] for r in runs}) == 3
    loaded = [load_run(tmp_path / "p.logs", 114, r["id"]) for r in runs]
    assert sorted(r["ayahs"][0]["time"] for r in loaded) == [1000, 1020, 1090]
    assert all(r["ayahs"][0]["confidence"] == 0 for r in loaded)
    assert all(r["raw_model_result"] for r in loaded)
    assert json.loads((tmp_path / "p.logs/sura_114_fast.json").read_text())["rows"][0][2] == 1020


def test_raw_snapshot_keeps_model_proposal_and_manual_database_is_protected(tmp_path, monkeypatch):
    from surah_cutter.core import pipeline
    from surah_cutter.data import quran_text

    path = tmp_path / "p.db"
    monkeypatch.setattr(quran_text, "surah_words", lambda surah: [])

    def align(*args, **kw):
        kw["trace"].update(ayahs=[{"ayah": 1, "conf": .9}], model_source="onnx-ctc:test")
        return [(114, 1, 1200)]

    monkeypatch.setattr(pipeline, "_full_align", align)
    with ProjectDB.open(str(path), "test", 1) as db:
        db.set_time(114, 1, 1000)
        protected = db.get_review(114, 1)
        trace = {}
        rows = pipeline.process_surah("dummy.mp3", 114, mode="full", project_db=db, trace=trace)
        assert rows == [(114, 1, 1000)]
        written, logs = write_alignment_result(db, rows, trace=trace, requested_mode="full", db_path=path)
        assert written == 0
        assert db.get_review(114, 1) == protected
        assert db.get_timings() == [(114, 1, 1000)]
    saved = load_run(tmp_path / "p.logs", 114, Path(logs["run"]).stem)
    assert saved["ayahs"] == [{"ayah": 1, "time": 1200, "confidence": .9}]


def test_failed_merge_publishes_no_snapshot_or_latest(tmp_path):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        result(db, path, "fast", 0)
        db.set_time(114, 2, 2000)
        before = list_runs(tmp_path / "p.logs", 114)
        with pytest.raises(ValueError, match="порядку"):
            result(db, path, "full", 2500)
        assert list_runs(tmp_path / "p.logs", 114) == before
        assert not (tmp_path / "p.logs/sura_114_full.json").exists()
        assert not list((tmp_path / "p.logs/runs").glob(".*.tmp"))


def test_legacy_log_is_available_and_archived_before_overwrite(tmp_path):
    path = tmp_path / "p.db"
    folder = tmp_path / "p.logs"
    folder.mkdir()
    legacy = {"surah": 114, "mode": "fast", "rows": [[114, 1, 999]],
              "trace": {"model_source": "old-model", "ayahs": [{"ayah": 1, "conf": .3}]}}
    (folder / "sura_114_fast.json").write_text(json.dumps(legacy))
    assert list_runs(folder, 114)[0]["id"] == "legacy-fast"
    assert load_run(folder, 114, "legacy-fast")["ayahs"][0]["confidence"] == .3
    with ProjectDB.open(str(path), "test", 1) as db:
        result(db, path, "fast", 20)
    runs = list_runs(folder, 114)
    assert len(runs) == 2
    old = next(load_run(folder, 114, r["id"]) for r in runs if not r["raw_model_result"])
    assert old["ayahs"][0] == {"ayah": 1, "time": 999, "confidence": .3}


def test_symlink_history_preflight_preserves_database(tmp_path):
    path = tmp_path / "p.db"
    (tmp_path / "p.logs").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "p.logs/runs").symlink_to(tmp_path / "elsewhere")
    with ProjectDB.open(str(path), "test", 1) as db:
        with pytest.raises(ValueError, match="ссылкой"):
            result(db, path, "fast", 20)
        assert not db.get_timings()


def test_api_history_is_read_only_and_rejects_traversal_other_sura_and_corruption(tmp_path, monkeypatch):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        result(db, path, "fast", 0)
        result(db, path, "full", 90)
    state = AppState()
    state.db_path = path
    monkeypatch.setattr(server, "STATE", state)
    def endpoint(route):
        return next(r.endpoint for r in server.app.routes if r.path == route)
    before = path.read_bytes()
    runs = endpoint("/api/sura/{surah}/runs")(114)["runs"]
    assert len(runs) == 2
    read = endpoint("/api/sura/{surah}/runs/{run_id}")
    assert len(read(114, runs[0]["id"])["ayahs"]) == 6
    for run_id in ("../p.db", runs[0]["id"].replace("114", "113"), "legacy-words"):
        with pytest.raises(HTTPException) as exc:
            read(114, run_id)
        assert exc.value.status_code == 404
    (tmp_path / "p.logs/runs" / f"{runs[0]['id']}.json").write_text("broken")
    assert len(endpoint("/api/sura/{surah}/runs")(114)["runs"]) == 1
    assert path.read_bytes() == before
