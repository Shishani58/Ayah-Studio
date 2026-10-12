# Call route endpoints directly: these regressions exercise project state and
# SQLite writes without requiring an HTTP client or real alignment models.
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from surah_cutter.data.project_db import ProjectDB
from surah_cutter.webgui import server
from surah_cutter.webgui.server import _fast_trace_summary
from surah_cutter.webgui.state import AppState, DB_CODE, DB_VERSION, classify_status


def test_fast_trace_summary_exposes_forecast_and_atomic_fallbacks():
    summary = _fast_trace_summary(
        {
            "ayahs": [
                {"ayah": 1, "reliable": True},
                {"ayah": 2, "repaired": True},
            ],
            "tempo_forecast": {
                "rate_units_per_ms": 0.12,
                "confidence": 0.8,
                "observation_count": 3,
            },
            "pause_sequence_events": [
                {"action": "unresolved_current", "ayah": 2},
                {"action": "unresolved_current", "ayah": 2},
            ],
            "model_reads": [
                {"phase": "wasl-run", "atomic": True},
                {"phase": "wasl-tiled-fallback", "atomic": True},
                {"phase": "wasl-run", "atomic": False},
            ],
        }
    )

    assert summary["tempo_rate"] == 0.12
    assert summary["tempo_confidence"] == 0.8
    assert summary["tempo_observations"] == 3
    assert summary["unresolved"] == 1
    assert summary["atomic_wasl_reads"] == 2


@pytest.fixture
def project(tmp_path, monkeypatch):
    state = AppState()
    state.set_db_path(str(tmp_path / "a.db"))
    state.audio_map = {113: tmp_path / "a/113.mp3", 114: tmp_path / "a/114.mp3"}
    with ProjectDB.open(str(state.db_path), DB_CODE, DB_VERSION):
        pass
    monkeypatch.setattr(server, "STATE", state)
    monkeypatch.setattr(server, "_JOB", None)
    monkeypatch.setattr(server.settings, "update", lambda **kwargs: kwargs)
    return state


def endpoint(path):
    return next(route.endpoint for route in server.app.routes if route.path == path)


def timings(path):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT sura, ayah, time FROM timings ORDER BY sura, ayah").fetchall()


def test_job_retains_audio_database_and_logs_when_state_changes(project, tmp_path, monkeypatch):
    from surah_cutter.core import pipeline

    original_db = project.db_path
    original_audio = dict(project.audio_map)
    seen = []
    job = server._GenJob(original_db, project.audio_map)

    def process(audio, surah, **kwargs):
        seen.append(Path(audio))
        project.db_path = tmp_path / "b.db"
        project.audio_map[114] = tmp_path / "b/114.mp3"
        kwargs["log_event"]("test alignment")
        return [(surah, 1, 100)]

    monkeypatch.setattr(pipeline, "process_surah", process)
    # Change the selected DB before the worker even opens its connection.
    project.db_path = tmp_path / "b.db"
    job._run([113, 114], "fast", None)

    assert seen == [original_audio[113], original_audio[114]]
    assert timings(original_db) == [(113, 1, 100), (114, 1, 100)]
    assert not project.db_path.exists()
    assert (tmp_path / "a.logs/sura_114_fast.log").read_text() == "test alignment\n"
    assert not (tmp_path / "b.logs").exists()
    events = list(job.q.queue)
    assert events[-2] == {"type": "done", "completed": 2}


def test_running_generation_rejects_project_switch_and_second_start(project, tmp_path, monkeypatch):
    def start(job, *args):
        job.running = True

    monkeypatch.setattr(server._GenJob, "start", start)
    generate = endpoint("/api/generate")
    generate({"suras": [114]})
    job = server._JOB
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/project/open")(
            {"audio_dir": str(tmp_path), "db_path": str(tmp_path / "b.db")}
        )
    assert exc.value.status_code == 409
    assert project.db_path == tmp_path / "a.db"
    assert 114 in project.audio_map
    with pytest.raises(HTTPException) as exc:
        generate({"suras": [113]})
    assert exc.value.status_code == 409
    assert server._JOB is job


def test_resume_fills_partial_sura_and_preserves_manual_ayahs(project, monkeypatch):
    from surah_cutter.core import pipeline

    with ProjectDB.open(str(project.db_path), DB_CODE, DB_VERSION) as db:
        db.write_timings([(113, a, a * 1000) for a in range(1, 6)], source="auto")
        db.set_time(114, 1, 123)
        db.set_time(114, 2, 456)
        db.set_status(114, 2, "verified")
    statuses = {row["surah"]: row["status"] for row in endpoint("/api/suras")()}
    assert statuses[113] == "processed"
    assert statuses[114] == "partial"
    assert server._processed_suras() == {113}

    monkeypatch.setattr(
        pipeline,
        "process_surah",
        lambda audio, surah, **kw: [(surah, a, a * 1000) for a in range(1, 7)],
    )
    monkeypatch.setattr(server._GenJob, "start", server._GenJob._run)
    result = endpoint("/api/generate")({"only_missing": True})
    assert result["suras"] == [114]
    assert timings(project.db_path)[5:] == [
        (114, 1, 123),
        (114, 2, 456),
        *[(114, a, a * 1000) for a in range(3, 7)],
    ]
    with sqlite3.connect(project.db_path) as conn:
        assert conn.execute(
            "SELECT status FROM review_meta WHERE sura=114 AND ayah<=2 ORDER BY ayah"
        ).fetchall() == [("edited",), ("verified",)]
    assert server._processed_suras() == {113, 114}


def test_wrong_ayah_set_is_partial_even_with_expected_row_count(project):
    with ProjectDB.open(str(project.db_path), DB_CODE, DB_VERSION) as db:
        db.write_timings([(114, a, a * 1000) for a in [1, 2, 3, 4, 5, 999]], source="auto")
    assert server._processed_suras() == set()
    statuses = {row["surah"]: row["status"] for row in endpoint("/api/suras")()}
    assert statuses[114] == "partial"


@pytest.mark.parametrize(
    "statuses,count,expected",
    [
        (None, 6, "not-processed"),
        (["verified"], 6, "partial"),
        (["auto"] * 6, 6, "processed"),
        (["verified"] * 6, 6, "verified"),
    ],
)
def test_classify_completeness(statuses, count, expected):
    assert classify_status(statuses, count) == expected


def test_pause_draft_keeps_original_database_after_decode(project, tmp_path, monkeypatch):
    from surah_cutter.core import decode, draft

    original_db = project.db_path
    with ProjectDB.open(str(original_db), DB_CODE, DB_VERSION) as db:
        db.set_time(114, 1, 123)

    def decode_audio(audio):
        assert Path(audio) == project.audio_map[114]
        project.db_path = tmp_path / "b.db"
        return []

    def draft_audio(pcm, sr, surah, anchors):
        assert anchors == {1: 123}
        return SimpleNamespace(
            rows=[(114, 1, 999), (114, 2, 1000)],
            pause_count=1,
            pause_backed=1,
            interpolated=0,
            anchored=1,
        )

    monkeypatch.setattr(decode, "decode", decode_audio)
    monkeypatch.setattr(draft, "draft_from_pauses", draft_audio)
    assert endpoint("/api/sura/{surah}/draft-pauses")(114)["written"] == 1
    assert timings(original_db) == [(114, 1, 123), (114, 2, 1000)]
    assert not project.db_path.exists()


def test_failed_project_open_does_not_publish_partial_state(project, tmp_path):
    original_map = dict(project.audio_map)
    with pytest.raises(sqlite3.OperationalError):
        endpoint("/api/project/open")({"audio_dir": str(tmp_path), "db_path": str(tmp_path)})
    assert project.db_path == tmp_path / "a.db"
    assert project.audio_dir is None
    assert project.audio_map == original_map
