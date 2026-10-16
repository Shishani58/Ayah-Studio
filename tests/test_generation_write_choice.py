"""Explicit Start choices replace human marks only after successful alignment."""

import json
import sqlite3

import numpy as np
import pytest
from fastapi import HTTPException

from surah_cutter.core import decode, pipeline, word_align
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.data.quran_text import surah_words
from surah_cutter.webgui import server
from surah_cutter.webgui.state import AppState, DB_CODE, DB_VERSION


def endpoint(path):
    return next(route.endpoint for route in server.app.routes if route.path == path)


@pytest.fixture
def project(tmp_path, monkeypatch):
    state = AppState()
    state.set_db_path(str(tmp_path / "choice.db"))
    state.audio_map = {114: tmp_path / "114.wav"}
    with ProjectDB.open(str(state.db_path), DB_CODE, DB_VERSION) as db:
        db.write_timings([(114, a, a * 4000) for a in range(1, 7)], "original")
        db.set_time(114, 1, 3900)
        db.set_status(114, 2, "verified")
        word = surah_words(114)[0]
        db.set_word_time(114, 1, word.index, 4000, 3900, 8000)
    monkeypatch.setattr(server, "STATE", state)
    monkeypatch.setattr(server, "_JOB", None)
    monkeypatch.setattr(server.settings, "update", lambda **kwargs: kwargs)
    monkeypatch.setattr(server.model_store, "active_kind", lambda: "phonetic")
    monkeypatch.setattr(server._GenJob, "start", server._GenJob._run)
    return state.db_path


@pytest.mark.parametrize("overwrite", [False, True])
def test_generation_choice_and_history(project, monkeypatch, overwrite):
    def align(audio, surah, **kwargs):
        assert (kwargs["project_db"] is None) == overwrite
        kwargs["trace"].update(mode="full", ayahs=[{"ayah": a, "conf": .9}
                                                   for a in range(1, 7)])
        return [(surah, a, a * 4000 + 100) for a in range(1, 7)]

    monkeypatch.setattr(pipeline, "process_surah", align)
    with ProjectDB.open(str(project), DB_CODE, DB_VERSION) as db:
        protected = [db.get_review(114, a) for a in (1, 2)]
    endpoint("/api/generate")({"suras": [114], "mode": "full",
                               "overwrite_manual": overwrite, "db_path": str(project)})
    with ProjectDB.open(str(project), DB_CODE, DB_VERSION) as db:
        assert [r[2] for r in db.get_timings()][:2] == ([4100, 8100] if overwrite else [3900, 8000])
        reviews = [db.get_review(114, a) for a in (1, 2)]
        if overwrite:
            assert [r["status"] for r in reviews] == ["auto", "auto"]
            assert [r["confidence"] for r in reviews] == [.9, .9]
        else:
            assert reviews == protected
        history = db._conn.execute(
            "SELECT ayah, old_time, new_time FROM edit_history "
            "WHERE action='generation_overwrite' ORDER BY ayah"
        ).fetchall()
        assert history == ([(1, 3900, 4100), (2, 8000, 8100)] if overwrite else [])
    trace = json.loads(project.with_name("choice.logs").joinpath("sura_114_full.json").read_text())
    assert trace["overwrite_manual"] is overwrite


def test_preview_matches_resume_scope_and_counts_stale_words(project):
    preview = endpoint("/api/generate/preview")
    assert preview({"suras": [114], "mode": "full"}) == {
        "suras": [114], "manual_ayahs": 2, "manual_words": 0, "db_path": str(project),
    }
    with ProjectDB.open(str(project), DB_CODE, DB_VERSION) as db:
        db.set_time(114, 1, 3800)  # Word snapshot is now stale but still protected.
    assert preview({"suras": [114], "mode": "words"})["manual_words"] == 1
    with pytest.raises(HTTPException, match="unknown mode"):
        preview({"suras": [114], "mode": "auto", "only_missing": True})


@pytest.mark.parametrize("value", ["true", 1, None])
def test_invalid_overwrite_never_starts(project, value):
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/generate")({"suras": [114], "mode": "full", "overwrite_manual": value})
    assert exc.value.status_code == 400
    assert server._JOB is None


def test_changed_project_never_starts(project):
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/generate")({"suras": [114], "mode": "full", "overwrite_manual": True,
                                   "db_path": str(project.with_name("other.db"))})
    assert exc.value.status_code == 409
    assert server._JOB is None


@pytest.mark.parametrize("failure", ["alignment", "order", "cancel"])
def test_failed_overwrite_leaves_data_and_history_unchanged(project, monkeypatch, failure):
    def align(*args, **kwargs):
        if failure == "alignment":
            raise ValueError("failed alignment")
        if failure == "cancel":
            server._JOB.request_cancel()
            return [(114, 1, 4100)]
        return [(114, 1, 9000)]  # Conflicts with the still saved second ayah.

    monkeypatch.setattr(pipeline, "process_surah", align)
    def snapshot():
        with sqlite3.connect(project) as conn:
            return [conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                    for table in ("timings", "review_meta", "edit_history", "word_timings")]
    before = snapshot()
    endpoint("/api/generate")({"suras": [114], "mode": "full", "overwrite_manual": True})
    assert snapshot() == before
    assert server._JOB.events.status()["terminal"]["type"] == ("cancelled" if failure == "cancel" else "error")


@pytest.mark.parametrize("overwrite", [False, True])
def test_words_choice_replaces_only_with_explicit_choice(project, monkeypatch, overwrite):
    monkeypatch.setattr(decode, "decode", lambda _: np.zeros(480000))
    monkeypatch.setattr(pipeline, "process_surah", lambda audio, surah, **kwargs:
                        [(surah, a, a * 4000) for a in range(1, 7)])
    monkeypatch.setattr(word_align, "align_ayah_words", lambda pcm, sr, words:
                        [(w.index, 50 + i * 100, .9) for i, w in enumerate(words)])
    endpoint("/api/generate")({"suras": [114], "mode": "words", "overwrite_manual": overwrite})
    with ProjectDB.open(str(project), DB_CODE, DB_VERSION) as db:
        mark = db.get_word_timings(114, 1)[0]
        assert (mark["time"], mark["status"]) == ((4050, "auto") if overwrite else (4000, "edited"))
        assert db.get_timings()[0][2] == (4000 if overwrite else 3900)
        history = db._conn.execute(
            "SELECT old_time, new_time FROM word_edit_history WHERE action='generation_overwrite'"
        ).fetchall()
        assert history == ([(4000, 4050)] if overwrite else [])
