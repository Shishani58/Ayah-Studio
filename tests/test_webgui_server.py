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


def test_saved_sura_logs_list_pairs_and_read_full_data(project):
    lines = [f"line {i}" for i in range(300)]
    paths = server._save_sura_log(114, "fast", {"phase": "complete"}, lines, [], project.db_path)
    server._save_sura_log(113, "full", {}, ["other sura"], [], project.db_path)
    folder = Path(paths["log"]).parent
    (folder / "sura_114_words.json").write_text('{"words": []}', encoding="utf-8")
    entries = endpoint("/api/sura/{surah}/logs")(114)["logs"]
    assert {item["id"] for item in entries} == {"sura_114_fast", "sura_114_words"}
    result = endpoint("/api/sura/{surah}/logs/{log_id}")(114, "sura_114_fast")
    assert result["log"].splitlines() == lines
    assert '"phase": "complete"' in result["json"]
    json_only = endpoint("/api/sura/{surah}/logs/{log_id}")(114, "sura_114_words")
    assert json_only["log"] is None
    assert json_only["json"] == '{"words": []}'


def test_saved_sura_logs_reject_other_sura_paths_and_links(project, tmp_path):
    folder = project.db_path.with_name(f"{project.db_path.stem}.logs")
    folder.mkdir()
    secret = tmp_path / "secret.log"
    secret.write_text("private", encoding="utf-8")
    (folder / "sura_114_fast.log").symlink_to(secret)
    assert endpoint("/api/sura/{surah}/logs")(114) == {"logs": []}
    read = endpoint("/api/sura/{surah}/logs/{log_id}")
    for log_id in ("sura_113_fast", "../secret", "sura_114_fast", "sura_114_missing"):
        with pytest.raises(HTTPException) as error:
            read(114, log_id)
        assert error.value.status_code == 404
    assert endpoint("/api/sura/{surah}/logs")(112) == {"logs": []}


def test_saved_sura_logs_require_open_project(project):
    project.db_path = None
    with pytest.raises(HTTPException) as error:
        endpoint("/api/sura/{surah}/logs")(114)
    assert error.value.status_code == 400


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


def test_full_job_saves_trace_confidence_and_preserves_manual_rows(project, monkeypatch):
    from surah_cutter.core import pipeline

    with ProjectDB.open(str(project.db_path), DB_CODE, DB_VERSION) as db:
        db.set_time(114, 1, 6000)
        db.set_status(114, 1, "verified")

    def process(audio, surah, **kwargs):
        kwargs["trace"].update({
            "mode": "full",
            "ayahs": [{"ayah": 1, "conf": 0.01}, {"ayah": 2, "conf": 0.15}],
        })
        kwargs["log_event"]("weak-onset")
        return [(114, 1, 5300), (114, 2, 10260)]

    monkeypatch.setattr(pipeline, "process_surah", process)
    job = server._GenJob(project.db_path, project.audio_map)
    job._run([114], "full", None)

    with ProjectDB.open(str(project.db_path), DB_CODE, DB_VERSION) as db:
        meta = {a: db.get_review(114, a) for a in (1, 2)}
        assert (114, 1, 6000) in db.get_timings()
        assert meta[1]["status"] == "verified"
        assert meta[1]["confidence"] is None
        assert meta[2]["confidence"] == 0.15
    import json

    saved = json.loads(project.db_path.with_name("a.logs").joinpath("sura_114_full.json").read_text())
    assert saved["trace"]["ayahs"][0]["conf"] == 0.01


def test_full_draft_correction_keeps_confidence_diagnostics(project, monkeypatch):
    from surah_cutter.core import pipeline

    def process(audio, surah, **kwargs):
        kwargs["trace"]["ayahs"] = [{"ayah": 1, "conf": 0.03, "reliable": False}]
        kwargs["log_event"]("weak-onset")
        return [(surah, 1, 6100)]

    monkeypatch.setattr(pipeline, "process_surah", process)
    result = endpoint("/api/sura/{surah}/correct-draft")(114, {"mode": "full"})

    assert result["written"] == 1
    with ProjectDB.open(str(project.db_path), DB_CODE, DB_VERSION) as db:
        assert db.get_review(114, 1)["confidence"] == 0.03
    assert project.db_path.with_name("a.logs").joinpath("sura_114_correct-full.log").read_text() == "weak-onset\n"


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


@pytest.fixture
def word_project(project, monkeypatch):
    from surah_cutter.core import decode

    with ProjectDB.open(str(project.db_path), DB_CODE, DB_VERSION) as db:
        db.write_timings([(114, 1, 1000), (114, 2, 5000)], "test")
    monkeypatch.setattr(decode, "duration_ms", lambda _: 30000)
    return project


def test_word_api_manual_edit_verify_stale_and_bounds(word_project):
    get = endpoint("/api/sura/{surah}/ayah/{ayah}/words")
    edit = endpoint("/api/sura/{surah}/ayah/{ayah}/words/edit")
    data = get(114, 1)
    words = server.quran_text.surah_words(114)
    assert [w["text"] for w in data["words"]] == [w.text for w in words if w.ayah == 1]
    payload = {"db_path": data["db_path"], "start_ms": 1000, "end_ms": 5000,
               "word_index": data["words"][0]["word_index"], "time": 1200}
    result = edit(114, 1, payload)
    assert result["words"][0]["status"] == "edited"
    assert edit(114, 1, {**payload, "action": "verify"})["words"][0]["status"] == "verified"
    for invalid in [True, -1, 5000, 1200.5]:
        with pytest.raises(HTTPException) as exc:
            edit(114, 1, {**payload, "time": invalid})
        assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        edit(114, 1, {**payload, "word_index": 9999})
    with ProjectDB.open(str(word_project.db_path), DB_CODE, DB_VERSION) as db:
        db.set_time(114, 2, 5500)
    assert get(114, 1)["words"][0]["stale"]
    with pytest.raises(HTTPException) as exc:
        edit(114, 1, payload)
    assert exc.value.status_code == 409
    payload["end_ms"] = 5500
    with pytest.raises(HTTPException):
        edit(114, 1, {**payload, "action": "verify"})
    assert not edit(114, 1, payload)["words"][0]["stale"]


def test_word_api_requires_adjacent_boundaries_and_current_project(word_project):
    get = endpoint("/api/sura/{surah}/ayah/{ayah}/words")
    with pytest.raises(HTTPException):
        get(114, 2)  # No ayah 3; cannot use the end of the surah for ayah 2.
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/sura/{surah}/ayah/{ayah}/words/edit")(114, 1, {"db_path": "wrong"})
    assert exc.value.status_code == 409


def test_word_alignment_rejects_changed_project_before_write(word_project, monkeypatch):
    from surah_cutter.core import decode, word_align
    import numpy as np

    get = endpoint("/api/sura/{surah}/ayah/{ayah}/words")
    data = get(114, 1)
    payload = {"db_path": data["db_path"], "start_ms": 1000, "end_ms": 5000}
    monkeypatch.setattr(decode, "decode_range", lambda *a, **kw: np.zeros(64000))

    def compute(pcm, sr, words):
        with ProjectDB.open(str(word_project.db_path), DB_CODE, DB_VERSION) as db:
            db.set_time(114, 2, 5500)
        return [(w.index, 100 + i * 500, .9) for i, w in enumerate(words)]

    monkeypatch.setattr(word_align, "align_ayah_words", compute)
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/sura/{surah}/ayah/{ayah}/words/align")(114, 1, payload)
    assert exc.value.status_code == 409
    with ProjectDB.open(str(word_project.db_path), DB_CODE, DB_VERSION) as db:
        assert db.get_word_timings(114, 1) == []
    assert not server._WORD_LOCK.locked()


def test_word_alignment_saves_all_words_without_changing_ayahs(word_project, monkeypatch):
    from surah_cutter.core import decode, word_align
    import numpy as np

    data = endpoint("/api/sura/{surah}/ayah/{ayah}/words")(114, 1)
    payload = {"db_path": data["db_path"], "start_ms": 1000, "end_ms": 5000}
    monkeypatch.setattr(decode, "decode_range", lambda *a, **kw: np.zeros(64000))
    monkeypatch.setattr(word_align, "align_ayah_words", lambda pcm, sr, words:
                        [(w.index, 100 + i * 500, .9) for i, w in enumerate(words)])
    before = timings(word_project.db_path)
    result = endpoint("/api/sura/{surah}/ayah/{ayah}/words/align")(114, 1, payload)
    assert all(w["status"] == "auto" for w in result["words"])
    assert len(result["words"]) == len(data["words"])
    assert timings(word_project.db_path) == before


@pytest.fixture
def complete_word_project(word_project):
    with ProjectDB.open(str(word_project.db_path), DB_CODE, DB_VERSION) as db:
        db.write_timings([(114, a, t) for a, t in enumerate(
            [1000, 5000, 10000, 15000, 20000, 25000], 1)], "test")
    return word_project


def test_word_sura_job_keeps_ayahs_and_preserves_verified_word(complete_word_project, monkeypatch):
    from surah_cutter.core import decode, pipeline, word_align
    import numpy as np

    monkeypatch.setattr(decode, "decode", lambda _: np.zeros(480000))
    def unexpected(*args, **kwargs):
        pytest.fail("Complete ayah marks must not be regenerated in words mode")
    monkeypatch.setattr(pipeline, "process_surah", unexpected)
    monkeypatch.setattr(word_align, "align_ayah_words", lambda pcm, sr, words:
                        [(w.index, 50 + i * 500, .9) for i, w in enumerate(words)])
    path = complete_word_project.db_path
    with ProjectDB.open(str(path), DB_CODE, DB_VERSION) as db:
        db.set_word_time(114, 1, 0, 1100, 1000, 5000)
        db.verify_word(114, 1, 0, 1000, 5000)
    before = timings(path)
    job = server._GenJob(path, complete_word_project.audio_map)
    job._run([114], "words", None)
    assert timings(path) == before
    event = next(e for e in job.q.queue if e and e["type"] == "sura")
    assert event["words_written"] == len(server.quran_text.surah_words(114)) - 1
    assert event["word_failures"] == 0
    assert event["written"] == 0
    with ProjectDB.open(str(path), DB_CODE, DB_VERSION) as db:
        assert db.get_word_timings(114, 1)[0]["time"] == 1100
        assert db.get_word_timings(114, 1)[0]["status"] == "verified"
    assert server._word_progress()[114]["complete"]
    with ProjectDB.open(str(path), DB_CODE, DB_VERSION) as db:
        db.set_time(114, 2, 5100)
    assert not server._word_progress()[114]["complete"]


def test_word_resume_uses_words_not_ayah_completeness(complete_word_project, monkeypatch):
    started = []
    monkeypatch.setattr(server._GenJob, "start", lambda job, suras, mode, reciter:
                        started.append((suras, mode)))
    result = endpoint("/api/generate")({"suras": [114], "mode": "words", "only_missing": True})
    assert result["mode"] == "words"
    assert started == [([114], "words")]
    with ProjectDB.open(str(complete_word_project.db_path), DB_CODE, DB_VERSION) as db:
        marks = {a: t for s, a, t in db.get_timings() if s == 114}
        for a in range(1, 7):
            words = [w for w in server.quran_text.surah_words(114) if w.ayah == a]
            db.write_word_timings(114, a, [(w.index, marks[a] + i * 100, .8)
                                          for i, w in enumerate(words)],
                                  marks[a], marks.get(a + 1, 30000), "test")
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/generate")({"suras": [114], "mode": "words", "only_missing": True})
    assert exc.value.status_code == 400
    assert endpoint("/api/settings")({"mode": "words"})["mode"] == "words"


def test_word_sura_job_reports_unaligned_ayah_and_continues(complete_word_project, monkeypatch):
    from surah_cutter.core import decode, word_align
    import numpy as np

    monkeypatch.setattr(decode, "decode", lambda _: np.zeros(480000))
    def compute(pcm, sr, words):
        if words[0].ayah == 3:
            raise ValueError("incomplete path")
        return [(w.index, 50 + i * 500, .9) for i, w in enumerate(words)]
    monkeypatch.setattr(word_align, "align_ayah_words", compute)
    job = server._GenJob(complete_word_project.db_path, complete_word_project.audio_map)
    job._run([114], "words", None)
    event = next(e for e in job.q.queue if e and e["type"] == "sura")
    assert event["word_failures"] == 1
    assert not server._word_progress()[114]["complete"]
    with ProjectDB.open(str(complete_word_project.db_path), DB_CODE, DB_VERSION) as db:
        assert not db.get_word_timings(114, 3)
        assert db.get_word_timings(114, 6)


def test_word_mode_builds_missing_ayah_marks_with_fast(project, monkeypatch):
    from surah_cutter.core import pipeline

    seen = []
    def process(audio, surah, **kwargs):
        seen.append(kwargs["mode"])
        return [(114, a, a * 1000) for a in range(1, 7)]
    monkeypatch.setattr(pipeline, "process_surah", process)
    monkeypatch.setattr(server, "_align_sura_words", lambda *args:
                        {"words_written": 20, "word_failures": 0})
    job = server._GenJob(project.db_path, project.audio_map)
    job._run([114], "words", None)
    assert seen == ["fast"]
    assert len(timings(project.db_path)) == 6



def test_sura_words_returns_one_canonical_list_with_edit_context(complete_word_project):
    get = endpoint("/api/sura/{surah}/words")
    with ProjectDB.open(str(complete_word_project.db_path), DB_CODE, DB_VERSION) as db:
        db.set_word_time(114, 1, 0, 1200, 1000, 5000)
    data = get(114)
    canonical = server.quran_text.surah_words(114)
    assert [w["text"] for w in data["words"]] == [w.text for w in canonical]
    assert [w["word_index"] for w in data["words"]] == [w.index for w in canonical]
    assert [w["number"] for w in data["words"][:4]] == [1, 2, 3, 4]
    assert data["words"][4]["ayah"] == 2
    assert data["words"][4]["number"] == 1
    assert (data["words"][0]["start_ms"], data["words"][0]["end_ms"]) == (1000, 5000)
    assert data["words"][-1]["end_ms"] == 30000
    assert not data["words"][0]["stale"]
    with ProjectDB.open(str(complete_word_project.db_path), DB_CODE, DB_VERSION) as db:
        db.set_time(114, 2, 5100)
    assert get(114)["words"][0]["stale"]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_settings_theme_survives_reload(project, tmp_path, monkeypatch, theme):
    from surah_cutter.webgui import settings

    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "settings.json")
    # The shared fixture stubs persistence; restore it for this regression.
    def persist(**fields):
        strict = fields.pop("strict", False)
        data = settings.load()
        data.update(fields)
        settings.save(data, strict=strict)
        return data

    monkeypatch.setattr(settings, "update", persist)
    settings.save({"mode": "words", "arabic_size": 36})
    assert endpoint("/api/settings")({"theme": theme})["theme"] == theme
    assert endpoint("/api/project")()["theme"] == theme
    assert settings.load()["mode"] == "words"
    assert settings.load()["arabic_size"] == 36


@pytest.mark.parametrize("theme", ["system", "", None, 1, []])
def test_settings_rejects_invalid_theme(project, theme):
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/settings")({"theme": theme})
    assert exc.value.status_code == 400


def test_settings_defaults_to_dark(project, monkeypatch):
    monkeypatch.setattr(server.settings, "load", lambda: {})
    assert endpoint("/api/project")()["theme"] == "dark"
