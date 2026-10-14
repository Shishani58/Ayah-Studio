"""Commit is the save boundary, including cancellation and diagnostic failures."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope
from surah_cutter.data import alignment_result
from surah_cutter.data.project_db import ProjectDB


def save(db, path, **kwargs):
    trace = {
        "model_source": "reference:test", "duration_ms": 4000,
        "generated_rows": [(1, 1, 1000), (1, 2, 2000)],
        "ayahs": [{"ayah": 1, "conf": .9}, {"ayah": 2, "conf": .8}],
    }
    trace.update(kwargs.pop("trace", {}))
    return alignment_result.write_alignment_result(
        db, [(1, 1, 1000), (1, 2, 2000)], trace=trace,
        requested_mode="fast", db_path=path, **kwargs,
    )


def snapshot(db):
    return {table: db._conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in ("timings", "review_meta", "edit_history")}


@pytest.mark.parametrize("status", ["edited", "verified"])
def test_manual_range_conflict_rolls_back_whole_surah(tmp_path, status):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(1, 1, 900), (1, 2, 4500)], "previous", .4)
        db.set_status(1, 2, status)
        before = snapshot(db)
        with pytest.raises(ValueError, match="длительности"):
            save(db, path)
        assert snapshot(db) == before
        assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True])
def test_invalid_duration_never_writes(tmp_path, duration):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        with pytest.raises(ValueError, match="Длительность"):
            save(db, path, trace={"duration_ms": duration})
        assert not db.get_timings()


@pytest.mark.parametrize("cancel_at", ["before", "precommit"])
def test_cancellation_preserves_timings_meta_and_history(tmp_path, monkeypatch, cancel_at):
    path = tmp_path / "p.db"
    cancelled = Event()
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(1, 1, 900), (1, 2, 1900)], "previous", .4)
        before = snapshot(db)
        if cancel_at == "before":
            cancelled.set()
        else:
            validate = db._validate_order

            def cancel_after_validation(surah):
                validate(surah)
                cancelled.set()

            monkeypatch.setattr(db, "_validate_order", cancel_after_validation)
        with cancellation_scope(cancelled), pytest.raises(GenerationCancelled):
            save(db, path)
        assert snapshot(db) == before
        assert not db._conn.in_transaction
        assert not list(tmp_path.rglob("*.json"))


def test_postcommit_cancellation_keeps_success_and_diagnostics(tmp_path, monkeypatch):
    path = tmp_path / "p.db"
    cancelled = Event()
    with ProjectDB.open(str(path), "test", 1) as db:
        transaction = db._write_transaction

        @contextmanager
        def cancel_after_commit():
            with transaction():
                yield
            cancelled.set()

        monkeypatch.setattr(db, "_write_transaction", cancel_after_commit)
        with cancellation_scope(cancelled):
            result = save(db, path)
        assert result.outcome == "saved"
        assert result.written == 2
        assert len(db.get_timings()) == 2
        assert json.loads(open(result.log_paths["json"]).read())["rows"] == [
            [1, 1, 1000], [1, 2, 2000],
        ]


@pytest.mark.parametrize("failure", ["stage", "publish", "postcommit_cancel"])
def test_diagnostics_failure_reports_committed_without_retry(tmp_path, monkeypatch, failure):
    path = tmp_path / "p.db"
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        if failure == "postcommit_cancel":
            raise GenerationCancelled()
        raise OSError("diagnostic disk failure")

    if failure == "stage":
        monkeypatch.setattr(alignment_result, "_stage_text", fail)
    else:
        monkeypatch.setattr(alignment_result.os, "replace", fail)
    with ProjectDB.open(str(path), "test", 1) as db:
        result = save(db, path)
        assert db.get_timings() == [(1, 1, 1000), (1, 2, 2000)]
        assert result.outcome == "saved_with_diagnostics_error"
        assert result.diagnostics_error == ("GenerationCancelled: " if failure == "postcommit_cancel"
                                            else "OSError: diagnostic disk failure")
        assert result.log_paths.keys() == {"json", "log", "run"}
        assert result[0] == 2
        written, logs = result
        assert written == 2 and logs == result.log_paths
        assert len(calls) == 1
        assert not [p for p in tmp_path.rglob(".*") if p.is_file()]


@pytest.mark.parametrize("status", ["edited", "verified"])
def test_concurrent_manual_commit_is_captured_with_its_own_confidence(tmp_path, monkeypatch, status):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(1, 1, 900), (1, 2, 1900)], "previous", .3)
    started, attempted = Event(), Event()

    def regenerate():
        with ProjectDB(sqlite3.connect(path)) as generator:
            generator._conn.set_trace_callback(
                lambda sql: attempted.set() if sql == "BEGIN IMMEDIATE" else None
            )
            assert started.wait(3)
            return save(generator, path)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(regenerate)
        with ProjectDB(sqlite3.connect(path)) as editor:
            transaction = editor._write_transaction

            @contextmanager
            def hold_edit():
                with transaction():
                    started.set()
                    assert attempted.wait(3)
                    yield

            monkeypatch.setattr(editor, "_write_transaction", hold_edit)
            if status == "edited":
                editor.set_time(1, 1, 1100)
            else:
                editor.set_status(1, 1, "verified")
            manual = editor.get_review(1, 1)
        saved = result.result(timeout=3)
    expected_time = 1100 if status == "edited" else 900
    assert saved.persisted_rows == ((1, 1, expected_time), (1, 2, 2000))
    assert saved.persisted_review[0]["confidence"] == .3
    assert saved.persisted_review[0]["status"] == status
    with ProjectDB(sqlite3.connect(path)) as db:
        assert db.get_review(1, 1) == manual
    latest = json.loads(open(saved.log_paths["json"]).read())
    assert latest["model_rows"][0] == [1, 1, 1000]
    assert latest["rows"][0] == [1, 1, expected_time]
    assert latest["persisted_review"][0]["confidence"] == .3
    run = json.loads(open(saved.log_paths["run"]).read())
    assert run["rows"][0] == [1, 1, 1000]
    assert run["persisted_rows"][0] == [1, 1, expected_time]
    assert run["confidences"]["1"] == .9


def test_snapshot_is_captured_under_lock_not_reread_after_commit(tmp_path, monkeypatch):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        transaction = db._write_transaction

        @contextmanager
        def edit_after_commit():
            with transaction():
                yield
            with ProjectDB(sqlite3.connect(path)) as editor:
                editor.set_time(1, 1, 1200)

        monkeypatch.setattr(db, "_write_transaction", edit_after_commit)
        result = save(db, path)
        assert db.get_timings()[0] == (1, 1, 1200)
        assert result.persisted_rows[0] == (1, 1, 1000)
        assert result.persisted_review[0]["status"] == "auto"


def test_stale_manual_overlay_does_not_replace_original_model_after_unprotect(tmp_path):
    path = tmp_path / "p.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        db.set_time(1, 1, 800)
        db.set_status(1, 1, "auto")
        result = alignment_result.write_alignment_result(
            db, [(1, 1, 800)], requested_mode="fast", db_path=path,
            trace={"model_source": "reference:test", "generated_rows": [(1, 1, 1000)],
                   "duration_ms": 2000},
        )
        assert result.persisted_rows == ((1, 1, 1000),)


@pytest.mark.parametrize("next_surah", [False, True])
def test_web_reports_saved_surah_before_honouring_postcommit_cancel(tmp_path, monkeypatch, next_surah):
    from surah_cutter.core import pipeline
    from surah_cutter.webgui import server

    path = tmp_path / "p.db"
    job = server._GenJob(path, {114: tmp_path / "114.mp3", 113: tmp_path / "113.mp3"})
    calls = []

    def process(_audio, surah, **kwargs):
        calls.append(surah)
        kwargs["trace"].update(model_source="reference:test", duration_ms=10000)
        return [(surah, a, a * 1000) for a in range(1, 7)]

    original = ProjectDB.write_timings

    def cancel_after_save(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        job.request_cancel()
        return result

    monkeypatch.setattr(pipeline, "process_surah", process)
    monkeypatch.setattr(ProjectDB, "write_timings", cancel_after_save)
    job._run([114, 113] if next_surah else [114], "fast", None)
    events = job.events.snapshot()
    saved = next(e for e in events if e["type"] == "sura")
    assert saved["written"] == 6 and saved["outcome"] == "saved"
    assert events[-1] == {"type": "cancelled" if next_surah else "done", "completed": 1}
    assert calls == [114]
    with ProjectDB(sqlite3.connect(path)) as db:
        assert len(db.get_timings()) == 6


def test_web_diagnostic_failure_remains_saved_in_event_and_terminal(tmp_path, monkeypatch):
    from surah_cutter.core import pipeline
    from surah_cutter.webgui import server

    path = tmp_path / "p.db"
    job = server._GenJob(path, {114: tmp_path / "114.mp3"})
    calls = []

    def process(_audio, surah, **kwargs):
        calls.append(surah)
        kwargs["trace"]["model_source"] = "reference:test"
        return [(surah, a, a * 1000) for a in range(1, 7)]

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "process_surah", process)
    monkeypatch.setattr(alignment_result.os, "replace", fail)
    job._run([114], "fast", None)
    events = job.events.snapshot()
    saved = next(e for e in events if e["type"] == "sura")
    assert saved["outcome"] == "saved_with_diagnostics_error"
    assert saved["written"] == 6
    assert events[-1]["type"] == "done" and events[-1]["completed"] == 1
    assert events[-1]["diagnostics_errors"][0]["error"] == "OSError: disk full"
    assert calls == [114]


def test_cli_diagnostic_failure_does_not_report_unsaved_or_retry(tmp_path, monkeypatch, capsys):
    from surah_cutter import cli
    from surah_cutter.core import pipeline

    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "114.mp3").touch()
    path = tmp_path / "p.db"
    calls = []

    def process(_audio, surah, **kwargs):
        calls.append(surah)
        kwargs["trace"]["model_source"] = "reference:test"
        return [(surah, a, a * 1000) for a in range(1, 7)]

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "process_surah", process)
    monkeypatch.setattr(alignment_result.os, "replace", fail)
    assert cli.main([str(audio), "-o", str(path), "--mode", "fast"]) == 0
    output = capsys.readouterr()
    assert "saved_with_diagnostics_error" in output.err
    assert "FAILED" not in output.err
    assert "6 ayah timings written" in output.out
    assert calls == [114]


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_unsaved_trace_never_becomes_successful_run(tmp_path, outcome):
    from surah_cutter.data.alignment_runs import list_runs

    path = tmp_path / "p.db"
    trace = {"acoustic_reads": [{"window_ms": [100, 900], "outcome": outcome}],
             "transitions": [{"state": "READ_LOCAL"}]}
    result = alignment_result.write_unsaved_diagnostics(
        db_path=path, surah=114, requested_mode="fast", trace=trace,
        outcome=outcome, error=GenerationCancelled() if outcome == "cancelled" else ValueError("model error"),
    )
    payload = json.loads(result.read_text())
    assert payload["saved"] is False and payload["outcome"] == outcome
    assert payload["trace"] == trace
    assert result.parent.name == "failures"
    assert list_runs(tmp_path / "p.logs", 114) == []
    assert not path.exists()


def test_unsaved_trace_failure_preserves_original_exception(tmp_path, monkeypatch):
    primary = ValueError("original model error")

    def fail(*args):
        raise OSError("diagnostics error")

    monkeypatch.setattr(alignment_result, "_stage_text", fail)
    with pytest.raises(ValueError, match="original model error") as raised:
        try:
            raise primary
        except ValueError as exc:
            assert alignment_result.write_unsaved_diagnostics(
                db_path=tmp_path / "p.db", surah=114, requested_mode="fast",
                trace={}, outcome="failed", error=exc,
            ) is None
            raise
    assert raised.value is primary


@pytest.mark.parametrize("cancelled", [False, True])
def test_web_failed_model_retains_ledger_without_saving_timings(tmp_path, monkeypatch, cancelled):
    from surah_cutter.core import pipeline
    from surah_cutter.webgui import server

    path = tmp_path / "p.db"
    job = server._GenJob(path, {114: tmp_path / "114.mp3"})
    outcome = "cancelled" if cancelled else "failed"

    def process(*args, **kwargs):
        kwargs["trace"]["acoustic_reads"] = [{"window_ms": [0, 700], "outcome": outcome}]
        if cancelled:
            raise GenerationCancelled()
        raise ValueError("original model failure")

    monkeypatch.setattr(pipeline, "process_surah", process)
    job._run([114], "fast", None)
    terminal = job.events.snapshot()[-1]
    assert terminal == ({"type": "cancelled", "completed": 0} if cancelled
                        else {"type": "error", "message": "original model failure"})
    diagnostic, = (tmp_path / "p.logs/failures").glob("*.json")
    assert json.loads(diagnostic.read_text())["trace"]["acoustic_reads"][0]["outcome"] == outcome
    with ProjectDB.open(str(path), "test", 1) as db:
        assert db.get_timings() == []
    assert not (tmp_path / "p.logs/runs").exists()
