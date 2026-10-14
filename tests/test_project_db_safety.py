"""Real SQLite writer interleavings, all-or-nothing merges, shared edit guards."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from surah_cutter.data.project_db import ProjectDB


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "project.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(1, 1, 1000), (1, 2, 2000), (1, 3, 3000)], "original", .7)
    return path


def snapshot(db):
    return {table: db._conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in ("timings", "review_meta", "edit_history", "word_timings",
                          "word_edit_history")}


@pytest.mark.parametrize("status", ["edited", "verified"])
def test_committing_review_wins_over_waiting_generator(project, status, monkeypatch):
    started, attempted = Event(), Event()

    def regenerate():
        # Open a separate connection without ProjectDB.open's schema/version writes.
        with ProjectDB(sqlite3.connect(project)) as generator:
            generator._conn.set_trace_callback(
                lambda sql: attempted.set() if sql == "BEGIN IMMEDIATE" else None
            )
            assert started.wait(3)
            return generator.write_timings([(1, 1, 1200)], "new", confidences={1: .1})

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(regenerate)
        with ProjectDB(sqlite3.connect(project)) as editor:
            transaction = editor._write_transaction

            @contextmanager
            def hold_review_transaction():
                with transaction():
                    started.set()
                    assert attempted.wait(3)
                    yield

            monkeypatch.setattr(editor, "_write_transaction", hold_review_transaction)
            if status == "edited":
                editor.set_time(1, 1, 1100)
            else:
                editor.set_status(1, 1, "verified")
        assert result.result(timeout=3) == 0
    with ProjectDB(sqlite3.connect(project)) as db:
        assert db.get_timings()[0] == (1, 1, 1100 if status == "edited" else 1000)
        assert db.get_review(1, 1)["status"] == status
        assert db.get_review(1, 1)["confidence"] == .7


@pytest.mark.parametrize("status", ["edited", "verified"])
def test_status_read_already_holds_writer_lock(project, status, monkeypatch):
    with ProjectDB(sqlite3.connect(project)) as generator:
        with ProjectDB(sqlite3.connect(project, timeout=0)) as editor:
            original = generator._status

            def read_status(sura, ayah):
                value = original(sura, ayah)
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    if status == "edited":
                        editor.set_time(1, 1, 1100)
                    else:
                        editor.set_status(1, 1, "verified")
                return value

            monkeypatch.setattr(generator, "_status", read_status)
            assert generator.write_timings([(1, 1, 1200)], "new") == 1
            if status == "edited":
                editor.set_time(1, 1, 1100)
            else:
                editor.set_status(1, 1, "verified")
            assert editor.get_review(1, 1)["status"] == status
            assert editor.get_timings()[0][2] == (1100 if status == "edited" else 1200)


def test_manual_merge_conflict_rolls_back_every_ayah_and_metadata(project):
    with ProjectDB(sqlite3.connect(project)) as db:
        db.set_time(1, 2, 2500)
        before = snapshot(db)
        with pytest.raises(ValueError, match="порядку"):
            db.write_timings([(1, 1, 1100), (1, 2, 2000), (1, 3, 2400)], "new", .2,
                             confidences={1: .1, 3: .3})
        assert snapshot(db) == before
        assert not db._conn.in_transaction
        # Failure does not poison subsequent saves.
        assert db.write_timings([(1, 1, 1100), (1, 2, 2200), (1, 3, 3100)], "new",
                                confidences={1: .1, 2: .2, 3: .3}) == 2
        assert [db.get_review(1, a)["confidence"] for a in (1, 2, 3)] == [.1, .7, .3]
        assert db.get_review(1, 2)["source"] == "manual"


def test_partial_merge_checks_untouched_neighbours(project):
    with ProjectDB(sqlite3.connect(project)) as db:
        before = snapshot(db)
        with pytest.raises(ValueError):
            db.write_timings([(1, 1, 2100)], "new")
        assert snapshot(db) == before


def test_sql_failure_rolls_back_earlier_writes(project):
    with ProjectDB(sqlite3.connect(project)) as db:
        before = snapshot(db)
        db._conn.execute("CREATE TRIGGER fail_third BEFORE INSERT ON timings "
                         "WHEN NEW.ayah=3 BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            db.write_timings([(1, 1, 1100), (1, 3, 3100)], "new")
        assert snapshot(db) == before


@pytest.mark.parametrize("time", [-1, 1.5, True, "1500", 2**63, 1000, 3000, 3500])
def test_manual_edit_rejects_invalid_time_or_order_without_history(project, time):
    with ProjectDB(sqlite3.connect(project)) as db:
        before = snapshot(db)
        with pytest.raises(ValueError):
            db.set_time(1, 2, time)
        assert snapshot(db) == before


@pytest.mark.parametrize("sura,ayah", [(0, 1), (115, 1), (1, 0), (114, 7), (True, 1), (1, 1.0)])
def test_all_ayah_writers_reject_noncanonical_keys(project, sura, ayah):
    with ProjectDB(sqlite3.connect(project)) as db:
        before = snapshot(db)
        for operation in (lambda: db.set_time(sura, ayah, 0),
                          lambda: db.write_timings([(sura, ayah, 0)], "new"),
                          lambda: db.set_status(sura, ayah, "auto")):
            with pytest.raises(ValueError):
                operation()
            assert snapshot(db) == before


def test_force_cannot_destroy_manual_work(project):
    with ProjectDB(sqlite3.connect(project)) as db:
        db.set_time(1, 2, 2500)
        before = snapshot(db)
        with pytest.raises(ValueError, match="запрещена"):
            db.write_timings([(1, 2, 2000)], "new", force=True)
        assert snapshot(db) == before


def test_missing_timing_cannot_be_marked_reviewed(project):
    with ProjectDB(sqlite3.connect(project)) as db:
        for status in ("edited", "verified"):
            with pytest.raises(ValueError):
                db.set_status(1, 4, status)
        db.set_status(1, 4, "auto")


def test_word_status_read_already_holds_writer_lock(project, monkeypatch):
    with ProjectDB(sqlite3.connect(project)) as generator:
        generator.write_word_timings(1, 1, [(0, 1100, .8), (1, 1500, .9)], 1000, 2000, "old")
        with ProjectDB(sqlite3.connect(project, timeout=0)) as editor:
            original = generator.get_word_timings

            def read_words(sura, ayah):
                rows = original(sura, ayah)
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    editor.set_word_time(1, 1, 0, 1150, 1000, 2000)
                return rows

            monkeypatch.setattr(generator, "get_word_timings", read_words)
            generator.write_word_timings(1, 1, [(0, 1200, .8), (1, 1600, .9)],
                                         1000, 2000, "new")
            editor.set_word_time(1, 1, 0, 1150, 1000, 2000)
            assert editor.get_word_timings(1, 1)[0]["time"] == 1150
