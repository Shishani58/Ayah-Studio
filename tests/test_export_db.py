"""Export must preserve project data and reject invalid distributable timings."""

import sqlite3

import pytest

from surah_cutter.data.export_db import ExportError, export_db
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.data.quran_text import surah_ayah_counts


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "project.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings(
            ((s, a, (a - 1) * 1000)
             for s, count in surah_ayah_counts().items()
             for a in range(1, count + 1)),
            source="test",
        )
        db.set_time(1, 2, 1100)
        db.set_status(1, 3, "verified")
    return path


def test_export_strips_only_service_tables_and_keeps_source(source, tmp_path):
    before = source.read_bytes()
    destination = tmp_path / "output.db"
    destination.write_bytes(b"previous export")
    assert export_db(str(source), str(destination)) == str(destination)
    assert source.read_bytes() == before
    with sqlite3.connect(destination) as db, sqlite3.connect(source) as original:
        assert set(db.execute("SELECT name FROM sqlite_master WHERE type='table'")) == {
            ("timings",), ("db_version",),
        }
        for table in ("timings", "db_version"):
            assert sorted(db.execute(f"SELECT * FROM {table}")) == sorted(
                original.execute(f"SELECT * FROM {table}"))


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
@pytest.mark.parametrize("strict", [True, False])
def test_refuses_source_aliases_without_losing_manual_work(source, tmp_path, alias, strict):
    destination = source
    if alias == "relative":
        destination = tmp_path / "subdir" / ".." / "project.db"
        (tmp_path / "subdir").mkdir()
    elif alias == "symlink":
        destination = tmp_path / "link.db"
        try:
            destination.symlink_to(source)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows account lacks symlink creation privilege")
            raise
    elif alias == "hardlink":
        destination = tmp_path / "link.db"
        destination.hardlink_to(source)
    before = source.read_bytes()
    with pytest.raises(ExportError, match="source project"):
        export_db(str(source), str(destination), strict=strict)
    assert source.read_bytes() == before


@pytest.mark.parametrize("sql", [
    "UPDATE timings SET time=-1 WHERE sura=1 AND ayah=1",
    "UPDATE timings SET time=0 WHERE sura=1 AND ayah=2",
    "UPDATE timings SET time=500 WHERE sura=1 AND ayah=3",
    "UPDATE timings SET time=0.5 WHERE sura=1 AND ayah=1",
    "UPDATE timings SET time=NULL WHERE sura=1 AND ayah=1",
    "UPDATE timings SET time='invalid' WHERE sura=1 AND ayah=1",
    "UPDATE timings SET ayah=999 WHERE sura=1 AND ayah=7",
    "UPDATE timings SET ayah=7.5 WHERE sura=1 AND ayah=7",
    "UPDATE timings SET sura=115 WHERE sura=114",
    "DELETE FROM timings WHERE sura=1 AND ayah=7",
    "DROP INDEX idx_timings_sura_ayah; INSERT INTO timings VALUES(1,1,0)",
])
def test_strict_rejects_invalid_rows_without_touching_destination(source, tmp_path, sql):
    with sqlite3.connect(source) as db:
        db.executescript(sql)
    destination = tmp_path / "output.db"
    destination.write_bytes(b"keep me")
    with pytest.raises(ExportError):
        export_db(str(source), str(destination))
    assert destination.read_bytes() == b"keep me"
    assert not list(tmp_path.glob(".output.db.*"))


def test_non_strict_still_allows_partial_export(source, tmp_path):
    with sqlite3.connect(source) as db:
        db.execute("DELETE FROM timings WHERE sura != 1")
    destination = tmp_path / "partial.db"
    export_db(str(source), str(destination), strict=False)
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT COUNT(*) FROM timings").fetchone() == (7,)


def test_failed_write_preserves_destination_and_removes_temporary(source, tmp_path):
    # An imported malformed version table causes a failure after timing writes.
    with sqlite3.connect(source) as db:
        db.executescript("DROP TABLE db_version; CREATE TABLE db_version(code, version);"
                         "INSERT INTO db_version VALUES ('test', NULL);")
    destination = tmp_path / "output.db"
    destination.write_bytes(b"previous export")
    with pytest.raises(sqlite3.IntegrityError):
        export_db(str(source), str(destination))
    assert destination.read_bytes() == b"previous export"
    assert not list(tmp_path.glob(".output.db.*"))


def test_failed_replace_preserves_destination_and_removes_temporary(source, tmp_path, monkeypatch):
    destination = tmp_path / "output.db"
    destination.write_bytes(b"previous export")

    def fail_replace(*args):
        raise PermissionError("destination is busy")

    monkeypatch.setattr("surah_cutter.data.export_db.os.replace", fail_replace)
    with pytest.raises(PermissionError):
        export_db(str(source), str(destination))
    assert destination.read_bytes() == b"previous export"
    assert not list(tmp_path.glob(".output.db.*"))


def test_missing_source_is_not_created(tmp_path):
    source = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        export_db(str(source), str(tmp_path / "output.db"))
    assert not source.exists()
