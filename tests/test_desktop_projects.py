import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException

from surah_cutter.data.project_db import ProjectDB
from surah_cutter.webgui import projects, server, settings
from surah_cutter.webgui.state import AppState, DB_CODE, DB_VERSION


@pytest.fixture
def local_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "config/settings.json")
    monkeypatch.setattr(server, "STATE", AppState())
    monkeypatch.setattr(server, "_JOB", None)
    audio = tmp_path / "source"
    audio.mkdir()
    (audio / "112.mp3").write_bytes(b"audio source")
    return audio


def endpoint(path):
    return next(route.endpoint for route in server.app.routes if route.path == path)


def test_import_is_independent_and_preserves_manual_history(local_projects, tmp_path):
    source = tmp_path / "source.db"
    with ProjectDB.open(str(source), DB_CODE, DB_VERSION) as db:
        db.set_time(112, 1, 123)
        db.set_status(112, 1, "verified")
    before = source.read_bytes()
    project = projects.create("Чтец", "Запись", str(local_projects), str(source))
    destination = Path(project["db_path"])
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT status FROM review_meta").fetchone() == ("verified",)
        assert conn.execute("SELECT COUNT(*) FROM edit_history").fetchone() == (1,)
    with ProjectDB.open(str(destination), DB_CODE, DB_VERSION) as db:
        db.set_time(112, 1, 456)
    assert source.read_bytes() == before
    imported_audio = Path(project["audio_dir"]) / "112.mp3"
    (local_projects / "112.mp3").write_bytes(b"changed")
    assert imported_audio.read_bytes() == b"audio source"


def test_failed_import_leaves_no_project(local_projects, tmp_path):
    source = tmp_path / "bad.db"
    source.write_bytes(b"not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        projects.create("Чтец", "Запись", str(local_projects), str(source))
    assert projects.list_projects() == []
    assert list(projects.root().iterdir()) == []


def test_activation_and_preferences_survive_restart(local_projects):
    create = endpoint("/api/projects/create")
    first = create({"reciter": "Первый", "recording": "Одна", "audio_dir": str(local_projects)})
    second = create({"reciter": "Второй", "recording": "Другая", "audio_dir": str(local_projects)})
    assert first["db_path"] != second["db_path"]
    endpoint("/api/projects/activate")({"id": first["project_id"]})
    endpoint("/api/settings")({"time_step": 50, "arabic_size": 36, "mode": "full"})
    server.STATE = AppState()
    server._restore_from_settings()
    restored = endpoint("/api/project")()
    assert restored["project_id"] == first["project_id"]
    assert restored["time_step"] == 50
    assert restored["arabic_size"] == 36


def test_generation_blocks_import_without_copying(local_projects, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(server, "_JOB", SimpleNamespace(running=True))
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/projects/create")({"reciter": "Чтец", "recording": "Запись",
                                           "audio_dir": str(local_projects)})
    assert exc.value.status_code == 409
    assert not projects.root().exists()


@pytest.mark.parametrize("time", [-1, 100, 300, 301, 2.5, True])
def test_inline_edit_rejects_invalid_or_crossed_boundaries(local_projects, time):
    project = endpoint("/api/projects/create")(
        {"reciter": "Чтец", "recording": "Запись", "audio_dir": str(local_projects)}
    )
    with ProjectDB.open(project["db_path"], DB_CODE, DB_VERSION) as db:
        for ayah, start in enumerate((100, 200, 300), 1):
            db.set_time(112, ayah, start)
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/timing")({"surah": 112, "ayah": 2, "time": time})
    assert exc.value.status_code == 400
    with sqlite3.connect(project["db_path"]) as conn:
        assert conn.execute("SELECT time FROM timings WHERE ayah=2").fetchone() == (200,)


def test_edit_cannot_write_into_newly_selected_project(local_projects):
    create = endpoint("/api/projects/create")
    first = create({"reciter": "А", "recording": "1", "audio_dir": str(local_projects)})
    second = create({"reciter": "Б", "recording": "1", "audio_dir": str(local_projects)})
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/timing")({"surah": 112, "ayah": 1, "time": 10, "db_path": first["db_path"]})
    assert exc.value.status_code == 409
    with sqlite3.connect(second["db_path"]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM timings").fetchone() == (0,)


def test_failed_persistence_keeps_current_selection(local_projects, monkeypatch):
    original = endpoint("/api/projects/create")(
        {"reciter": "Первый", "recording": "1", "audio_dir": str(local_projects)}
    )
    other = projects.create("Второй", "1", str(local_projects))

    def fail(**kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr(settings, "update", fail)
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/projects/activate")({"id": other["id"]})
    assert exc.value.status_code == 500
    assert server.STATE.db_path == Path(original["db_path"])
