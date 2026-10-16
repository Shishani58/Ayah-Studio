"""Concurrent UI/model/project writers retain independently confirmed fields."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from surah_cutter.webgui import settings


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)
    return path


def test_removed_auto_setting_migrates_to_full_without_changing_project(settings_file, monkeypatch):
    from surah_cutter.webgui import server

    settings.save({"mode": "auto", "db_path": "/old/project.db", "theme": "dark"})
    monkeypatch.setattr(server, "_restore_from_settings", lambda: None)
    monkeypatch.setattr(server.model_store, "available_catalog", lambda: [{"id": "quran-phonetics"}])
    monkeypatch.setattr(server.model_store, "ACTIVE_ID", "quran-phonetics")
    server._initialize_app_settings()
    assert settings.load() == {"mode": "full", "db_path": "/old/project.db", "theme": "dark"}


def test_removed_auto_setting_uses_full_when_migration_cannot_be_saved(settings_file, monkeypatch):
    from surah_cutter.webgui import server
    from surah_cutter.webgui.state import AppState

    settings.save({"mode": "auto", "language": "ru"})
    before = settings_file.read_bytes()

    def fail_replace(*args):
        raise PermissionError("read-only settings")

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(server, "_restore_from_settings", lambda: None)
    monkeypatch.setattr(server, "STATE", AppState())
    monkeypatch.setattr(server.model_store, "available_catalog", lambda: [{"id": "quran-phonetics"}])
    monkeypatch.setattr(server.model_store, "ACTIVE_ID", "quran-phonetics")
    server._initialize_app_settings()
    get_project = next(r.endpoint for r in server.app.routes if r.path == "/api/project")
    assert get_project()["mode"] == "full"
    assert settings_file.read_bytes() == before


def run_concurrently(actions):
    barrier = threading.Barrier(len(actions))

    def run(action):
        barrier.wait(timeout=2)
        return action()

    with ThreadPoolExecutor(max_workers=len(actions)) as pool:
        return list(pool.map(run, actions))


def test_concurrent_settings_model_theme_language_and_project_preserved(settings_file, monkeypatch):
    settings.save({"arabic_size": 36})
    original_load = settings.load

    def delayed_read():
        data = original_load()
        # Force overlapping read/merge/save in the previous unlocked implementation.
        time.sleep(.02)
        return data

    monkeypatch.setattr(settings, "load", delayed_read)
    fields = {"theme": "dark", "language": "ru", "mode": "words",
              "alignment_model": "quran-phonetics", "db_path": "/tmp/project.db",
              "audio_dir": "/tmp/audio", "time_step": 50}
    run_concurrently([lambda k=k, v=v: settings.update(strict=True, **{k: v})
                      for k, v in fields.items()])
    assert original_load() == {"arabic_size": 36, **fields}


def test_concurrent_external_project_name_merges_and_ui_updates(settings_file, monkeypatch):
    original_load = settings.load

    def delayed_read():
        result = original_load()
        time.sleep(.01)
        return result

    monkeypatch.setattr(settings, "load", delayed_read)
    paths = [f"/tmp/project-{n}.db" for n in range(4)]
    run_concurrently([
        lambda path=path: settings.update_external_project(
            path, {"reciter": path, "recording": "Test"}, strict=True,
        ) for path in paths
    ] + [lambda: settings.update(strict=True, theme="dark", mode="full")])
    result = original_load()
    assert set(result["external_projects"]) == set(paths)
    assert (result["theme"], result["mode"]) == ("dark", "full")


def test_saves_use_unique_temporaries_and_replace_atomically(settings_file, monkeypatch):
    replaced = []
    original_replace = Path.replace

    def record_replace(path, target):
        replaced.append(path)
        assert path.parent == settings_file.parent
        assert path != settings_file
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", record_replace)
    run_concurrently([lambda n=n: settings.update(strict=True, **{f"field{n}": n})
                      for n in range(6)])
    assert len(set(replaced)) == 6
    assert len(settings.load()) == 6
    assert list(settings_file.parent.glob("*.tmp")) == []


def test_failed_replace_keeps_confirmed_settings_and_cleans_temporary(settings_file, monkeypatch):
    settings.save({"theme": "light"}, strict=True)
    before = settings_file.read_bytes()

    def fail_replace(path, target):
        raise PermissionError("read-only settings")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError):
        settings.update(strict=True, theme="dark")
    assert settings_file.read_bytes() == before
    assert list(settings_file.parent.glob("*.tmp")) == []


def test_player_preferences_survive_reload(settings_file, monkeypatch):
    from surah_cutter.webgui import server

    monkeypatch.setattr(server.settings, "system_language", lambda: "ru")
    route = next(r.endpoint for r in server.app.routes if r.path == "/api/settings")
    project = next(r.endpoint for r in server.app.routes if r.path == "/api/project")
    route({"playback_mode": "repeat", "playback_rate": 1.75})
    assert settings.load()["playback_rate"] == 1.75
    assert (project()["playback_mode"], project()["playback_rate"]) == ("repeat", 1.75)
    route({"theme": "dark"})
    assert project()["playback_rate"] == 1.75


@pytest.mark.parametrize("payload", [
    {"playback_mode": "unknown"}, {"playback_rate": 0}, {"playback_rate": 2.1},
    {"playback_rate": True}, {"playback_rate": "1"}, {"playback_rate": None},
    {"playback_rate": float("nan")}, {"playback_rate": float("inf")},
])
def test_invalid_player_preferences_not_saved(settings_file, payload):
    from fastapi import HTTPException
    from surah_cutter.webgui import server

    route = next(r.endpoint for r in server.app.routes if r.path == "/api/settings")
    with pytest.raises(HTTPException) as exc:
        route(payload)
    assert exc.value.status_code == 400
    assert settings.load() == {}


def test_settings_mode_waits_for_concurrent_model_selection(settings_file, monkeypatch):
    from fastapi import HTTPException
    from surah_cutter.webgui import server

    active_kind = ["phonetic"]
    monkeypatch.setattr(server.model_store, "ACTIVE_ID", "quran-phonetics")
    selection_saved = threading.Event()
    release_selection = threading.Event()
    request_started = threading.Event()
    monkeypatch.setattr(server.model_store, "active_kind", lambda: active_kind[0])
    route = next(r.endpoint for r in server.app.routes if r.path == "/api/settings")

    def select_model():
        # The real select route also saves fields before publishing ACTIVE_ID,
        # with _PROJECT_LOCK held throughout both operations.
        with server._PROJECT_LOCK:
            settings.update(strict=True, alignment_model="whisper-medium", mode="full")
            selection_saved.set()
            assert release_selection.wait(2)
            active_kind[0] = "arabic"
            server.model_store.ACTIVE_ID = "whisper-medium"

    def change_mode():
        request_started.set()
        return route({"mode": "fast"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        selection = pool.submit(select_model)
        assert selection_saved.wait(2)
        change = pool.submit(change_mode)
        assert request_started.wait(2)
        try:
            # The stale phonetic choice must not be saved while selection is held.
            time.sleep(.03)
            assert settings.load()["mode"] == "full"
        finally:
            release_selection.set()
        selection.result(timeout=2)
        with pytest.raises(HTTPException) as exc:
            change.result(timeout=2)
        assert exc.value.status_code == 400
    assert settings.load()["mode"] == "full"
