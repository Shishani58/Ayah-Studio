"""Local project folders. Imports copy audio and snapshot SQLite, never move sources."""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from surah_cutter.data.project_db import ProjectDB
from surah_cutter.webgui import settings
from surah_cutter.webgui.state import DB_CODE, DB_VERSION, discover_audio


def root() -> Path:
    return settings.SETTINGS_PATH.parent / "projects"


def get(project_id: str) -> dict[str, Any]:
    # IDs are directory names, never user-supplied paths.
    if len(project_id) != 32 or any(c not in "0123456789abcdef" for c in project_id):
        raise ValueError("Проект не найден. Выберите проект из списка.")
    folder = root() / project_id
    data = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or not isinstance(data.get("reciter"), str)
            or not isinstance(data.get("recording"), str)
            or not (folder / "timings.db").is_file()):
        raise ValueError("Проект повреждён. Импортируйте исходную разметку в новый проект.")
    return {**data, "id": project_id, "audio_dir": str(folder / "audio"),
            "db_path": str(folder / "timings.db"), "folder": str(folder)}


def list_projects() -> list[dict[str, Any]]:
    if not root().exists():
        return []
    result = []
    for folder in sorted(root().iterdir()):
        try:
            result.append(get(folder.name))
        except (OSError, ValueError, KeyError):
            continue
    return result


def create(reciter: str, recording: str, audio_dir: str, db_path: str = "") -> dict[str, Any]:
    reciter, recording = reciter.strip(), recording.strip()
    if not reciter or not recording or max(len(reciter), len(recording)) > 120:
        raise ValueError("Укажите чтеца и название записи (до 120 символов).")
    audio = discover_audio(audio_dir)
    if not audio:
        raise ValueError("В папке нет аудио. Выберите файлы с номерами сур: 001.mp3 … 114.mp3.")
    source = Path(db_path).resolve() if db_path else None
    if source and not source.is_file():
        raise ValueError("БД не найдена. Выберите существующую .db или оставьте поле пустым.")
    project_id = uuid.uuid4().hex
    folder = root() / project_id
    folder.mkdir(parents=True)
    try:
        (folder / "audio").mkdir()
        for surah, path in audio.items():
            shutil.copy2(path, folder / "audio" / f"{surah:03d}{path.suffix.lower()}")
        destination = folder / "timings.db"
        if source:
            with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src:
                src.execute("SELECT sura, ayah, time FROM timings LIMIT 1")
                with sqlite3.connect(destination) as dst:
                    src.backup(dst)
        with ProjectDB.open(str(destination), DB_CODE, DB_VERSION):
            pass
        (folder / "project.json").write_text(
            json.dumps({"reciter": reciter, "recording": recording}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(folder)
        raise
    return get(project_id)


def rename(project_id: str, reciter: str, recording: str) -> None:
    project = get(project_id)
    reciter, recording = reciter.strip(), recording.strip()
    if not reciter or not recording or max(len(reciter), len(recording)) > 120:
        raise ValueError("Укажите чтеца и название записи (до 120 символов).")
    path = Path(project["folder"]) / "project.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"reciter": reciter, "recording": recording},
                                    ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
