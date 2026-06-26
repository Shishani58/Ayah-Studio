"""Persisted UI settings — remember the last project across restarts.

Stores a tiny JSON blob (last audio folder, last project DB, last mode) under the
user config dir so the app reopens where it left off. Best-effort: any IO error
is swallowed (settings are a convenience, never required for correctness).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: User-level settings file (cross-platform: under the home dir).
SETTINGS_PATH = Path.home() / ".surah-cutter" / "settings.json"


def load() -> dict[str, Any]:
    """Return the saved settings dict (empty on first run / any error)."""
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(data: dict[str, Any]) -> None:
    """Write ``data`` as the settings file (best-effort, never raises)."""
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def update(**fields: Any) -> dict[str, Any]:
    """Merge non-``None`` ``fields`` into the saved settings and persist."""
    data = load()
    data.update({k: v for k, v in fields.items() if v is not None})
    save(data)
    return data
