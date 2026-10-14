"""Persisted UI settings — remember the last project across restarts.

Stores a tiny JSON blob (last audio folder, last project DB, last mode) under the
user config dir so the app reopens where it left off. Best-effort: any IO error
is swallowed (settings are a convenience, never required for correctness).
"""

from __future__ import annotations

import json
import locale
import os
import re
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

#: User-level settings file (cross-platform: under the home dir).
SETTINGS_PATH = Path.home() / ".surah-cutter" / "settings.json"
_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def system_language() -> str:
    """Read the OS display language, including packaged desktop applications."""
    language = ""
    try:
        if sys.platform == "darwin":
            # GUI bundles need the user's preferred language, not the shell locale.
            result = subprocess.run(
                ["/usr/bin/defaults", "read", "-g", "AppleLanguages"],
                capture_output=True, text=True, timeout=2, check=True,
            )
            match = re.search(r"[a-zA-Z]{2,3}(?:[-_][a-zA-Z0-9]+)*", result.stdout)
            language = match.group(0) if match else ""
        elif sys.platform == "win32":
            import ctypes

            buffer = ctypes.create_unicode_buffer(85)
            kernel = ctypes.windll.kernel32
            # Display language may differ from regional date/number formats.
            if kernel.LCIDToLocaleName(kernel.GetUserDefaultUILanguage(), buffer, len(buffer), 0):
                language = buffer.value
        else:
            language = next((os.environ[key] for key in ("LC_ALL", "LC_MESSAGES", "LANG")
                             if os.environ.get(key)), "")
    except (OSError, subprocess.SubprocessError, AttributeError):
        pass
    if not language:
        try:
            language = locale.getlocale()[0] or ""
        except ValueError:
            pass
    return "ru" if re.split(r"[-_.]", language.lower())[0] == "ru" else "en"


def language() -> str:
    """Initialize once from the OS; saved choices take precedence on every launch."""
    with _LOCK:
        saved = load().get("language")
        if saved in ("ru", "en"):
            return saved
        chosen = system_language()
        update(language=chosen)
        return chosen


def load() -> dict[str, Any]:
    """Return the saved settings dict (empty on first run / any error)."""
    with _LOCK:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def save(data: dict[str, Any], *, strict: bool = False) -> None:
    """Atomically write settings; ``strict`` surfaces errors to explicit UI saves."""
    with _LOCK:
        temporary = None
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=SETTINGS_PATH.parent,
                prefix=f".{SETTINGS_PATH.name}.", suffix=".tmp", delete=False,
            ) as output:
                temporary = Path(output.name)
                json.dump(data, output, ensure_ascii=False, indent=2)
            temporary.replace(SETTINGS_PATH)
        except Exception:
            if strict:
                raise
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def update(*, strict: bool = False, **fields: Any) -> dict[str, Any]:
    """Merge non-``None`` ``fields`` into the saved settings and persist."""
    with _LOCK:
        data = load()
        data.update({k: v for k, v in fields.items() if v is not None})
        save(data, strict=strict)
        return data


def update_external_project(path: str, names: dict[str, str], *, strict: bool = False,
                            **fields: Any) -> dict[str, Any]:
    """Merge one external project's names in the same transaction as UI fields."""
    with _LOCK:
        external = dict(load().get("external_projects", {}))
        external[path] = names
        return update(strict=strict, external_projects=external, **fields)
