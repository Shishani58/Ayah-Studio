"""Resolve bundled read-only assets in a checkout, frozen app or installed wheel."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Return the path to a required small asset without depending on cwd."""
    relative = Path(*parts)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Resource path must be relative to the bundle")
    package = Path(__file__).resolve().parent
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / relative
    checkout = package.parent / relative
    if (package.parent / "pyproject.toml").is_file() and checkout.exists():
        return checkout
    return package / "bundled" / relative


def external_model_path(name: str) -> Path:
    """Find an optional large model outside a wheel, respecting the override."""
    if Path(name).name != name:
        raise ValueError("Model name must be a filename")
    override = os.environ.get("SURAH_CUTTER_MODELS_DIR")
    if override:
        return Path(override).expanduser() / name
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "models" / name
    root = Path(__file__).resolve().parent.parent
    checkout = root / "models" / name
    if (root / "pyproject.toml").is_file() and checkout.is_file():
        return checkout
    return Path.home() / ".surah-cutter" / "models" / name
