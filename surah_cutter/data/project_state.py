"""Shared audio discovery and canonical surah-completeness checks."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from surah_cutter.data.quran_text import surah_ayah_counts

AUDIO_EXTS = (".mp3", ".m4a", ".ogg", ".flac", ".wav")
_NUMBER = re.compile(r"^(\d{1,3})(?=$|\D)")


def surah_from_name(name: str) -> int | None:
    """Accept a leading 1..114 surah number, including named recordings."""
    match = _NUMBER.match(Path(name).stem)
    number = int(match.group(1)) if match else 0
    return number if 1 <= number <= 114 else None


def discover_audio(audio_dir: str | Path) -> dict[int, Path]:
    """Choose one recording per surah by extension priority, then filename."""
    folder = Path(audio_dir)
    if not folder.is_dir():
        return {}
    ranked: dict[int, Path] = {}
    priority = {extension: rank for rank, extension in enumerate(AUDIO_EXTS)}
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        number = surah_from_name(path.name)
        suffix = path.suffix.lower()
        if not path.is_file() or number is None or suffix not in priority:
            continue
        previous = ranked.get(number)
        if previous is None or (priority[suffix], path.name) < (
            priority[previous.suffix.lower()], previous.name
        ):
            ranked[number] = path
    return dict(sorted(ranked.items()))


def complete_suras(
    db_path: str | Path | None,
    counts: dict[int, int] | None = None,
) -> set[int]:
    """Suras with exactly the canonical ayah numbers in their timing rows."""
    if db_path is None or not Path(db_path).is_file():
        return set()
    with sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True) as conn:
        try:
            rows = conn.execute("SELECT sura, ayah FROM timings").fetchall()
        except sqlite3.OperationalError:
            return set()
    expected = counts if counts is not None else surah_ayah_counts()
    by_sura: dict[int, list[int]] = {}
    invalid_suras: set[int] = set()
    for sura, ayah in rows:
        if type(sura) is not int:
            try:
                candidate = int(sura)
            except (TypeError, ValueError, OverflowError):
                continue
            if candidate in expected:
                invalid_suras.add(candidate)
            continue
        if type(ayah) is not int:
            invalid_suras.add(sura)
            continue
        by_sura.setdefault(sura, []).append(ayah)
    return {sura for sura, count in expected.items()
            if sura not in invalid_suras and len(by_sura.get(sura, [])) == count
            and set(by_sura[sura]) == set(range(1, count + 1))}
