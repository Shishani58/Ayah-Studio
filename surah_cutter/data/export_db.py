"""Export a clean, distributable timings DB.

Copies ONLY ``timings`` + ``db_version`` from a project DB into a fresh file;
service tables (``review_meta``, ``edit_history``) are stripped. By default the
export is strict and refuses to produce an incomplete Quran: exactly 6236 ayahs
across 114 suras.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from surah_cutter.data.quran_text import surah_ayah_counts

# Canonical full-Quran totals.
EXPECTED_AYAHS = 6236
EXPECTED_SURAS = 114

_EXPORT_SCHEMA = """
CREATE TABLE timings (
    sura INTEGER,
    ayah INTEGER,
    time INTEGER
);
CREATE TABLE db_version (
    code TEXT PRIMARY KEY,
    version INTEGER NOT NULL
);
"""


class ExportError(RuntimeError):
    """Raised when an export would overwrite its source or fails validation."""


def export_db(src_path: str, dst_path: str, *, strict: bool = True) -> str:
    """Write a clean ``timings`` + ``db_version`` DB from ``src_path``.

    With ``strict=True`` (default) the source must contain exactly
    :data:`EXPECTED_AYAHS` canonical ayahs across :data:`EXPECTED_SURAS` suras,
    with nonnegative integer times increasing strictly within each sura.
    Otherwise :class:`ExportError` is raised and no file is written.
    ``strict=False`` permits partial/unvalidated exports.

    ``dst_path`` is atomically replaced on success, but must never refer to the
    source file (including symbolic and hard links). Returns ``dst_path``.
    """
    source = Path(src_path).resolve()
    destination = Path(dst_path)
    if source == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ExportError("refusing to export over the source project DB")

    src = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    try:
        src.execute("BEGIN")
        rows = list(
            src.execute("SELECT sura, ayah, time FROM timings ORDER BY sura, ayah")
        )
        versions = list(src.execute("SELECT code, version FROM db_version"))
    finally:
        src.close()

    if strict:
        n_rows = len(rows)
        n_ayahs = len({(sura, ayah) for sura, ayah, _ in rows})
        n_suras = len({sura for sura, _, _ in rows})
        if n_rows != n_ayahs:
            raise ExportError(
                f"refusing to export DB with duplicate ayahs: {n_rows} rows but "
                f"{n_ayahs} distinct (sura, ayah)"
            )
        if n_ayahs != EXPECTED_AYAHS or n_suras != EXPECTED_SURAS:
            raise ExportError(
                f"refusing to export incomplete DB: {n_ayahs} ayahs across "
                f"{n_suras} suras (expected {EXPECTED_AYAHS} across {EXPECTED_SURAS})"
            )
        expected = {
            (sura, ayah)
            for sura, count in surah_ayah_counts().items()
            for ayah in range(1, count + 1)
        }
        if any(type(sura) is not int or type(ayah) is not int for sura, ayah, _ in rows):
            raise ExportError("refusing to export non-integer sura/ayah numbers")
        if {(sura, ayah) for sura, ayah, _ in rows} != expected:
            raise ExportError("refusing to export DB with a non-canonical ayah set")
        previous_sura = None
        previous_time = -1
        for sura, ayah, time in rows:
            if type(time) is not int or time < 0:
                raise ExportError(f"invalid time at {sura}:{ayah}: expected nonnegative integer")
            if sura == previous_sura and time <= previous_time:
                raise ExportError(f"times must increase strictly within sura {sura}: ayah {ayah}")
            previous_sura, previous_time = sura, time

    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                     dir=destination.parent)
    os.close(fd)
    try:
        dst = sqlite3.connect(temporary)
        try:
            dst.executescript(_EXPORT_SCHEMA)
            dst.executemany(
                "INSERT INTO timings (sura, ayah, time) VALUES (?, ?, ?)", rows
            )
            dst.executemany(
                "INSERT INTO db_version (code, version) VALUES (?, ?)", versions
            )
            dst.commit()
        finally:
            dst.close()
        os.replace(temporary, dst_path)
    finally:
        Path(temporary).unlink(missing_ok=True)

    return dst_path
