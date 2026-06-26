"""Export a clean, distributable timings DB.

Copies ONLY ``timings`` + ``db_version`` from a project DB into a fresh file;
service tables (``review_meta``, ``edit_history``) are stripped. By default the
export is strict and refuses to produce an incomplete Quran: exactly 6236 ayahs
across 114 suras.
"""

from __future__ import annotations

import os
import sqlite3

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
    """Raised when the source DB fails the completeness guard."""


def export_db(src_path: str, dst_path: str, *, strict: bool = True) -> str:
    """Write a clean ``timings`` + ``db_version`` DB from ``src_path``.

    With ``strict=True`` (default) the source must contain exactly
    :data:`EXPECTED_AYAHS` timing rows spanning :data:`EXPECTED_SURAS` distinct
    suras, otherwise :class:`ExportError` is raised and no file is written.

    ``dst_path`` is overwritten if it already exists. Returns ``dst_path``.
    """
    src = sqlite3.connect(src_path)
    try:
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

    if os.path.exists(dst_path):
        os.remove(dst_path)

    dst = sqlite3.connect(dst_path)
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

    return dst_path
