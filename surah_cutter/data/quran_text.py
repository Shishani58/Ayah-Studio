"""Read-only access to the bundled Quran text (``data/quran_mushaf.db``).

Pure stdlib (``sqlite3``). No PySide6/torch; safe to import from ``core``.

The DB stores one row per *word* in ``words(id, surah, ayah, qpc_hafs, ...)``
ordered by ``id`` in recitation order. Whitespace-only rows act as
ayah-end separators (exactly one per ayah, 6236 total) and are skipped.
``ayahs_search_text(surah, ayah, text)`` holds plain per-ayah text and is
used only as an optional cross-check.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Bundled DB: <repo>/data/quran_mushaf.db (this file is surah_cutter/data/quran_text.py).
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quran_mushaf.db"


@dataclass(frozen=True)
class Word:
    """A single recited word of the Quran.

    Attributes:
        surah: Surah number (1..114).
        ayah: Ayah number within the surah (1-based).
        text: Word text in QPC Hafs script (``qpc_hafs``).
        index: Position of this word within its surah (0-based, recitation order).
    """

    surah: int
    ayah: int
    text: str
    index: int


class QuranText:
    """Read-only reader over ``quran_mushaf.db`` with a reused connection."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Quran DB not found: {self._path}")
        # Open read-only via URI so the bundled DB is never mutated.
        uri = f"file:{self._path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> QuranText:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @lru_cache(maxsize=1)
    def surah_ayah_counts(self) -> dict[int, int]:
        """Map surah -> ayah count, derived from the DB.

        Returns 114 entries summing to 6236. Ayah numbering is contiguous
        per surah, so ``max(ayah)`` equals the count.
        """
        rows = self._conn.execute(
            "SELECT surah, MAX(ayah) FROM words GROUP BY surah ORDER BY surah"
        ).fetchall()
        return {int(s): int(n) for s, n in rows}

    def ayah_count(self, surah: int) -> int:
        """Number of ayahs in ``surah``."""
        return self.surah_ayah_counts()[surah]

    def surah_words(self, surah: int) -> list[Word]:
        """Words of ``surah`` in recitation order (by ``words.id``).

        Whitespace-only separator rows are skipped. The per-surah word
        index is assigned over the kept words. This is the word->ayah map
        consumed by forced alignment.
        """
        rows = self._conn.execute(
            "SELECT ayah, qpc_hafs FROM words "
            "WHERE surah = ? AND TRIM(qpc_hafs) != '' ORDER BY id",
            (surah,),
        ).fetchall()
        return [
            Word(surah=surah, ayah=int(ayah), text=text, index=i)
            for i, (ayah, text) in enumerate(rows)
        ]

    def ayah_text(self, surah: int, ayah: int) -> str:
        """QPC Hafs words of ``(surah, ayah)`` joined by a single space."""
        rows = self._conn.execute(
            "SELECT qpc_hafs FROM words "
            "WHERE surah = ? AND ayah = ? AND TRIM(qpc_hafs) != '' ORDER BY id",
            (surah, ayah),
        ).fetchall()
        return " ".join(r[0] for r in rows)

    def surah_text(self, surah: int) -> str:
        """Full QPC Hafs text of ``surah`` (words joined by single spaces)."""
        return " ".join(w.text for w in self.surah_words(surah))


# Module-level convenience API over a shared default reader.


@lru_cache(maxsize=4)
def _reader(db_path: str | Path = DEFAULT_DB_PATH) -> QuranText:
    return QuranText(db_path)


def surah_ayah_counts(db_path: str | Path = DEFAULT_DB_PATH) -> dict[int, int]:
    """See :meth:`QuranText.surah_ayah_counts`."""
    return _reader(db_path).surah_ayah_counts()


def ayah_count(surah: int, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """See :meth:`QuranText.ayah_count`."""
    return _reader(db_path).ayah_count(surah)


def ayah_text(surah: int, ayah: int, db_path: str | Path = DEFAULT_DB_PATH) -> str:
    """See :meth:`QuranText.ayah_text`."""
    return _reader(db_path).ayah_text(surah, ayah)


def surah_words(surah: int, db_path: str | Path = DEFAULT_DB_PATH) -> list[Word]:
    """See :meth:`QuranText.surah_words`."""
    return _reader(db_path).surah_words(surah)


def surah_text(surah: int, db_path: str | Path = DEFAULT_DB_PATH) -> str:
    """See :meth:`QuranText.surah_text`."""
    return _reader(db_path).surah_text(surah)
