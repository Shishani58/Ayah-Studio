"""Working project DB (SQLite).

Doubles as a valid output DB (``timings`` + ``db_version``) plus service
tables used during alignment review: ``review_meta`` and ``edit_history``.

Pure-Python ``sqlite3`` only — no GUI/heavy deps. Every mutating call commits
immediately (autosave).

Status invariant (CLAUDE.md non-negotiable): regeneration must not clobber
manual work. ``write_timings`` overwrites an ayah only when its
``review_meta.status`` is ``auto`` or absent; ``edited``/``verified`` rows are
left untouched.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

# Allowed review statuses. ``auto`` = machine-generated (overwritable);
# ``edited`` = manually changed; ``verified`` = human-confirmed.
STATUSES = ("auto", "edited", "verified")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS timings (
    sura INTEGER,
    ayah INTEGER,
    time INTEGER
);
CREATE TABLE IF NOT EXISTS db_version (
    code TEXT PRIMARY KEY,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS review_meta (
    sura INTEGER,
    ayah INTEGER,
    end_ms INTEGER,
    confidence REAL,
    status TEXT,
    source TEXT,
    updated_at TEXT,
    PRIMARY KEY (sura, ayah)
);
CREATE TABLE IF NOT EXISTS edit_history (
    id INTEGER PRIMARY KEY,
    sura INTEGER,
    ayah INTEGER,
    old_time INTEGER,
    new_time INTEGER,
    ts TEXT,
    action TEXT
);
CREATE TABLE IF NOT EXISTS word_timings (
    sura INTEGER NOT NULL, ayah INTEGER NOT NULL, word_index INTEGER NOT NULL,
    time INTEGER NOT NULL, confidence REAL, status TEXT NOT NULL,
    source TEXT NOT NULL, ayah_start INTEGER NOT NULL, ayah_end INTEGER NOT NULL,
    updated_at TEXT NOT NULL, PRIMARY KEY (sura, ayah, word_index)
);
CREATE TABLE IF NOT EXISTS word_edit_history (
    id INTEGER PRIMARY KEY, sura INTEGER, ayah INTEGER, word_index INTEGER,
    old_time INTEGER, new_time INTEGER, ts TEXT, action TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_timings_sura_ayah ON timings (sura, ayah);
"""


def _now() -> str:
    """UTC timestamp in ISO-8601 (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectDB:
    """Working DB handle. Use :meth:`open` to construct.

    All mutating methods commit before returning.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: str, code: str, version: int) -> "ProjectDB":
        """Open (or create) the DB at ``path`` and ensure schema + db_version.

        ``code``/``version`` identify the reciter/edition; the row is upserted
        so an existing project keeps a single ``db_version`` entry in sync.
        """
        conn = sqlite3.connect(path)
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO db_version (code, version) VALUES (?, ?) "
            "ON CONFLICT(code) DO UPDATE SET version = excluded.version",
            (code, version),
        )
        conn.commit()
        return cls(conn)

    # Backwards-friendly alias.
    create = open

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> "ProjectDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internal helpers --------------------------------------------------

    def _status(self, sura: int, ayah: int) -> str | None:
        """Return review status for an ayah, or ``None`` if no meta row."""
        row = self._conn.execute(
            "SELECT status FROM review_meta WHERE sura = ? AND ayah = ?",
            (sura, ayah),
        ).fetchone()
        return row[0] if row else None

    def _set_timing(self, sura: int, ayah: int, time: int) -> None:
        """Upsert a single ``timings`` row (no commit)."""
        self._conn.execute(
            "DELETE FROM timings WHERE sura = ? AND ayah = ?", (sura, ayah)
        )
        self._conn.execute(
            "INSERT INTO timings (sura, ayah, time) VALUES (?, ?, ?)",
            (sura, ayah, time),
        )

    # -- writes ------------------------------------------------------------

    def write_timings(
        self,
        rows: Iterable[Sequence[int]],
        source: str,
        confidence: float | None = None,
        force: bool = False,
    ) -> int:
        """Merge auto-generated timings.

        ``rows`` is an iterable of ``(sura, ayah, time_ms)``. For each row the
        ayah is written only if its current ``review_meta.status`` is ``auto``
        or absent; ``edited``/``verified`` ayahs are skipped (left untouched).
        Written ayahs get/keep ``review_meta.status = 'auto'``.

        With ``force=True`` the skip is bypassed: every row is written and reset
        to ``auto`` (used by an explicit per-sura "recompute from scratch" that
        the user opted into, discarding prior manual edits for those ayahs).

        Returns the number of ayahs actually written.
        """
        written = 0
        now = _now()
        for sura, ayah, time in rows:
            status = self._status(sura, ayah)
            if not force and status in ("edited", "verified"):
                continue
            self._set_timing(sura, ayah, time)
            self._conn.execute(
                "INSERT INTO review_meta "
                "(sura, ayah, end_ms, confidence, status, source, updated_at) "
                "VALUES (?, ?, NULL, ?, 'auto', ?, ?) "
                "ON CONFLICT(sura, ayah) DO UPDATE SET "
                "confidence = excluded.confidence, status = 'auto', "
                "source = excluded.source, updated_at = excluded.updated_at",
                (sura, ayah, confidence, source, now),
            )
            written += 1
        self._conn.commit()
        return written

    def update_auto_confidences(
        self,
        sura: int,
        confidences: dict[int, float | None],
    ) -> None:
        """Update per-ayah confidence for machine-generated rows only."""
        for ayah, confidence in confidences.items():
            self._conn.execute(
                "UPDATE review_meta SET confidence = ? "
                "WHERE sura = ? AND ayah = ? AND status = 'auto'",
                (confidence, sura, ayah),
            )
        self._conn.commit()

    def set_time(self, sura: int, ayah: int, time: int) -> None:
        """Manually set an ayah's start time.

        Records the change in ``edit_history`` and marks the ayah ``edited``
        so future regeneration leaves it alone.
        """
        old = self._conn.execute(
            "SELECT time FROM timings WHERE sura = ? AND ayah = ?", (sura, ayah)
        ).fetchone()
        old_time = old[0] if old else None
        now = _now()
        self._set_timing(sura, ayah, time)
        self._conn.execute(
            "INSERT INTO edit_history "
            "(sura, ayah, old_time, new_time, ts, action) "
            "VALUES (?, ?, ?, ?, ?, 'set_time')",
            (sura, ayah, old_time, time, now),
        )
        self._conn.execute(
            "INSERT INTO review_meta "
            "(sura, ayah, end_ms, confidence, status, source, updated_at) "
            "VALUES (?, ?, NULL, NULL, 'edited', 'manual', ?) "
            "ON CONFLICT(sura, ayah) DO UPDATE SET "
            "status = 'edited', source = 'manual', updated_at = excluded.updated_at",
            (sura, ayah, now),
        )
        self._conn.commit()

    def set_status(self, sura: int, ayah: int, status: str) -> None:
        """Set the review status for an ayah (one of :data:`STATUSES`)."""
        if status not in STATUSES:
            raise ValueError(f"invalid status {status!r}; expected one of {STATUSES}")
        now = _now()
        self._conn.execute(
            "INSERT INTO review_meta "
            "(sura, ayah, end_ms, confidence, status, source, updated_at) "
            "VALUES (?, ?, NULL, NULL, ?, NULL, ?) "
            "ON CONFLICT(sura, ayah) DO UPDATE SET "
            "status = excluded.status, updated_at = excluded.updated_at",
            (sura, ayah, status, now),
        )
        self._conn.commit()

    # -- reads -------------------------------------------------------------

    def get_timings(self) -> list[tuple[int, int, int]]:
        """All ``(sura, ayah, time)`` rows ordered by (sura, ayah)."""
        return list(
            self._conn.execute(
                "SELECT sura, ayah, time FROM timings ORDER BY sura, ayah"
            )
        )

    def get_review(self, sura: int, ayah: int) -> dict[str, object] | None:
        """Review-meta dict for an ayah, or ``None`` if no row exists."""
        row = self._conn.execute(
            "SELECT sura, ayah, end_ms, confidence, status, source, updated_at "
            "FROM review_meta WHERE sura = ? AND ayah = ?",
            (sura, ayah),
        ).fetchone()
        if row is None:
            return None
        keys = ("sura", "ayah", "end_ms", "confidence", "status", "source", "updated_at")
        return dict(zip(keys, row))

    def get_word_timings(self, sura: int, ayah: int) -> list[dict]:
        """Saved words in canonical order, including boundary snapshots."""
        cursor = self._conn.execute(
            "SELECT word_index, time, confidence, status, source, ayah_start, ayah_end "
            "FROM word_timings WHERE sura=? AND ayah=? ORDER BY word_index", (sura, ayah)
        )
        keys = [col[0] for col in cursor.description]
        return [dict(zip(keys, row)) for row in cursor]

    def write_word_timings(self, sura: int, ayah: int, rows, start: int, end: int,
                           source: str) -> int:
        """Atomically merge auto words; preserve every manual/verified word."""
        existing = {r["word_index"]: r for r in self.get_word_timings(sura, ayah)}
        merged = []
        for index, time, confidence in rows:
            old = existing.get(index)
            if old and old["status"] in ("edited", "verified"):
                if (old["ayah_start"], old["ayah_end"]) != (start, end):
                    raise ValueError("Границы аята изменились. Сначала обновите ручные метки слов.")
                merged.append((index, old["time"], old["confidence"], False))
            else:
                merged.append((index, time, confidence, True))
        times = [r[1] for r in merged]
        if (not times or any(type(t) is not int or not start <= t < end for t in times)
                or any(a >= b for a, b in zip(times, times[1:]))):
            raise ValueError("Метки слов должны идти по порядку внутри аята; проверьте ручные метки.")
        written = 0
        with self._conn:
            for index, time, confidence, writable in merged:
                if not writable:
                    continue
                self._conn.execute(
                    "INSERT INTO word_timings VALUES (?, ?, ?, ?, ?, 'auto', ?, ?, ?, ?) "
                    "ON CONFLICT(sura, ayah, word_index) DO UPDATE SET "
                    "time=excluded.time, confidence=excluded.confidence, source=excluded.source, "
                    "ayah_start=excluded.ayah_start, ayah_end=excluded.ayah_end, "
                    "updated_at=excluded.updated_at",
                    (sura, ayah, index, time, confidence, source, start, end, _now()),
                )
                written += 1
        return written

    def set_word_time(self, sura: int, ayah: int, index: int, time: int,
                      start: int, end: int) -> None:
        """Manual word edit with neighbour guards and a separate audit trail."""
        rows = self.get_word_timings(sura, ayah)
        old = next((r for r in rows if r["word_index"] == index), None)
        before = [r["time"] for r in rows if r["word_index"] < index]
        after = [r["time"] for r in rows if r["word_index"] > index]
        if (type(time) is not int or not start <= time < end
                or (before and time <= max(before)) or (after and time >= min(after))):
            raise ValueError("Начало слова должно быть внутри аята между соседними словами.")
        with self._conn:
            self._conn.execute(
                "INSERT INTO word_timings VALUES (?, ?, ?, ?, NULL, 'edited', 'manual', ?, ?, ?) "
                "ON CONFLICT(sura, ayah, word_index) DO UPDATE SET "
                "time=excluded.time, confidence=NULL, status='edited', source='manual', "
                "ayah_start=excluded.ayah_start, ayah_end=excluded.ayah_end, "
                "updated_at=excluded.updated_at", (sura, ayah, index, time, start, end, _now()),
            )
            self._conn.execute(
                "INSERT INTO word_edit_history "
                "(sura, ayah, word_index, old_time, new_time, ts, action) "
                "VALUES (?, ?, ?, ?, ?, ?, 'set_time')",
                (sura, ayah, index, old["time"] if old else None, time, _now()),
            )

    def verify_word(self, sura: int, ayah: int, index: int, start: int, end: int) -> None:
        """Verify a current saved mark; an old boundary snapshot cannot be verified."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE word_timings SET status='verified', updated_at=? "
                "WHERE sura=? AND ayah=? AND word_index=? AND ayah_start=? AND ayah_end=?",
                (_now(), sura, ayah, index, start, end),
            )
            if not cursor.rowcount:
                raise ValueError("Сначала поставьте или обновите метку слова.")
