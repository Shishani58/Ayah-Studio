"""Working project DB (SQLite).

Doubles as a valid output DB (``timings`` + ``db_version``) plus service
tables used during alignment review: ``review_meta`` and ``edit_history``.

Pure-Python ``sqlite3`` only — no GUI/heavy deps. Every mutating call commits
immediately (autosave).

By default regeneration must not clobber manual work.
``write_timings`` overwrites an ayah only when its
``review_meta.status`` is ``auto`` or absent; ``edited``/``verified`` rows are
left untouched. An explicit GUI overwrite choice permits replacement with
old/new timings recorded in the existing edit history.
"""

from __future__ import annotations

import sqlite3
import math
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone

from surah_cutter.data.quran_text import surah_ayah_counts

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
        """Open (or create) the DB and initialize missing edition metadata.

        ``code``/``version`` identify a new reciter/edition. Opening an existing
        project must preserve its edition, including its original version.
        """
        conn = sqlite3.connect(path)
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO db_version (code, version) SELECT ?, ? "
            "WHERE NOT EXISTS (SELECT 1 FROM db_version)",
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

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        """Lock the writer before reading statuses/neighbours; roll back on failure."""
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            yield

    @staticmethod
    def _validate_key(sura: int, ayah: int) -> None:
        counts = surah_ayah_counts()
        if (type(sura) is not int or type(ayah) is not int
                or sura not in counts or not 1 <= ayah <= counts[sura]):
            raise ValueError("Укажите существующие целые номера суры и аята.")

    @staticmethod
    def _validate_time(time: int) -> None:
        if type(time) is not int or not 0 <= time <= 2**63 - 1:
            raise ValueError("Начало аята должно быть неотрицательным целым числом миллисекунд.")

    def _validate_order(self, sura: int) -> None:
        previous = None
        for ayah, time in self._conn.execute(
            "SELECT ayah, time FROM timings WHERE sura=? ORDER BY ayah", (sura,)
        ):
            self._validate_key(sura, ayah)
            self._validate_time(time)
            if previous is not None and time <= previous:
                raise ValueError(
                    "Начала аятов должны идти строго по порядку; проверьте ручные метки."
                )
            previous = time

    # -- writes ------------------------------------------------------------

    def write_timings(
        self,
        rows: Iterable[Sequence[int]],
        source: str,
        confidence: float | None = None,
        force: bool = False,
        *,
        confidences: dict[int, float | None] | None = None,
        overwrite_manual: bool = False,
        duration_ms: float | None = None,
        persisted: list[dict] | None = None,
        replace_fast_sura: int | None = None,
    ) -> int:
        """Merge auto-generated timings.

        ``rows`` is an iterable of ``(sura, ayah, time_ms)``. For each row the
        ayah is written only if its current ``review_meta.status`` is ``auto``
        or absent; ``edited``/``verified`` ayahs are skipped (left untouched).
        Written ayahs get/keep ``review_meta.status = 'auto'``.

        ``overwrite_manual=True`` requires an explicit user choice and records
        replaced human timings in edit_history. ``force=True`` stays rejected.
        The complete merged order is checked under the same writer lock; any
        conflict rolls back all timings and metadata from this call.

        Returns the number of ayahs actually written.
        """
        if force:
            raise ValueError("Принудительная перезапись ручных меток запрещена.")
        if duration_ms is not None and (
            isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float))
            or not math.isfinite(duration_ms) or duration_ms <= 0
        ):
            raise ValueError("Длительность аудио должна быть конечной и положительной.")
        from surah_cutter.core.cancellation import check_cancelled

        check_cancelled()
        rows = list(rows)  # Consume generators before acquiring the short DB lock.
        keys = set()
        for sura, ayah, time in rows:
            self._validate_key(sura, ayah)
            self._validate_time(time)
            if (sura, ayah) in keys:
                raise ValueError("Повторяющийся аят в результате разметки.")
            keys.add((sura, ayah))
        if replace_fast_sura is not None:
            self._validate_key(replace_fast_sura, 1)
            if any(s != replace_fast_sura for s, _ in keys):
                raise ValueError("FAST replacement must contain one surah")
        written = 0
        now = _now()
        snapshot = []
        with self._write_transaction():
            check_cancelled()
            if replace_fast_sura is not None:
                # A new partial FAST supersedes only previous machine FAST
                # proposals. FULL and human work have independent provenance.
                from surah_cutter.data.alignment_identity import is_fast_source
                stale = self._conn.execute(
                    "SELECT t.ayah, r.source FROM timings t JOIN review_meta r "
                    "ON r.sura=t.sura AND r.ayah=t.ayah "
                    "WHERE t.sura=? AND r.status='auto'", (replace_fast_sura,),
                ).fetchall()
                for ayah, old_source in stale:
                    if (replace_fast_sura, ayah) not in keys and is_fast_source(old_source):
                        self._conn.execute("DELETE FROM timings WHERE sura=? AND ayah=?",
                                           (replace_fast_sura, ayah))
                        self._conn.execute("DELETE FROM review_meta WHERE sura=? AND ayah=?",
                                           (replace_fast_sura, ayah))
            for sura, ayah, time in rows:
                status = self._status(sura, ayah)
                if status in ("edited", "verified"):
                    if not overwrite_manual:
                        continue
                    old = self._conn.execute(
                        "SELECT time FROM timings WHERE sura=? AND ayah=?", (sura, ayah)
                    ).fetchone()
                    self._conn.execute(
                        "INSERT INTO edit_history (sura, ayah, old_time, new_time, ts, action) "
                        "VALUES (?, ?, ?, ?, ?, 'generation_overwrite')",
                        (sura, ayah, old[0] if old else None, time, now),
                    )
                self._set_timing(sura, ayah, time)
                self._conn.execute(
                    "INSERT INTO review_meta "
                    "(sura, ayah, end_ms, confidence, status, source, updated_at) "
                    "VALUES (?, ?, NULL, ?, 'auto', ?, ?) "
                    "ON CONFLICT(sura, ayah) DO UPDATE SET "
                    "confidence = excluded.confidence, status = 'auto', "
                    "source = excluded.source, updated_at = excluded.updated_at",
                    (sura, ayah, confidences.get(ayah, confidence)
                     if confidences is not None else confidence, source, now),
                )
                written += 1
            for sura in ({s for s, _ in keys} | ({replace_fast_sura} if replace_fast_sura is not None else set())):
                self._validate_order(sura)
                for ayah, time in self._conn.execute(
                    "SELECT ayah, time FROM timings WHERE sura=? ORDER BY ayah", (sura,)
                ):
                    if duration_ms is not None and time >= duration_ms:
                        raise ValueError("Начало аята должно быть внутри длительности аудио; "
                                         "проверьте ручные метки.")
                    if persisted is not None:
                        snapshot.append({"sura": sura, "ayah": ayah, "time": time,
                                         **(self.get_review(sura, ayah) or {})})
            # The last cancellation gate is inside the transaction, immediately
            # before SQLite commits. A request arriving later loses to commit.
            check_cancelled()
        if persisted is not None:
            persisted[:] = snapshot
        return written

    def update_auto_confidences(
        self,
        sura: int,
        confidences: dict[int, float | None],
    ) -> None:
        """Update per-ayah confidence for machine-generated rows only."""
        with self._write_transaction():
            for ayah, confidence in confidences.items():
                self._conn.execute(
                    "UPDATE review_meta SET confidence = ? "
                    "WHERE sura = ? AND ayah = ? AND status = 'auto'",
                    (confidence, sura, ayah),
                )

    def set_time(self, sura: int, ayah: int, time: int) -> None:
        """Manually set an ayah's start time.

        Records the change in ``edit_history`` and marks the ayah ``edited``
        so future regeneration leaves it alone.
        """
        self._validate_key(sura, ayah)
        self._validate_time(time)
        with self._write_transaction():
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
            self._validate_order(sura)

    def set_status(self, sura: int, ayah: int, status: str) -> None:
        """Set the review status for an ayah (one of :data:`STATUSES`)."""
        if status not in STATUSES:
            raise ValueError(f"invalid status {status!r}; expected one of {STATUSES}")
        self._validate_key(sura, ayah)
        with self._write_transaction():
            if status != "auto" and self._conn.execute(
                "SELECT 1 FROM timings WHERE sura=? AND ayah=?", (sura, ayah)
            ).fetchone() is None:
                raise ValueError("Сначала укажите начало аята.")
            now = _now()
            self._conn.execute(
                "INSERT INTO review_meta "
                "(sura, ayah, end_ms, confidence, status, source, updated_at) "
                "VALUES (?, ?, NULL, NULL, ?, NULL, ?) "
                "ON CONFLICT(sura, ayah) DO UPDATE SET "
                "status = excluded.status, updated_at = excluded.updated_at",
                (sura, ayah, status, now),
            )

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
                           source: str, *, overwrite_manual: bool = False) -> int:
        """Merge words; replacing human marks requires an explicit user choice."""
        rows = list(rows)
        with self._write_transaction():
            existing = {r["word_index"]: r for r in self.get_word_timings(sura, ayah)}
            merged = []
            for index, time, confidence in rows:
                old = existing.get(index)
                if old and old["status"] in ("edited", "verified") and not overwrite_manual:
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
            for index, time, confidence, writable in merged:
                if not writable:
                    continue
                old = existing.get(index)
                if old and old["status"] in ("edited", "verified"):
                    self._conn.execute(
                        "INSERT INTO word_edit_history "
                        "(sura, ayah, word_index, old_time, new_time, ts, action) "
                        "VALUES (?, ?, ?, ?, ?, ?, 'generation_overwrite')",
                        (sura, ayah, index, old["time"], time, _now()),
                    )
                self._conn.execute(
                    "INSERT INTO word_timings VALUES (?, ?, ?, ?, ?, 'auto', ?, ?, ?, ?) "
                    "ON CONFLICT(sura, ayah, word_index) DO UPDATE SET "
                    "time=excluded.time, confidence=excluded.confidence, status='auto', source=excluded.source, "
                    "ayah_start=excluded.ayah_start, ayah_end=excluded.ayah_end, "
                    "updated_at=excluded.updated_at",
                    (sura, ayah, index, time, confidence, source, start, end, _now()),
                )
                written += 1
        return written

    def set_word_time(self, sura: int, ayah: int, index: int, time: int,
                      start: int, end: int) -> None:
        """Manual word edit with neighbour guards and a separate audit trail."""
        with self._write_transaction():
            rows = self.get_word_timings(sura, ayah)
            old = next((r for r in rows if r["word_index"] == index), None)
            before = [r["time"] for r in rows if r["word_index"] < index]
            after = [r["time"] for r in rows if r["word_index"] > index]
            if (type(time) is not int or not start <= time < end
                    or (before and time <= max(before)) or (after and time >= min(after))):
                raise ValueError("Начало слова должно быть внутри аята между соседними словами.")
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
        self.set_word_status(sura, ayah, index, "verified", start, end)

    def set_word_status(self, sura: int, ayah: int, index: int, status: str,
                        start: int, end: int) -> None:
        """Change review state without changing time or accepting stale boundaries."""
        if status not in STATUSES:
            raise ValueError("Неизвестный статус слова.")
        with self._write_transaction():
            cursor = self._conn.execute(
                "UPDATE word_timings SET status=?, updated_at=? "
                "WHERE sura=? AND ayah=? AND word_index=? AND ayah_start=? AND ayah_end=?",
                (status, _now(), sura, ayah, index, start, end),
            )
            if not cursor.rowcount:
                raise ValueError("Сначала поставьте или обновите метку слова.")
