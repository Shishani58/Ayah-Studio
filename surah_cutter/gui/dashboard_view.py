"""Phase-1 dashboard: surah grid + audio→timings generation.

A scrollable grid of all 114 suras (number + ayah count + a status badge
derived from an open :class:`~surah_cutter.data.project_db.ProjectDB`).
Controls pick an audio folder and a project DB, then run generation off the
UI thread (a :class:`GenerationWorker` in a ``QThread``) so the heavy pipeline
never blocks the GUI.

Heavy deps (decode/align via ``core.pipeline``) are imported lazily inside the
worker; this module imports only PySide6 + pure-Python ``data`` helpers.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# Project DB identity (reciter/edition). Phase-1 placeholder; review screen
# rewrites it via reference metadata in later phases.
_DB_CODE = "surah-cutter"
_DB_VERSION = 1

# Audio extensions accepted in the chosen folder (one file = one sura).
_AUDIO_EXTS = (".mp3", ".m4a", ".ogg", ".flac", ".wav")

# Surah-number prefix: leading digits of the filename stem (001 / 01 / 1).
_NUM_RE = re.compile(r"^(\d{1,3})")

# Status -> (label, background colour). Pure presentation.
_STATUS_STYLE = {
    "not-processed": ("—", "#3a3a3a"),
    "processed": ("auto", "#7a5a1a"),
    "verified": ("✓", "#1f5f2f"),
}


def surah_from_filename(name: str) -> int | None:
    """Derive a 1..114 surah number from a filename, or ``None``.

    Matches the leading digit run of the stem (``003.mp3``/``03_x.m4a``/``3`` →
    3). Returns ``None`` when there is no leading number or it is out of range.
    """
    m = _NUM_RE.match(Path(name).stem)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 114 else None


def _surah_status(reviews: dict[int, list[str]], ayah_count: int, surah: int) -> str:
    """Classify a surah from its per-ayah review statuses.

    ``reviews`` maps surah -> list of ``review_meta.status`` strings present
    for that surah. ``verified`` only when every ayah is verified; ``processed``
    when any timing exists; else ``not-processed``.
    """
    statuses = reviews.get(surah)
    if not statuses:
        return "not-processed"
    if len(statuses) == ayah_count and all(s == "verified" for s in statuses):
        return "verified"
    return "processed"


class GenerationWorker(QObject):
    """Runs ``process_surah`` over an audio folder, writing timings to the DB.

    Lives in a worker ``QThread``. Heavy imports happen inside :meth:`run`.
    Resumable in spirit: suras already present in ``done_suras`` are skipped,
    and ``write_timings`` MERGEs (edited/verified ayahs are never clobbered).
    """

    progress = Signal(int, int, str)  # done, total, message
    surah_done = Signal(int)  # surah number just written
    finished = Signal(str)  # final summary message
    error = Signal(str)

    def __init__(
        self,
        audio_dir: str,
        db_path: str,
        done_suras: set[int],
        reciter: str | None = None,
    ) -> None:
        super().__init__()
        self._audio_dir = audio_dir
        self._db_path = db_path
        self._done = set(done_suras)
        self._reciter = reciter
        self._cancel = False

    def cancel(self) -> None:
        """Request cancellation; honoured between suras."""
        self._cancel = True

    def run(self) -> None:
        """Process each audio file in the folder; emit progress as we go."""
        try:
            self._run()
        except Exception as exc:  # surface to UI rather than crash the thread
            self.error.emit(str(exc))

    def _run(self) -> None:
        # Lazy: opening the DB is light, but keep all real work out of import.
        from surah_cutter.core.pipeline import process_surah
        from surah_cutter.data.project_db import ProjectDB

        folder = Path(self._audio_dir)
        files: list[tuple[int, Path]] = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in _AUDIO_EXTS:
                continue
            surah = surah_from_filename(path.name)
            if surah is None:
                continue
            files.append((surah, path))

        total = len(files)
        if total == 0:
            self.finished.emit("No NNN.{mp3,m4a,ogg,flac,wav} files found.")
            return

        written = 0
        skipped = 0
        with ProjectDB.open(self._db_path, _DB_CODE, _DB_VERSION) as db:
            for i, (surah, path) in enumerate(files):
                if self._cancel:
                    self.finished.emit(f"Cancelled after {written} surah(s).")
                    return
                if surah in self._done:
                    skipped += 1
                    self.progress.emit(i + 1, total, f"Skip surah {surah} (done)")
                    continue
                self.progress.emit(i + 1, total, f"Aligning surah {surah}…")
                rows = process_surah(str(path), surah, reciter=self._reciter)
                db.write_timings(rows, source="auto")
                written += 1
                self.surah_done.emit(surah)
                self.progress.emit(i + 1, total, f"Wrote surah {surah}")

        self.finished.emit(f"Done: {written} written, {skipped} skipped.")


class _SurahCell(QFrame):
    """One surah tile: number, ayah count, status badge.

    Double-clicking (or the "Review" button) emits :data:`review_requested`
    so the dashboard can open the review screen for that surah.
    """

    review_requested = Signal(int)  # surah number

    def __init__(self, surah: int, ayahs: int) -> None:
        super().__init__()
        self.surah = surah
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        self._title = QLabel(f"Surah {surah}")
        self._count = QLabel(f"{ayahs} ayahs")
        self._badge = QLabel()
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._review_btn = QPushButton("Review")
        self._review_btn.clicked.connect(
            lambda: self.review_requested.emit(self.surah)
        )
        layout.addWidget(self._title)
        layout.addWidget(self._count)
        layout.addWidget(self._badge)
        layout.addWidget(self._review_btn)
        self.set_status("not-processed")

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.review_requested.emit(self.surah)
        super().mouseDoubleClickEvent(event)

    def set_status(self, status: str) -> None:
        """Apply a status colour/badge to the cell."""
        label, colour = _STATUS_STYLE[status]
        self._badge.setText(f"{label}  {status}")
        self._badge.setStyleSheet(
            f"background:{colour}; color:#eee; border-radius:4px; padding:2px;"
        )


class DashboardView(QWidget):
    """Surah dashboard with audio-folder/DB pickers and a Generate action.

    Emits :data:`review_requested` (db_path, surah, audio_path) when a surah
    cell asks for review; the host (``app.main``) swaps in the review screen.
    """

    review_requested = Signal(str, int, str)  # db_path, surah, audio_path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("surah-cutter — dashboard")

        self._audio_dir: str | None = None
        self._db_path: str | None = None
        self._thread: QThread | None = None
        self._worker: GenerationWorker | None = None

        from surah_cutter.data.quran_text import surah_ayah_counts

        self._counts = surah_ayah_counts()
        self._cells: dict[int, _SurahCell] = {}

        self._build_ui()
        self._refresh_buttons()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        self._audio_btn = QPushButton("Choose audio folder…")
        self._db_btn = QPushButton("Open project DB…")
        self._gen_btn = QPushButton("Generate")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._audio_btn.clicked.connect(self._choose_audio)
        self._db_btn.clicked.connect(self._choose_db)
        self._gen_btn.clicked.connect(self._start_generation)
        self._cancel_btn.clicked.connect(self._cancel_generation)
        controls.addWidget(self._audio_btn)
        controls.addWidget(self._db_btn)
        controls.addWidget(self._gen_btn)
        controls.addWidget(self._cancel_btn)
        controls.addStretch(1)
        root.addLayout(controls)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        self._status_label = QLabel("Pick an audio folder and a project DB.")
        root.addWidget(self._status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        cols = 5
        for surah in range(1, 115):
            cell = _SurahCell(surah, self._counts.get(surah, 0))
            cell.review_requested.connect(self._request_review)
            self._cells[surah] = cell
            grid.addWidget(cell, (surah - 1) // cols, (surah - 1) % cols)
        scroll.setWidget(grid_host)
        root.addWidget(scroll, 1)

    # -- pickers -----------------------------------------------------------

    def _choose_audio(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose audio folder")
        if path:
            self._audio_dir = path
            self._status_label.setText(f"Audio folder: {path}")
        self._refresh_buttons()

    def _choose_db(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Open/Create project DB", "", "SQLite DB (*.db)"
        )
        if path:
            self.open_db(path)
        self._refresh_buttons()

    # -- review navigation -------------------------------------------------

    def _audio_for_surah(self, surah: int) -> str | None:
        """Find the audio file in the chosen folder whose name maps to ``surah``."""
        if not self._audio_dir:
            return None
        folder = Path(self._audio_dir)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in _AUDIO_EXTS:
                continue
            if surah_from_filename(path.name) == surah:
                return str(path)
        return None

    def _request_review(self, surah: int) -> None:
        """Resolve DB + audio for ``surah`` and emit :data:`review_requested`."""
        if not self._db_path:
            self._status_label.setText("Open a project DB before reviewing.")
            return
        audio = self._audio_for_surah(surah)
        if audio is None:
            self._status_label.setText(
                f"No audio file for surah {surah} in the chosen folder."
            )
            return
        self.review_requested.emit(self._db_path, surah, audio)

    # -- DB / status -------------------------------------------------------

    def open_db(self, path: str) -> None:
        """Open/create a project DB at ``path`` and refresh surah statuses."""
        from surah_cutter.data.project_db import ProjectDB

        # Ensure schema/version exist (open is idempotent), then read back.
        ProjectDB.open(path, _DB_CODE, _DB_VERSION).close()
        self._db_path = path
        self._status_label.setText(f"Project DB: {path}")
        self.refresh_statuses()

    def _read_reviews(self) -> dict[int, list[str]]:
        """Map surah -> list of per-ayah statuses from the open DB."""
        from surah_cutter.data.project_db import ProjectDB

        reviews: dict[int, list[str]] = {}
        if not self._db_path:
            return reviews
        with ProjectDB.open(self._db_path, _DB_CODE, _DB_VERSION) as db:
            timings = db.get_timings()
            for sura, ayah, _time in timings:
                rev = db.get_review(sura, ayah)
                status = (rev or {}).get("status", "auto")
                reviews.setdefault(sura, []).append(str(status))
        return reviews

    def refresh_statuses(self) -> None:
        """Recompute every cell's badge from the open DB."""
        reviews = self._read_reviews()
        for surah, cell in self._cells.items():
            status = _surah_status(reviews, self._counts.get(surah, 0), surah)
            cell.set_status(status)

    def _done_suras(self) -> set[int]:
        """Suras with any existing timing (skipped on resume)."""
        return set(self._read_reviews())

    # -- generation lifecycle ---------------------------------------------

    def _start_generation(self) -> None:
        if not (self._audio_dir and self._db_path):
            return
        if self._thread is not None:
            return  # already running

        self._progress.setRange(0, 0)  # busy until first progress tick
        self._status_label.setText("Starting generation…")
        self._set_running(True)

        thread = QThread(self)
        worker = GenerationWorker(
            self._audio_dir, self._db_path, self._done_suras()
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.surah_done.connect(self._on_surah_done)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _cancel_generation(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status_label.setText("Cancelling…")

    def _on_progress(self, done: int, total: int, message: str) -> None:
        if self._progress.maximum() == 0:
            self._progress.setRange(0, total)
        self._progress.setValue(done)
        self._status_label.setText(message)

    def _on_surah_done(self, surah: int) -> None:
        reviews = self._read_reviews()
        status = _surah_status(reviews, self._counts.get(surah, 0), surah)
        self._cells[surah].set_status(status)

    def _on_finished(self, message: str) -> None:
        self._status_label.setText(message)
        self.refresh_statuses()
        self._teardown_thread()

    def _on_error(self, message: str) -> None:
        self._status_label.setText(f"Error: {message}")
        self._teardown_thread()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._progress.setRange(0, 100)
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self._gen_btn.setEnabled(not running and self._can_generate())
        self._cancel_btn.setEnabled(running)
        self._audio_btn.setEnabled(not running)
        self._db_btn.setEnabled(not running)

    def _can_generate(self) -> bool:
        return bool(self._audio_dir and self._db_path)

    def _refresh_buttons(self) -> None:
        self._gen_btn.setEnabled(self._thread is None and self._can_generate())
