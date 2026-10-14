"""Phase-5 review screen: audio player + waveform + Quran text + boundary editing.

A :class:`ReviewView` opens one surah of an existing project DB plus its audio
file and lets a human verify/adjust per-ayah start times:

* **Player** — ``QMediaPlayer`` + ``QAudioOutput`` with play/pause, seek, a time
  readout, play-this-ayah (start→next start), loop-ayah, and a playback-speed
  selector (0.5/0.75/1.0).
* **Waveform** — a ``pyqtgraph`` plot of the *downsampled* decoded audio (never
  raw millions of points). One draggable ``InfiniteLine`` per ayah start plus a
  playhead line; dragging a boundary rewrites that ayah's start.
* **Text** — the surah ayat rendered RTL in the bundled ``qpc_hafs`` font; the
  ayah under the playhead is highlighted, and clicking an ayah selects it.
* **Editing** — all writes go through :class:`~surah_cutter.data.project_db.ProjectDB`
  (autosave): drag boundary, nudge ±10/±50 ms, set-boundary-at-playhead, mark
  verified, with in-session undo/redo.

Decoding happens once, off the UI thread, in :class:`_DecodeWorker`. Heavy
imports (``core.decode``, ``numpy``) live inside that worker; this module's
top-level imports are PySide6 + pyqtgraph + pure-Python ``data`` helpers only.
"""

from __future__ import annotations

from surah_cutter.resources import resource_path

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

# DB identity must match the dashboard so we reopen the same project.
_DB_CODE = "surah-cutter"
_DB_VERSION = 1

# Bundled Quran font: <repo>/assets/fonts/qpc_hafs.ttf.
_FONT_PATH = resource_path("assets", "fonts", "qpc_hafs.ttf")

# Confidence -> colour for the ayah's boundary line / chip. Low confidence is
# flagged red so the reviewer focuses there first; None means unknown (grey).
_CONF_HIGH = 0.66
_CONF_MID = 0.33
_COLOR_HIGH = "#2f9e44"
_COLOR_MID = "#e8a13a"
_COLOR_LOW = "#e03131"
_COLOR_NONE = "#888888"
_COLOR_VERIFIED = "#1f8fff"


def _confidence_color(review: dict | None) -> str:
    """Map a ``review_meta`` dict to a boundary colour."""
    if review is not None and review.get("status") == "verified":
        return _COLOR_VERIFIED
    conf = None if review is None else review.get("confidence")
    if conf is None:
        return _COLOR_NONE
    conf = float(conf)
    if conf >= _CONF_HIGH:
        return _COLOR_HIGH
    if conf >= _CONF_MID:
        return _COLOR_MID
    return _COLOR_LOW


class _DecodeWorker(QObject):
    """Decodes the audio once (off the UI thread) and downsamples for display.

    Emits ``done(times_ms, mins, maxs, duration_ms)`` where ``times_ms`` is the
    x-axis (bucket centre in ms) and ``mins``/``maxs`` are the per-bucket
    envelope used to draw the waveform without plotting every sample.
    """

    done = Signal(object, object, object, int)  # x_ms, mins, maxs, duration_ms
    error = Signal(str)

    # Target number of envelope buckets across the whole file. Keeps the plot
    # to a few thousand points regardless of audio length.
    _BUCKETS = 4000

    def __init__(self, audio_path: str) -> None:
        super().__init__()
        self._audio_path = audio_path

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # surface to UI rather than crash the thread
            self.error.emit(str(exc))

    def _run(self) -> None:
        import numpy as np

        from surah_cutter.core.decode import DEFAULT_SR, decode

        pcm = decode(self._audio_path, sr=DEFAULT_SR)
        n = int(pcm.shape[0])
        duration_ms = int(round(n * 1000 / DEFAULT_SR)) if n else 0
        if n == 0:
            self.done.emit(np.zeros(0), np.zeros(0), np.zeros(0), 0)
            return

        buckets = min(self._BUCKETS, n)
        # Split into ~equal contiguous chunks; per-chunk min/max gives the
        # waveform envelope. array_split tolerates a non-divisible length.
        chunks = np.array_split(pcm, buckets)
        mins = np.array([float(c.min()) for c in chunks], dtype=np.float32)
        maxs = np.array([float(c.max()) for c in chunks], dtype=np.float32)
        # x = centre time of each bucket in ms.
        edges = np.linspace(0, duration_ms, buckets + 1)
        x_ms = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float64)
        self.done.emit(x_ms, mins, maxs, duration_ms)


class _AyahLabel(QLabel):
    """One clickable ayah in the centred text panel.

    Carries its ayah number and emits :data:`clicked` so the view can select it.
    Highlight state is purely presentational (stylesheet swap).
    """

    clicked = Signal(int)

    def __init__(self, ayah: int, text: str, font_family: str) -> None:
        super().__init__(text)
        self._ayah = ayah
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        font = QFont(font_family)
        font.setPointSize(26)
        self.setFont(font)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._highlighted = False
        self._apply_style()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.clicked.emit(self._ayah)
        super().mousePressEvent(event)

    def set_highlighted(self, on: bool) -> None:
        if on != self._highlighted:
            self._highlighted = on
            self._apply_style()

    def _apply_style(self) -> None:
        if self._highlighted:
            self.setStyleSheet(
                "background:#2d4a6b; color:#fff; border-radius:6px; padding:6px;"
            )
        else:
            self.setStyleSheet("color:#ddd; padding:6px;")


class ReviewView(QWidget):
    """Per-ayah boundary review for one surah.

    Construct with the open project DB path, the surah number and the audio
    file path. The view reads timings/review-meta itself (reopening the DB by
    path so it never shares a connection with the dashboard) and writes every
    edit straight back (autosave).
    """

    # Emitted when the user leaves the review (so the host can re-show the
    # dashboard and refresh statuses).
    back_requested = Signal()

    # Nudge amounts (ms) for the fine-adjust buttons.
    _NUDGES = (-50, -10, 10, 50)

    def __init__(
        self,
        db_path: str,
        surah: int,
        audio_path: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"surah-cutter — review surah {surah}")

        self._db_path = db_path
        self._surah = surah
        self._audio_path = audio_path

        # Lazy pure-Python data helpers (no torch/numpy at import time).
        from surah_cutter.data.quran_text import ayah_count, surah_words

        self._ayah_count = ayah_count(surah)
        self._words = surah_words(surah)

        # starts[ayah] = start_ms (int). Loaded from the DB; mutated on edit.
        self._starts: dict[int, int] = {}
        self._reviews: dict[int, dict | None] = {}
        self._selected: int = 1
        self._current_ayah: int = 1  # ayah under the playhead during playback
        self._duration_ms: int = 0
        self._loop_ayah = False

        # In-session undo/redo of boundary edits. Each entry is
        # (ayah, old_ms, new_ms); redo holds undone entries.
        self._undo: list[tuple[int, int, int]] = []
        self._redo: list[tuple[int, int, int]] = []

        self._decode_thread: QThread | None = None
        self._decode_worker: _DecodeWorker | None = None
        self._lines: dict[int, pg.InfiniteLine] = {}
        self._ayah_labels: dict[int, _AyahLabel] = {}

        self._font_family = self._load_font()
        self._setup_player()
        self._load_timings()
        self._build_ui()
        self._start_decode()
        self._refresh_text_panel()
        self._select_ayah(1, seek=False)

    # -- setup -------------------------------------------------------------

    def _load_font(self) -> str:
        """Register the bundled qpc_hafs font; return its family name."""
        family = "qpc_hafs"
        if _FONT_PATH.exists():
            fid = QFontDatabase.addApplicationFont(str(_FONT_PATH))
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                family = fams[0]
        return family

    def _setup_player(self) -> None:
        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._player.setSource(QUrl.fromLocalFile(self._audio_path))
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)

    def _load_timings(self) -> None:
        """Read this surah's starts + review-meta from the project DB."""
        from surah_cutter.data.project_db import ProjectDB

        with ProjectDB.open(self._db_path, _DB_CODE, _DB_VERSION) as db:
            for sura, ayah, time in db.get_timings():
                if sura == self._surah:
                    self._starts[ayah] = int(time)
            for ayah in range(1, self._ayah_count + 1):
                self._reviews[ayah] = db.get_review(self._surah, ayah)

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        root.addLayout(self._build_topbar())
        root.addWidget(self._build_waveform(), 2)
        root.addLayout(self._build_transport())
        root.addLayout(self._build_edit_bar())
        root.addWidget(self._build_text_panel(), 3)

        self._status = QLabel("")
        root.addWidget(self._status)

    def _build_topbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._back_btn.clicked.connect(self.back_requested.emit)
        bar.addWidget(self._back_btn)

        bar.addWidget(QLabel(f"<b>Surah {self._surah}</b>"))

        bar.addWidget(QLabel("Ayah:"))
        self._prev_btn = QPushButton("‹ Prev")
        self._next_btn = QPushButton("Next ›")
        self._prev_btn.clicked.connect(self._prev_ayah)
        self._next_btn.clicked.connect(self._next_ayah)
        bar.addWidget(self._prev_btn)
        bar.addWidget(self._next_btn)

        bar.addWidget(QLabel("Jump:"))
        self._jump = QSpinBox()
        self._jump.setRange(1, self._ayah_count)
        self._jump.valueChanged.connect(lambda v: self._select_ayah(v))
        bar.addWidget(self._jump)

        bar.addStretch(1)
        self._sel_label = QLabel("")
        bar.addWidget(self._sel_label)
        return bar

    def _build_waveform(self) -> QWidget:
        self._plot = pg.PlotWidget(background="#1e1e1e")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.getPlotItem().hideButtons()
        self._plot.setYRange(-1.0, 1.0, padding=0)
        self._plot.getAxis("left").setStyle(showValues=False)
        # Placeholder curve; filled in once decode finishes.
        self._wave_curve = self._plot.plot([], [], pen=pg.mkPen("#5b8db8", width=1))
        # Playhead (movable=False; we drive it from the player position).
        self._playhead = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen("#f1c40f", width=2)
        )
        self._plot.addItem(self._playhead)
        return self._plot

    def _build_transport(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_play)
        bar.addWidget(self._play_btn)

        self._play_ayah_btn = QPushButton("Play ayah")
        self._play_ayah_btn.clicked.connect(self._play_current_ayah)
        bar.addWidget(self._play_ayah_btn)

        self._loop_btn = QPushButton("Loop ayah")
        self._loop_btn.setCheckable(True)
        self._loop_btn.toggled.connect(self._on_loop_toggled)
        bar.addWidget(self._loop_btn)

        bar.addWidget(QLabel("Speed:"))
        self._speed = QComboBox()
        for label in ("0.5", "0.75", "1.0"):
            self._speed.addItem(label)
        self._speed.setCurrentText("1.0")
        self._speed.currentTextChanged.connect(self._on_speed)
        bar.addWidget(self._speed)

        self._seek = QSlider(Qt.Orientation.Horizontal)
        self._seek.setRange(0, 0)
        self._seek.sliderMoved.connect(self._on_seek_moved)
        bar.addWidget(self._seek, 1)

        self._time_label = QLabel("0:00 / 0:00")
        bar.addWidget(self._time_label)
        return bar

    def _build_edit_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Start (ms):"))
        self._start_spin = QSpinBox()
        self._start_spin.setRange(0, 10_000_000)
        self._start_spin.setSingleStep(10)
        self._start_spin.editingFinished.connect(self._on_start_spin)
        bar.addWidget(self._start_spin)

        self._end_label = QLabel("End: —")
        bar.addWidget(self._end_label)

        for ms in self._NUDGES:
            btn = QPushButton(f"{ms:+d}")
            btn.clicked.connect(lambda _=False, d=ms: self._nudge(d))
            bar.addWidget(btn)

        self._set_here_btn = QPushButton("Set @ playhead")
        self._set_here_btn.clicked.connect(self._set_boundary_at_playhead)
        bar.addWidget(self._set_here_btn)

        self._verify_btn = QPushButton("Mark verified")
        self._verify_btn.clicked.connect(self._mark_verified)
        bar.addWidget(self._verify_btn)

        bar.addStretch(1)
        self._undo_btn = QPushButton("Undo")
        self._redo_btn = QPushButton("Redo")
        self._undo_btn.clicked.connect(self._do_undo)
        self._redo_btn.clicked.connect(self._do_redo)
        bar.addWidget(self._undo_btn)
        bar.addWidget(self._redo_btn)
        self._refresh_undo_buttons()
        return bar

    def _build_text_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        self._text_layout = QVBoxLayout(host)
        self._text_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(host)
        self._text_scroll = scroll
        return scroll

    # -- text panel --------------------------------------------------------

    def _refresh_text_panel(self) -> None:
        """(Re)build the centred RTL ayah labels."""
        # Clear existing.
        while self._text_layout.count():
            item = self._text_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._ayah_labels.clear()

        for ayah in range(1, self._ayah_count + 1):
            text = self._ayah_text(ayah)
            row = QFrame()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            num = QLabel(f"{ayah}")
            num.setStyleSheet("color:#888; padding:6px;")
            label = _AyahLabel(ayah, text, self._font_family)
            label.clicked.connect(self._select_ayah)
            self._ayah_labels[ayah] = label
            row_layout.addWidget(num)
            row_layout.addWidget(label, 1)
            self._text_layout.addWidget(row)
        self._highlight_ayah(self._current_ayah)

    def _ayah_text(self, ayah: int) -> str:
        """QPC Hafs words for ``ayah`` joined by a single space (recitation order)."""
        return " ".join(w.text for w in self._words if w.ayah == ayah)

    def _highlight_ayah(self, ayah: int) -> None:
        for a, label in self._ayah_labels.items():
            label.set_highlighted(a == ayah)

    # -- waveform / decode -------------------------------------------------

    def _start_decode(self) -> None:
        self._set_status("Decoding audio…")
        thread = QThread(self)
        worker = _DecodeWorker(self._audio_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_decoded)
        worker.error.connect(self._on_decode_error)
        self._decode_thread = thread
        self._decode_worker = worker
        thread.start()

    def _on_decoded(self, x_ms, mins, maxs, duration_ms: int) -> None:
        # Draw the min/max envelope as a vertical-fill style step. We plot the
        # envelope as two stacked curves sharing x, filled between, which keeps
        # the point count to ~2*buckets.
        import numpy as np

        if len(x_ms):
            x = np.asarray(x_ms, dtype=np.float64)
            top = self._plot.plot(x, np.asarray(maxs), pen=pg.mkPen("#5b8db8", width=1))
            bot = self._plot.plot(x, np.asarray(mins), pen=pg.mkPen("#5b8db8", width=1))
            fill = pg.FillBetweenItem(top, bot, brush=pg.mkBrush("#3a6a91"))
            self._plot.addItem(fill)
            # Remove the placeholder curve.
            self._plot.removeItem(self._wave_curve)
        if duration_ms and not self._duration_ms:
            self._duration_ms = duration_ms
        self._rebuild_lines()
        self._teardown_decode()
        self._set_status("Ready.")

    def _on_decode_error(self, message: str) -> None:
        self._set_status(f"Decode error: {message}")
        self._teardown_decode()

    def _teardown_decode(self) -> None:
        if self._decode_thread is not None:
            self._decode_thread.quit()
            self._decode_thread.wait()
            self._decode_thread = None
        self._decode_worker = None

    def _rebuild_lines(self) -> None:
        """Create one draggable InfiniteLine per ayah start."""
        for line in self._lines.values():
            self._plot.removeItem(line)
        self._lines.clear()
        for ayah, start in self._starts.items():
            color = _confidence_color(self._reviews.get(ayah))
            line = pg.InfiniteLine(
                pos=start,
                angle=90,
                movable=True,
                pen=pg.mkPen(color, width=2),
                label=str(ayah),
                labelOpts={"position": 0.95, "color": color},
            )
            line.sigPositionChangeFinished.connect(
                lambda ln=line, a=ayah: self._on_line_dragged(a, ln)
            )
            self._plot.addItem(line)
            self._lines[ayah] = line

    def _on_line_dragged(self, ayah: int, line: pg.InfiniteLine) -> None:
        new_ms = max(0, int(round(float(line.value()))))
        self._apply_edit(ayah, new_ms, record=True)

    # -- selection / navigation -------------------------------------------

    def _select_ayah(self, ayah: int, seek: bool = True) -> None:
        ayah = max(1, min(self._ayah_count, int(ayah)))
        self._selected = ayah
        self._current_ayah = ayah
        # Keep the spinbox in sync without re-triggering selection.
        if self._jump.value() != ayah:
            blocked = self._jump.blockSignals(True)
            self._jump.setValue(ayah)
            self._jump.blockSignals(blocked)
        self._highlight_ayah(ayah)
        self._refresh_edit_fields()
        self._sel_label.setText(f"Selected ayah {ayah}/{self._ayah_count}")
        if seek and ayah in self._starts:
            self._player.setPosition(self._starts[ayah])

    def _prev_ayah(self) -> None:
        self._select_ayah(self._selected - 1)

    def _next_ayah(self) -> None:
        self._select_ayah(self._selected + 1)

    def _next_start_ms(self, ayah: int) -> int | None:
        """Start of ``ayah+1`` (= end of ``ayah``), or None at the last ayah."""
        return self._starts.get(ayah + 1)

    # -- edit fields -------------------------------------------------------

    def _refresh_edit_fields(self) -> None:
        ayah = self._selected
        start = self._starts.get(ayah)
        blocked = self._start_spin.blockSignals(True)
        self._start_spin.setValue(start if start is not None else 0)
        self._start_spin.setEnabled(start is not None)
        self._start_spin.blockSignals(blocked)
        end = self._next_start_ms(ayah)
        if start is None:
            self._end_label.setText("End: — (no timing)")
        elif end is None:
            self._end_label.setText("End: (surah end)")
        else:
            self._end_label.setText(f"End: {end} ms")

    def _on_start_spin(self) -> None:
        if not self._start_spin.isEnabled():
            return
        self._apply_edit(self._selected, int(self._start_spin.value()), record=True)

    # -- editing (all autosave via ProjectDB) ------------------------------

    def _apply_edit(self, ayah: int, new_ms: int, *, record: bool) -> bool:
        """Set ``ayah``'s start to ``new_ms``: persist, log undo, refresh UI."""
        new_ms = max(0, int(new_ms))
        old_ms = self._starts.get(ayah)
        if old_ms is not None and old_ms == new_ms:
            return True
        try:
            self._persist_time(ayah, new_ms)
        except ValueError as exc:
            if old_ms is not None:
                self._sync_line(ayah)
            self._refresh_edit_fields()
            self._set_status(str(exc))
            return False
        self._starts[ayah] = new_ms
        if record and old_ms is not None:
            self._undo.append((ayah, old_ms, new_ms))
            self._redo.clear()
            self._refresh_undo_buttons()
        self._sync_line(ayah)
        if ayah == self._selected:
            self._refresh_edit_fields()
        self._set_status(f"Ayah {ayah} start → {new_ms} ms")
        return True

    def _persist_time(self, ayah: int, new_ms: int) -> None:
        """Write the new start to the DB (set_time → status 'edited', history)."""
        from surah_cutter.data.project_db import ProjectDB

        with ProjectDB.open(self._db_path, _DB_CODE, _DB_VERSION) as db:
            db.set_time(self._surah, ayah, new_ms)
            self._reviews[ayah] = db.get_review(self._surah, ayah)

    def _sync_line(self, ayah: int) -> None:
        """Move the boundary line to match ``starts[ayah]`` and recolour it."""
        line = self._lines.get(ayah)
        if line is None:
            return
        blocked = line.blockSignals(True)
        line.setValue(self._starts[ayah])
        line.blockSignals(blocked)
        color = _confidence_color(self._reviews.get(ayah))
        line.setPen(pg.mkPen(color, width=2))

    def _nudge(self, delta_ms: int) -> None:
        start = self._starts.get(self._selected)
        if start is None:
            return
        self._apply_edit(self._selected, start + delta_ms, record=True)

    def _set_boundary_at_playhead(self) -> None:
        self._apply_edit(self._selected, int(self._player.position()), record=True)

    def _mark_verified(self) -> None:
        from surah_cutter.data.project_db import ProjectDB

        with ProjectDB.open(self._db_path, _DB_CODE, _DB_VERSION) as db:
            db.set_status(self._surah, self._selected, "verified")
            self._reviews[self._selected] = db.get_review(self._surah, self._selected)
        self._sync_line(self._selected)
        self._set_status(f"Ayah {self._selected} marked verified.")

    def _do_undo(self) -> None:
        if not self._undo:
            return
        ayah, old_ms, new_ms = self._undo.pop()
        if self._apply_edit(ayah, old_ms, record=False):
            self._redo.append((ayah, old_ms, new_ms))
        else:
            self._undo.append((ayah, old_ms, new_ms))
        self._refresh_undo_buttons()

    def _do_redo(self) -> None:
        if not self._redo:
            return
        ayah, old_ms, new_ms = self._redo.pop()
        if self._apply_edit(ayah, new_ms, record=False):
            self._undo.append((ayah, old_ms, new_ms))
        else:
            self._redo.append((ayah, old_ms, new_ms))
        self._refresh_undo_buttons()

    def _refresh_undo_buttons(self) -> None:
        self._undo_btn.setEnabled(bool(self._undo))
        self._redo_btn.setEnabled(bool(self._redo))

    # -- player ------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _play_current_ayah(self) -> None:
        """Play from this ayah's start to the next ayah's start (then stop/loop)."""
        ayah = self._selected
        start = self._starts.get(ayah)
        if start is None:
            return
        self._player.setPosition(start)
        self._player.play()

    def _on_loop_toggled(self, on: bool) -> None:
        self._loop_ayah = on

    def _on_speed(self, text: str) -> None:
        try:
            self._player.setPlaybackRate(float(text))
        except ValueError:
            pass

    def _on_position(self, pos_ms: int) -> None:
        self._playhead.setValue(pos_ms)
        if self._seek.maximum() and not self._seek.isSliderDown():
            blocked = self._seek.blockSignals(True)
            self._seek.setValue(pos_ms)
            self._seek.blockSignals(blocked)
        self._time_label.setText(
            f"{_fmt_time(pos_ms)} / {_fmt_time(self._duration_ms)}"
        )
        self._follow_playback(pos_ms)

    def _follow_playback(self, pos_ms: int) -> None:
        """Highlight the ayah under the playhead; loop if looping is on."""
        # Loop: when we cross into the next ayah, jump back to this ayah start.
        if self._loop_ayah:
            start = self._starts.get(self._selected)
            end = self._next_start_ms(self._selected)
            if start is not None and end is not None and pos_ms >= end:
                self._player.setPosition(start)
                return
        ayah = self._ayah_at(pos_ms)
        if ayah is not None and ayah != self._current_ayah:
            self._current_ayah = ayah
            self._highlight_ayah(ayah)
            self._scroll_to_ayah(ayah)

    def _ayah_at(self, pos_ms: int) -> int | None:
        """Which ayah's window contains ``pos_ms`` (largest start ≤ pos)."""
        best: int | None = None
        best_start = -1
        for ayah, start in self._starts.items():
            if start <= pos_ms and start > best_start:
                best_start = start
                best = ayah
        return best

    def _scroll_to_ayah(self, ayah: int) -> None:
        label = self._ayah_labels.get(ayah)
        if label is not None:
            self._text_scroll.ensureWidgetVisible(label)

    def _on_duration(self, dur_ms: int) -> None:
        self._duration_ms = dur_ms
        self._seek.setRange(0, dur_ms)
        self._time_label.setText(f"0:00 / {_fmt_time(dur_ms)}")

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("Pause" if playing else "Play")

    def _on_seek_moved(self, pos_ms: int) -> None:
        self._player.setPosition(pos_ms)

    # -- misc --------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self._status.setText(message)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._player.stop()
        self._teardown_decode()
        super().closeEvent(event)


def _fmt_time(ms: int) -> str:
    """``mm:ss`` from milliseconds."""
    total = int(ms) // 1000
    return f"{total // 60}:{total % 60:02d}"
