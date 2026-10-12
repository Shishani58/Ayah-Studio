"""Shared project state + small helpers for the web GUI.

Holds the currently-open project (audio folder + working DB path + the
sura-number → audio-file map) and reusable helpers (filename → sura number,
audio discovery, sura-status classification, waveform peaks). The HTTP layer in
:mod:`surah_cutter.webgui.server` reads/writes this state; the heavy work
(decode/align) stays in :mod:`surah_cutter.core`.
"""

from __future__ import annotations

import re
from pathlib import Path

# db_version identity for working/exported DBs (mirrors gui/dashboard + cli).
DB_CODE = "surah-cutter"
DB_VERSION = 1

#: Audio file extensions a sura file may use (decode.decode handles them all).
AUDIO_EXTS = (".mp3", ".m4a", ".ogg", ".flac", ".wav")

#: Sura number = the leading digit run of the filename stem (``003.mp3`` /
#: ``03_x.m4a`` / ``3`` → 3). Same rule as the Qt dashboard/CLI.
_NUM_RE = re.compile(r"^(\d{1,3})")


def surah_from_name(name: str) -> int | None:
    """Return the sura number parsed from a filename, or ``None``.

    Matches the leading digit run of the stem and accepts 1..114.
    """
    stem = Path(name).stem
    m = _NUM_RE.match(stem)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 114 else None


def discover_audio(audio_dir: str | Path) -> dict[int, Path]:
    """Map sura number → audio file path for a folder (best file per sura)."""
    out: dict[int, Path] = {}
    d = Path(audio_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        n = surah_from_name(p.name)
        if n is not None and n not in out:
            out[n] = p
    return out


def classify_status(statuses: list[str] | None, ayah_count: int) -> str:
    """Classify a sura from its per-ayah review statuses.

    ``statuses`` contains one status for each timing row (missing metadata
    defaults to ``auto``).
    Returns one of ``not-processed`` / ``partial`` / ``processed`` / ``verified``.
    """
    if not statuses:
        return "not-processed"
    if len(statuses) != ayah_count:
        return "partial"
    if all(s == "verified" for s in statuses):
        return "verified"
    return "processed"


MIN_WAVEFORM_BUCKETS = 1600
MAX_WAVEFORM_BUCKETS = 65536
WAVEFORM_BUCKET_MS = 100


def waveform_peaks(audio_path: str, buckets: int | None = None) -> tuple[list[int], int]:
    """Downsampled |amplitude| envelope for the waveform canvas.

    Decodes the audio to 16 kHz mono PCM and reduces it to peak values scaled
    to 0..100 (small ints → compact JSON). By default the bucket count grows
    with audio duration, capped so long suras still get enough horizontal zoom
    detail without returning an unbounded JSON payload. Returns
    ``(peaks, duration_ms)``.
    """
    import numpy as np

    from surah_cutter.core import decode

    pcm = decode.decode(audio_path)
    n = pcm.shape[0]
    duration_ms = int(round(n / decode.DEFAULT_SR * 1000))
    if n == 0:
        return [], 0
    if buckets is None:
        buckets = max(
            MIN_WAVEFORM_BUCKETS,
            min(MAX_WAVEFORM_BUCKETS, duration_ms // WAVEFORM_BUCKET_MS),
        )
    buckets = max(1, min(buckets, n))
    # Pad to a whole number of buckets, then peak-reduce per bucket.
    edges = np.linspace(0, n, buckets + 1, dtype=np.int64)
    amp = np.abs(pcm)
    peaks = np.empty(buckets, dtype=np.float64)
    for i in range(buckets):
        a, b = edges[i], edges[i + 1]
        peaks[i] = amp[a:b].max() if b > a else 0.0
    peak = peaks.max()
    if peak > 0:
        peaks = peaks / peak * 100.0
    return [int(round(v)) for v in peaks], duration_ms


class AppState:
    """Mutable singleton holding the open project (audio dir + DB + audio map)."""

    def __init__(self) -> None:
        self.audio_dir: Path | None = None
        self.db_path: Path | None = None
        self.audio_map: dict[int, Path] = {}

    def set_audio_dir(self, path: str) -> None:
        self.audio_dir = Path(path)
        self.audio_map = discover_audio(self.audio_dir)

    def set_db_path(self, path: str) -> None:
        self.db_path = Path(path)

    def audio_for(self, surah: int) -> Path | None:
        return self.audio_map.get(surah)


#: Process-wide project state (one open project per running app).
STATE = AppState()
