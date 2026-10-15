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

from surah_cutter.data.project_state import (
    discover_audio, surah_from_name,  # noqa: F401 — compatibility API
)

# db_version identity for working/exported DBs (mirrors gui/dashboard + cli).
DB_CODE = "surah-cutter"
DB_VERSION = 1

#: Audio file extensions a sura file may use (decode.decode handles them all).

#: Sura number = the leading digit run of the filename stem (``003.mp3`` /
#: ``03_x.m4a`` / ``3`` → 3). Same rule as the Qt dashboard/CLI.


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


async def waveform_events(audio_path: str):
    """Decode incrementally, preserving a partial final bucket.

    Full-scale amplitudes stay stable during loading. The final event carries
    the exact PCM duration and normalization factor. Closing the iterator also
    stops ffmpeg; stderr goes to a file so its pipe can never block decoding.
    """
    import asyncio
    import tempfile

    from anyio import CancelScope
    import numpy as np

    from surah_cutter.core import decode

    # A metadata-only pass chooses roughly the same resolution as waveform_peaks.
    # The duration hint never determines the final duration or truncates the PCM.
    probe = await asyncio.create_subprocess_exec(
        decode._ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", audio_path,
        "-t", "0", "-f", "null", "-",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, metadata = await probe.communicate()
    finally:
        if probe.returncode is None:
            probe.kill()
            with CancelScope(shield=True):
                await probe.communicate()
    match = re.search(rb"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", metadata)
    estimated_ms = 0
    samples_per_bucket = 160  # 10 ms when a container has no duration metadata.
    if match:
        hours, minutes, seconds = map(float, match.groups())
        estimated_ms = round((hours * 3600 + minutes * 60 + seconds) * 1000)
        buckets = max(MIN_WAVEFORM_BUCKETS,
                      min(MAX_WAVEFORM_BUCKETS, estimated_ms // WAVEFORM_BUCKET_MS))
        if estimated_ms > 0:
            samples_per_bucket = max(1, round(estimated_ms * decode.DEFAULT_SR / 1000 / buckets))
    block_bytes = samples_per_bucket * 4 * 200
    sample_count = 0
    offset = 0
    maximum = 0.0
    with tempfile.TemporaryFile() as errors:
        proc = await asyncio.create_subprocess_exec(
            decode._ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", audio_path, "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(decode.DEFAULT_SR), "-",
            stdout=asyncio.subprocess.PIPE, stderr=errors,
        )
        try:
            yield {"type": "start", "bucket_ms": samples_per_bucket * 1000 / decode.DEFAULT_SR,
                   "duration_hint_ms": estimated_ms}
            while True:
                try:
                    block = await proc.stdout.readexactly(block_bytes)
                    tail = False
                except asyncio.IncompleteReadError as exc:
                    block = exc.partial
                    tail = True
                if block:
                    pcm = np.abs(np.frombuffer(block, dtype="<f4"))
                    sample_count += len(pcm)
                    peaks = [float(pcm[i:i + samples_per_bucket].max())
                             for i in range(0, len(pcm), samples_per_bucket)]
                    maximum = max(maximum, max(peaks))
                    yield {"type": "chunk", "offset": offset, "peaks": peaks,
                           "decoded_ms": round(sample_count * 1000 / decode.DEFAULT_SR)}
                    offset += len(peaks)
                if tail:
                    break
            if await proc.wait() != 0:
                errors.seek(0)
                raise decode.DecodeError(errors.read().decode("utf-8", "replace").strip())
            yield {"type": "done", "duration_ms": round(sample_count * 1000 / decode.DEFAULT_SR),
                   "maximum": maximum}
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                with CancelScope(shield=True):
                    # Drain the pipe too: Python 3.12 wait() can block on a full buffer.
                    await proc.communicate()


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
