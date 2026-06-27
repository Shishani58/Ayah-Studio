"""Opt-in ayah-audio fingerprint alignment.

This path is useful only when the user has individual ayah audio clips from the
same recording as the full sura file. It matches simple amplitude envelopes,
then returns per-ayah starts. It is intentionally separate from CTC modes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_AUDIO_EXTS = (".mp3", ".m4a", ".ogg", ".flac", ".wav")


def align_surah_by_fingerprint(
    audio_path: str,
    surah: int,
    words: list,
    *,
    ayah_audio_dir: str | Path | None = None,
) -> list[tuple[int, int, int]]:
    """Match per-ayah clips into a full sura recording by envelope correlation."""
    from surah_cutter.core import decode

    directory = _ayah_dir(ayah_audio_dir)
    full = decode.decode(audio_path)
    sr = decode.DEFAULT_SR
    full_env = _envelope(full, sr)
    if full_env.size < 4:
        raise ValueError("fingerprint: full audio is too short")

    ayahs = sorted({w.ayah for w in words})
    clip_paths = {a: _find_ayah_file(directory, surah, a) for a in ayahs}
    missing = [a for a, p in clip_paths.items() if p is None]
    if missing:
        raise FileNotFoundError(
            f"fingerprint: missing ayah audio for sura {surah}, ayahs {missing[:8]}"
        )

    total_frames = full_env.size
    starts: dict[int, int] = {}
    prev_frame = 0
    for idx, ayah in enumerate(ayahs):
        clip = decode.decode(str(clip_paths[ayah]))
        clip_env = _envelope(clip, sr)
        if clip_env.size < 4:
            starts[ayah] = _frame_to_ms(prev_frame)
            continue

        prior = int(total_frames * idx / max(1, len(ayahs)))
        next_prior = int(total_frames * (idx + 1) / max(1, len(ayahs)))
        lo = max(prev_frame, prior - max(100, clip_env.size))
        hi = min(total_frames, next_prior + max(200, clip_env.size * 2))
        if hi - lo < clip_env.size:
            lo, hi = prev_frame, total_frames

        start_frame = _best_match(full_env, clip_env, lo, hi)
        starts[ayah] = _frame_to_ms(start_frame)
        prev_frame = max(prev_frame, start_frame)

    rows: list[tuple[int, int, int]] = []
    prev_ms = 0
    for ayah in ayahs:
        ms = max(prev_ms, starts[ayah])
        rows.append((surah, ayah, ms))
        prev_ms = ms
    return rows


def _ayah_dir(ayah_audio_dir: str | Path | None) -> Path:
    raw = ayah_audio_dir or os.environ.get("SURAH_CUTTER_AYAH_AUDIO_DIR")
    if not raw:
        raise ValueError(
            "fingerprint mode needs ayah_audio_dir or SURAH_CUTTER_AYAH_AUDIO_DIR"
        )
    path = Path(raw)
    if not path.is_dir():
        raise FileNotFoundError(f"fingerprint ayah audio dir not found: {path}")
    return path


def _find_ayah_file(directory: Path, surah: int, ayah: int) -> Path | None:
    stems = (
        f"{surah:03d}{ayah:03d}",
        f"{surah:03d}_{ayah:03d}",
        f"{surah:03d}-{ayah:03d}",
        f"{surah}_{ayah}",
        f"{ayah:03d}",
        str(ayah),
    )
    dirs = (directory, directory / f"{surah:03d}", directory / str(surah))
    for base in dirs:
        for stem in stems:
            for ext in _AUDIO_EXTS:
                path = base / f"{stem}{ext}"
                if path.exists():
                    return path
    return None


def _envelope(pcm: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(sr * 0.05))
    n = (len(pcm) // hop) * hop
    if n <= 0:
        return np.asarray([], dtype=np.float32)
    env = np.mean(np.abs(np.asarray(pcm[:n], dtype=np.float32).reshape(-1, hop)), axis=1)
    env = env - float(env.mean())
    std = float(env.std())
    if std > 1e-6:
        env = env / std
    return np.ascontiguousarray(env, dtype=np.float32)


def _best_match(full_env: np.ndarray, clip_env: np.ndarray, lo: int, hi: int) -> int:
    hi = min(hi, full_env.size)
    lo = max(0, lo)
    if hi - lo < clip_env.size:
        return lo
    segment = full_env[lo:hi]
    corr = np.correlate(segment, clip_env, mode="valid")
    return lo + int(np.argmax(corr))


def _frame_to_ms(frame: int) -> int:
    return int(round(frame * 50.0))
