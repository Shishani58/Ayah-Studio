"""Opt-in ayah-audio fingerprint alignment.

This path is useful only when the user has individual ayah audio clips from the
same recording as the full sura file. It matches simple amplitude envelopes,
then returns per-ayah starts. It is intentionally separate from CTC modes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

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


def refine_fast_rows_by_fingerprint(
    pcm: np.ndarray,
    sr: int,
    rows: list[tuple[int, int, float]],
    surah: int,
    *,
    ayah_audio_dir: str | Path,
    search_radius_ms: int = 15_000,
    max_shift_ms: int = 2_000,
    max_strong_forward_ms: int = 2_400,
    max_rollback_ms: int = 15_000,
    min_correlation: float = 0.80,
    min_gap_ms: int = 30,
    owner_intervals: Mapping[int, tuple[float, float]] | None = None,
) -> tuple[list[tuple[int, int, float]], list[dict[str, int | float | str]]]:
    """Refine supplied FAST rows from matching clips of the *same* recording.

    This is intentionally opt-in and partial: absent clip files are ignored.
    A clip may replace its FAST position only after a normalized envelope match
    in the local search interval is strong and remains strictly between its
    neighbours. A bounded forward correction keeps a local match from becoming
    a future-skip when a repeated passage happens to sound similar. A much
    stronger, source-consensus-gated match may roll a boundary back locally;
    it cannot change the ordering of the sura.
    """
    from surah_cutter.core import decode

    directory = _ayah_dir(ayah_audio_dir)
    full_env = _envelope(pcm, sr)
    if full_env.size < 4:
        return rows, []
    radius_frames = max(1, int(round(search_radius_ms / 50.0)))
    max_shift_frames = max(1, int(round(max_shift_ms / 50.0)))
    max_strong_forward_frames = max(
        max_shift_frames,
        int(round(max_strong_forward_ms / 50.0)),
    )
    max_rollback_frames = max(max_shift_frames, int(round(max_rollback_ms / 50.0)))
    minimum = float(min_correlation)
    refined: list[tuple[int, int, float]] = []
    repairs: list[dict[str, int | float | str]] = []
    full_log: np.ndarray | None = None
    full_delta: np.ndarray | None = None
    available_clips = 0
    near_repairs = 0
    block_forward_candidates: list[dict[str, int | float]] = []
    for index, (ayah, fast_ms, confidence) in enumerate(rows):
        clip_path = _find_ayah_file(directory, surah, ayah)
        if clip_path is None:
            refined.append((ayah, fast_ms, confidence))
            continue
        available_clips += 1
        clip_pcm = decode.decode(str(clip_path))
        clip_env = _envelope(clip_pcm, sr)
        if clip_env.size < 4:
            refined.append((ayah, fast_ms, confidence))
            continue
        prior_frame = max(0, int(round(fast_ms / 50.0)))
        lo = max(0, prior_frame - radius_frames)
        hi = min(full_env.size, prior_frame + radius_frames + clip_env.size)
        match_frame, score = _best_normalized_match(full_env, clip_env, lo, hi)
        evidence: dict[str, float | str] = {"match_method": "envelope"}
        corroborated = False
        strong_energy_owner_evidence = False
        if score < minimum:
            if full_log is None:
                full_log = _log_rms_envelope(pcm, sr)
                full_delta = _delta_envelope(full_log)
            clip_log = _log_rms_envelope(clip_pcm, sr)
            clip_delta = _delta_envelope(clip_log)
            log_frame, log_score = _best_normalized_match(full_log, clip_log, lo, hi)
            delta_frame, delta_score = _best_normalized_match(
                full_delta, clip_delta, lo, hi
            )
            corroborated = (
                score >= 0.70
                and log_score >= 0.72
                and delta_score >= 0.85
                and max(abs(match_frame - log_frame), abs(match_frame - delta_frame)) <= 2
            )
            if corroborated:
                evidence = {
                    "match_method": "energy_consensus",
                    "log_correlation": round(log_score, 4),
                    "delta_correlation": round(delta_score, 4),
                }
            # A quiet ayah can have a weak amplitude envelope while its log
            # energy and energy transitions still identify the same frame.
            # This deliberately is not a general lower threshold: it can only
            # move a boundary inside its already claimed pause owner below.
            strong_energy_owner_evidence = (
                (
                    score >= 0.60
                    and log_score >= 0.78
                    and delta_score >= 0.90
                )
                or (
                    # Full CTC oracle checks on 24:19/21 show that a clipped
                    # transition may weaken delta while amplitude and log-RMS
                    # still agree on the exact owner-internal onset.
                    score >= 0.65
                    and log_score >= 0.85
                    and delta_score >= 0.65
                )
            ) and max(
                abs(match_frame - log_frame), abs(match_frame - delta_frame)
            ) <= 2
        matched_ms = _frame_to_ms(match_frame)
        previous_ms = refined[-1][1] if refined else -min_gap_ms
        next_ms = rows[index + 1][1] if index + 1 < len(rows) else None
        within_neighbours = (
            matched_ms >= previous_ms + min_gap_ms
            and (next_ms is None or matched_ms <= next_ms - min_gap_ms)
        )
        distant_rollback = (
            match_frame < prior_frame - max_shift_frames
            and prior_frame - match_frame <= max_rollback_frames
            and score >= max(
                minimum,
                0.95 if prior_frame - match_frame > 200 else 0.85,
            )
            and within_neighbours
        )
        strong_forward = (
            match_frame > prior_frame + max_shift_frames
            and match_frame - prior_frame <= max_strong_forward_frames
            and score >= 0.90
            and within_neighbours
        )
        owner_interval = (owner_intervals or {}).get(ayah)
        owner_bound_forward = (
            owner_interval is not None
            and owner_interval[0] <= matched_ms <= owner_interval[1]
            and match_frame > prior_frame + max_strong_forward_frames
            and score >= 0.90
        )
        owner_bound_energy_forward = (
            owner_interval is not None
            and owner_interval[0] <= matched_ms <= owner_interval[1]
            and match_frame > prior_frame + 4
            and strong_energy_owner_evidence
        )
        coordinated_forward = (
            match_frame > prior_frame + max_strong_forward_frames
            and match_frame - prior_frame <= 200
            and score >= 0.90
        )
        if coordinated_forward:
            block_forward_candidates.append({
                "index": index,
                "ayah": ayah,
                "fast_ms": fast_ms,
                "matched_ms": matched_ms,
                "confidence": confidence,
                "correlation": round(score, 4),
                "search_start_ms": _frame_to_ms(lo),
                "search_end_ms": _frame_to_ms(hi),
                "clip_frames": int(clip_env.size),
            })
        if (
            (score < minimum and not corroborated and not owner_bound_energy_forward)
            or (
                abs(match_frame - prior_frame) > max_shift_frames
                and not distant_rollback
                and not strong_forward
                and not owner_bound_forward
                and not owner_bound_energy_forward
            )
            or not within_neighbours
        ):
            refined.append((ayah, fast_ms, confidence))
            continue
        if distant_rollback:
            evidence = {"match_method": "distant_rollback"}
        elif strong_forward:
            evidence = {"match_method": "strong_forward"}
            near_repairs += 1
        elif owner_bound_forward:
            evidence = {"match_method": "owner_bound_forward"}
            near_repairs += 1
        elif owner_bound_energy_forward:
            evidence = {
                "match_method": "owner_bound_energy_consensus",
                "log_correlation": round(log_score, 4),
                "delta_correlation": round(delta_score, 4),
            }
            near_repairs += 1
        else:
            near_repairs += 1
        refined.append((ayah, matched_ms, max(confidence, score)))
        repairs.append({
            "ayah": ayah,
            "fast_ms": fast_ms,
            "matched_ms": matched_ms,
            "correlation": round(score, 4),
            "search_start_ms": _frame_to_ms(lo),
            "search_end_ms": _frame_to_ms(hi),
            "clip_frames": int(clip_env.size),
            **evidence,
        })
    minimum_consensus = max(3, (available_clips + 9) // 10)
    if available_clips >= 6 and near_repairs < minimum_consensus:
        # A complete-looking folder with only isolated matches is a different
        # release of the recitation, not a safe source of per-ayah evidence.
        return rows, []
    by_index = {int(candidate["index"]): candidate for candidate in block_forward_candidates}
    indices = sorted(by_index)
    run_start = 0
    while run_start < len(indices):
        run_end = run_start + 1
        while run_end < len(indices) and indices[run_end] == indices[run_end - 1] + 1:
            run_end += 1
        run = indices[run_start:run_end]
        first, last = run[0], run[-1]
        proposed = [int(by_index[i]["matched_ms"]) for i in run]
        left = refined[first - 1][1] if first else -min_gap_ms
        right = refined[last + 1][1] if last + 1 < len(refined) else None
        if (
            len(run) >= 2
            and all(b > a + min_gap_ms for a, b in zip(proposed, proposed[1:]))
            and proposed[0] >= left + min_gap_ms
            and (right is None or proposed[-1] <= right - min_gap_ms)
        ):
            for index in run:
                candidate = by_index[index]
                ayah = int(candidate["ayah"])
                matched_ms = int(candidate["matched_ms"])
                confidence = float(candidate["confidence"])
                correlation = float(candidate["correlation"])
                refined[index] = (ayah, matched_ms, max(confidence, correlation))
                repairs.append({
                    key: value
                    for key, value in candidate.items()
                    if key not in {"index", "confidence"}
                } | {"match_method": "paired_strong_forward"})
        run_start = run_end
    return refined, repairs


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


def _log_rms_envelope(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Return a normalized log-energy envelope on the fingerprint frame grid."""
    hop = max(1, int(sr * 0.05))
    n = (len(pcm) // hop) * hop
    if n <= 0:
        return np.asarray([], dtype=np.float32)
    blocks = np.asarray(pcm[:n], dtype=np.float32).reshape(-1, hop)
    energy = np.log(np.sqrt(np.mean(np.square(blocks), axis=1)) + 1e-5)
    energy = energy - float(energy.mean())
    std = float(energy.std())
    if std > 1e-6:
        energy = energy / std
    return np.ascontiguousarray(energy, dtype=np.float32)


def _delta_envelope(envelope: np.ndarray) -> np.ndarray:
    """Emphasize phrase-shape changes independently from loudness."""
    if envelope.size < 2:
        return envelope
    delta = np.diff(envelope, prepend=envelope[0])
    delta = delta - float(delta.mean())
    std = float(delta.std())
    if std > 1e-6:
        delta = delta / std
    return np.ascontiguousarray(delta, dtype=np.float32)


def _best_match(full_env: np.ndarray, clip_env: np.ndarray, lo: int, hi: int) -> int:
    hi = min(hi, full_env.size)
    lo = max(0, lo)
    if hi - lo < clip_env.size:
        return lo
    segment = full_env[lo:hi]
    corr = np.correlate(segment, clip_env, mode="valid")
    return lo + int(np.argmax(corr))


def _best_normalized_match(
    full_env: np.ndarray,
    clip_env: np.ndarray,
    lo: int,
    hi: int,
) -> tuple[int, float]:
    """Return the local envelope match and a recording-identity score."""
    hi = min(hi, full_env.size)
    lo = max(0, lo)
    if hi - lo < clip_env.size:
        return lo, 0.0
    segment = full_env[lo:hi]
    numerator = np.correlate(segment, clip_env, mode="valid")
    clip_norm = float(np.linalg.norm(clip_env))
    if clip_norm <= 1e-9:
        return lo, 0.0
    local_energy = np.convolve(
        np.square(segment),
        np.ones(clip_env.size, dtype=np.float32),
        mode="valid",
    )
    denominator = np.sqrt(np.maximum(local_energy, 1e-9)) * clip_norm
    scores = numerator / denominator
    best = int(np.argmax(scores))
    return lo + best, float(scores[best])


def _frame_to_ms(frame: int) -> int:
    return int(round(frame * 50.0))
