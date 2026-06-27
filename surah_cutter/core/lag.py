"""Conservative per-sura timing lag calibration."""

from __future__ import annotations

from statistics import median


def calibrate_rows_to_pauses(
    pcm,
    sr: int,
    rows: list[tuple[int, int, int]],
    *,
    surah: int | None = None,
    frozen: set[int] | None = None,
    max_adjust_ms: int = 80,
) -> tuple[list[tuple[int, int, int]], int]:
    """Shift non-frozen rows by a small median offset to nearby pause endings.

    CTC lag compensation is mostly model-wide, but individual recordings can
    still carry a small residual. We estimate it only from boundaries already
    very close to a clear pause end, and cap the adjustment tightly.
    """
    if not rows:
        return rows, 0

    from surah_cutter.core import candidates

    frozen = frozen or set()
    pauses = candidates.detect_combined_pauses(pcm, sr, surah=surah)
    pause_ends = [p.end_ms for p in pauses if p.depth_ms >= 250.0]
    if len(pause_ends) < 3:
        return rows, 0

    offsets: list[float] = []
    for _s, ayah, start in rows:
        if ayah in frozen:
            continue
        nearest = min(pause_ends, key=lambda p: abs(p - start))
        delta = nearest - start
        if abs(delta) <= 180.0:
            offsets.append(delta)

    if len(offsets) < 3:
        return rows, 0

    adjust = int(round(median(offsets)))
    adjust = max(-max_adjust_ms, min(max_adjust_ms, adjust))
    if abs(adjust) < 10:
        return rows, 0

    out: list[tuple[int, int, int]] = []
    prev = 0
    for sura, ayah, start in rows:
        new_t = start if ayah in frozen else max(0, start + adjust)
        if new_t < prev:
            new_t = prev
        prev = new_t
        out.append((sura, ayah, int(new_t)))
    return out, adjust
