"""Draft ayah starts from waveform pause centers.

This is intentionally cheaper and less authoritative than forced alignment. It
creates a complete rough timing pass from pauses + the known ayah count/relative
text weight, while treating manual/verified timings as anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from surah_cutter.core import candidates
from surah_cutter.data.quran_text import DEFAULT_DB_PATH


@dataclass(frozen=True)
class DraftResult:
    """Pause-draft output plus compact diagnostics for the UI."""

    rows: list[tuple[int, int, int]]
    pause_count: int
    pause_backed: int
    interpolated: int
    anchored: int


def draft_from_pauses(
    pcm,
    sr: int,
    surah: int,
    *,
    anchors: dict[int, int] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> DraftResult:
    """Build rough per-ayah starts from silence-center candidates.

    ``anchors`` should contain user-trusted timings (``edited``/``verified``).
    They are used to estimate local recitation speed and are returned unchanged;
    callers should still write through ``ProjectDB.write_timings`` so protected
    rows cannot be overwritten.
    """
    total_ms = _duration_ms(pcm, sr)
    anchors = {int(a): int(t) for a, t in (anchors or {}).items()}
    pauses = _pause_marks(pcm, sr, surah)
    expected = _expected_ms(surah, total_ms, pauses, db_path=db_path)
    ayahs = sorted(expected)
    adjusted = _anchor_adjusted_expected(ayahs, expected, anchors, total_ms)

    used: set[int] = set()
    rows: list[tuple[int, int, int]] = []
    pause_backed = 0
    interpolated = 0
    anchored = 0
    prev_time = 0

    for idx, ayah in enumerate(ayahs):
        if ayah in anchors:
            t = anchors[ayah]
            anchored += 1
        elif ayah == ayahs[0] and surah in (1, 9):
            t = 0
            interpolated += 1
        else:
            tol = _local_tolerance(ayahs, adjusted, idx, total_ms)
            pidx = _best_pause(pauses, adjusted[ayah], tol, used)
            if pidx is None:
                t = adjusted[ayah]
                interpolated += 1
            else:
                used.add(pidx)
                # Keep the review start inside the quiet shelf. Cutting at
                # pause.end_ms starts abruptly on the next phrase's attack.
                t = pauses[pidx].center_ms
                pause_backed += 1

        fixed = int(round(max(prev_time, min(total_ms, t))))
        rows.append((surah, ayah, fixed))
        prev_time = fixed

    return DraftResult(
        rows=rows,
        pause_count=len(pauses),
        pause_backed=pause_backed,
        interpolated=interpolated,
        anchored=anchored,
    )


def _duration_ms(pcm, sr: int) -> float:
    return (len(pcm) * 1000.0 / sr) if sr else 0.0


def _pause_marks(pcm, sr: int, surah: int) -> list[candidates.Pause]:
    try:
        pauses = candidates.detect_combined_pauses(pcm, sr, surah=surah)
    except RuntimeError:
        pauses = candidates.detect_pauses(pcm, sr)
    return sorted(pauses, key=lambda p: p.center_ms)


def _expected_ms(
    surah: int,
    total_ms: float,
    pauses: list[candidates.Pause],
    *,
    db_path: str | Path,
) -> dict[int, float]:
    fractions = candidates.expected_boundary_fractions(surah, db_path=db_path)
    expected = {ayah: frac * total_ms for ayah, frac in fractions.items()}
    intro = _intro_basmala_end(pauses, total_ms, surah)
    if intro is not None and 1 in expected:
        expected[1] = intro
    return expected


def _intro_basmala_end(
    pauses: list[candidates.Pause],
    total_ms: float,
    surah: int,
) -> float | None:
    if surah in (1, 9):
        return None
    limit = min(30_000.0, max(8_000.0, total_ms * 0.18))
    for p in pauses:
        if p.end_ms <= limit:
            return p.center_ms
    return None


def _anchor_adjusted_expected(
    ayahs: list[int],
    expected: dict[int, float],
    anchors: dict[int, int],
    total_ms: float,
) -> dict[int, float]:
    points: list[tuple[int, float, float]] = []
    first = ayahs[0]
    if first not in anchors:
        points.append((first, expected[first], expected[first]))
    for ayah in ayahs:
        if ayah in anchors:
            points.append((ayah, expected[ayah], float(anchors[ayah])))
    points.append((ayahs[-1] + 1, total_ms, total_ms))
    points.sort(key=lambda item: item[0])

    out: dict[int, float] = {}
    for left, right in zip(points, points[1:]):
        _la, le, lt = left
        ra, re, rt = right
        span_e = max(1.0, re - le)
        for ayah in ayahs:
            if ayah < left[0] or ayah >= ra:
                continue
            ratio = (expected[ayah] - le) / span_e
            out[ayah] = lt + ratio * (rt - lt)
    for ayah in ayahs:
        out.setdefault(ayah, expected[ayah])
    return out


def _local_tolerance(
    ayahs: list[int],
    expected: dict[int, float],
    idx: int,
    total_ms: float,
) -> float:
    ayah = ayahs[idx]
    prev_ms = expected[ayahs[idx - 1]] if idx > 0 else 0.0
    next_ms = expected[ayahs[idx + 1]] if idx + 1 < len(ayahs) else total_ms
    local_span = max(expected[ayah] - prev_ms, next_ms - expected[ayah], 1.0)
    return max(700.0, min(8_000.0, local_span * 0.55))


def _best_pause(
    pauses: list[candidates.Pause],
    expected_ms: float,
    tolerance_ms: float,
    used: set[int],
) -> int | None:
    best_idx: int | None = None
    best_score = float("inf")
    for idx, pause in enumerate(pauses):
        if idx in used:
            continue
        dist = abs(pause.center_ms - expected_ms)
        if dist > tolerance_ms:
            continue
        depth_bonus = min(pause.depth_ms, 2_000.0) * 0.12
        score = dist - depth_bonus
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx
