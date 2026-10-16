"""Empirical ayah-position prior derived from bundled reference timings.

The structural text prior is useful but approximate: recited duration is not
strictly proportional to characters. The bundled reference DBs give a cheap,
Quran-specific prior for where each ayah usually starts inside a sura. This
module exposes that prior as median start fractions in ``[0, 1)``.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from statistics import median

from surah_cutter.data.references import REFERENCES_DIR

__all__ = ["surah_start_fractions"]


def surah_start_fractions(
    surah: int,
    *,
    references_dir: str | Path | None = None,
    min_references: int = 3,
) -> dict[int, float]:
    """Return median ``ayah -> start_fraction`` for ``surah``.

    Fractions are measured as ``ayah_start_ms / estimated_sura_duration_ms`` per
    reference, then median-combined across references. Reference DBs store ayah
    starts but not file ends, so the duration is estimated as ``last_start`` plus
    the median positive inter-ayah gap for that sura. If too few references are
    usable, returns ``{}`` so callers can fall back to their existing prior.
    """
    directory = Path(references_dir) if references_dir is not None else REFERENCES_DIR
    return dict(_surah_start_fractions_cached(int(surah), directory, int(min_references)))


@lru_cache(maxsize=256)
def _surah_start_fractions_cached(
    surah: int,
    references_dir: Path,
    min_references: int,
) -> tuple[tuple[int, float], ...]:
    samples: dict[int, list[float]] = {}
    usable_refs = 0

    for path in sorted(references_dir.glob("*.db")):
        rows = _load_sura_rows(path, surah)
        if len(rows) < 2:
            continue

        times = [time for _ayah, time in rows]
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        if not gaps:
            continue

        estimated_duration = times[-1] + median(gaps)
        if estimated_duration <= 0:
            continue

        usable_refs += 1
        for ayah, time in rows:
            frac = max(0.0, min(0.999, float(time) / float(estimated_duration)))
            samples.setdefault(ayah, []).append(frac)

    if usable_refs < min_references:
        return ()

    out: list[tuple[int, float]] = []
    prev = 0.0
    for ayah in sorted(samples):
        vals = samples[ayah]
        if len(vals) < min_references:
            continue
        frac = max(prev, float(median(vals)))
        out.append((ayah, frac))
        prev = frac
    return tuple(out)


def _load_sura_rows(path: Path, surah: int) -> list[tuple[int, int]]:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT ayah, time FROM timings WHERE sura = ? ORDER BY ayah",
                (surah,),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [(int(ayah), int(time)) for ayah, time in rows]
