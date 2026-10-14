"""Public FAST entrypoint: shared sequential CTC/Tarteel executor."""

from __future__ import annotations

from typing import Any, Callable


def align_surah_fast(
    pcm,
    sr: int,
    words: list,
    *,
    surah: int | None = None,
    bundle: Any = None,
    trace: dict | None = None,
    log_event: Callable[[str], None] | None = None,
    ayah_audio_dir: str | None = None,
    ayah_progress: Callable[[int], None] | None = None,
    pause_decisions: list | None = None,
) -> list[tuple[int, int, float]]:
    """Run the shared sequential FAST executor for CTC and Tarteel."""
    from surah_cutter.core.fast_executor import align_surah_sequential

    if ayah_audio_dir:
        raise ValueError("Единый FAST не поддерживает внешние аятные клипы. Выберите режим fingerprint.")
    return align_surah_sequential(
        pcm, sr, words, surah=surah, bundle=bundle, trace=trace,
        log_event=log_event, ayah_progress=ayah_progress, pause_decisions=pause_decisions,
    )
