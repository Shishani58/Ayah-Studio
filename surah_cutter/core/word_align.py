"""Strict local word alignment; never substitutes guessed/uniform timings."""

from __future__ import annotations

import numpy as np

from surah_cutter.core import align
from surah_cutter.core._ctc import forced_align, aligned_segments
from surah_cutter.data import transliteration
from surah_cutter.data.quran_text import Word


def align_ayah_words(pcm, sr: int, words: list[Word]) -> list[tuple[int, int, float]]:
    """Return (surah word index, local start ms, confidence) for an ayah clip.

    The caller clips at reviewed ayah boundaries. No intro basmala is inserted:
    the clip already excludes it (except Fatiha 1:1, whose words include it).
    """
    if sr != align.EXPECTED_SR or not words or len(pcm) < sr // 10:
        raise ValueError("Недостаточно аудио для пословной разметки.")
    if len({(w.surah, w.ayah) for w in words}) != 1:
        raise ValueError("Ожидаются слова одного аята.")
    bundle = align._get_onnx_bundle()
    if getattr(bundle, "text_kind", "phonetic") in ("whisper", "mfa"):
        return bundle.emitter.align_words(pcm, words)
    if (hasattr(getattr(bundle, "emitter", None), "build_targets")
            or getattr(bundle, "text_kind", "phonetic") == "arabic"):
        word_trs = []
    else:
        trs = transliteration.surah_word_tr(words[0].surah)
        if any(w.index >= len(trs) for w in words):
            raise ValueError("Нет полной транслитерации аята.")
        word_trs = [trs[w.index] for w in words]
    targets, word_spans = align._build_targets(words, word_trs, bundle)
    if not targets or any(end <= start for start, end in word_spans):
        raise ValueError("Не все слова представлены в словаре модели.")
    if align._full_ctc_requires_tiling(len(pcm), sr, len(targets)):
        raise ValueError("Аят слишком длинный для локального расчёта. Уточните его границы.")
    emissions = align._emissions(bundle, pcm, np)
    if getattr(bundle, "text_kind", "").startswith("fastconformer"):
        targets, word_spans = align._build_targets(words, word_trs, bundle)
    path, scores = forced_align(
        np.ascontiguousarray(emissions[None], dtype=np.float32),
        np.asarray(targets, dtype=np.int64)[None], blank=bundle.blank_id,
    )
    try:
        segments = aligned_segments(path, targets, blank=bundle.blank_id)
    except ValueError as exc:
        raise ValueError("Не удалось сопоставить все слова. Проверьте границы аята и запись.") from exc
    stride = align._stride_ms(len(pcm), sr, len(emissions))
    scores = scores.reshape(-1)
    rows = []
    previous = -1
    for word, (start, end) in zip(words, word_spans):
        first, last = segments[start], segments[end - 1]
        ms = max(0, round(first.start * stride - getattr(bundle, "lag_ms", align.ALIGN_LAG_MS)))
        if ms <= previous or ms >= len(pcm) * 1000 / sr:
            raise ValueError("Границы слов не различимы. Проверьте аят на слух.")
        rows.append((word.index, ms, align._span_confidence(scores, first.start, last.end)))
        previous = ms
    return rows
