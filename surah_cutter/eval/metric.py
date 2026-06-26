"""Метрика точности: абсолютная ошибка старта аята (мс) гипотезы против эталона.

Чистая библиотека (stdlib + numpy). ``compare`` считает ошибки по ключам
``(sura, ayah)``, общим для обоих наборов; ``compare_dbs`` подгружает две БД
через :mod:`surah_cutter.data.references` и зовёт ``compare``.

Цели точности проекта: ``median_ms < 150`` и ``p95_ms < 500``; доли попадания в
эти допуски возвращаются как ``within_150ms`` / ``within_500ms``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Union

import numpy as np

__all__ = ["CompareResult", "compare", "compare_dbs"]

# Допуски (мс), соответствующие целям точности проекта.
TOL_TIGHT_MS = 150
TOL_LOOSE_MS = 500

Timings = Union[Mapping[tuple[int, int], int], Iterable[tuple[int, int, int]]]


@dataclass(frozen=True)
class CompareResult:
    """Сводка абсолютной ошибки старта аята (мс).

    Attributes:
        n: Число общих аятов (по ключу ``(sura, ayah)``), на которых считалась ошибка.
        missing: Сколько аятов есть в эталоне, но нет в гипотезе.
        extra: Сколько аятов есть в гипотезе, но нет в эталоне.
        median_ms: Медиана абсолютной ошибки.
        mean_ms: Среднее абсолютной ошибки.
        p95_ms: 95-й перцентиль абсолютной ошибки.
        max_ms: Максимальная абсолютная ошибка.
        within_150ms: Доля аятов с ошибкой ``<= 150`` мс (``0..1``).
        within_500ms: Доля аятов с ошибкой ``<= 500`` мс (``0..1``).
    """

    n: int
    missing: int
    extra: int
    median_ms: float
    mean_ms: float
    p95_ms: float
    max_ms: float
    within_150ms: float
    within_500ms: float


def _as_mapping(timings: Timings) -> dict[tuple[int, int], int]:
    """Привести вход к ``{(sura, ayah): start_ms}``.

    Принимает либо отображение ``{(sura, ayah): time}``, либо итерируемое
    троек ``(sura, ayah, time)``.
    """
    if isinstance(timings, Mapping):
        return {(int(s), int(a)): int(t) for (s, a), t in timings.items()}
    return {(int(s), int(a)): int(t) for s, a, t in timings}


def compare(ref: Timings, hyp: Timings) -> CompareResult:
    """Сравнить гипотезу ``hyp`` с эталоном ``ref`` по старту аятов.

    Ошибка считается как ``abs(hyp - ref)`` (мс) на пересечении ключей
    ``(sura, ayah)``. Отсутствующие в гипотезе аяты идут в ``missing``,
    лишние — в ``extra``; в статистику ошибок они не входят.

    При пустом пересечении статистики ошибок равны ``0.0``.
    """
    ref_map = _as_mapping(ref)
    hyp_map = _as_mapping(hyp)

    common = ref_map.keys() & hyp_map.keys()
    missing = len(ref_map.keys() - hyp_map.keys())
    extra = len(hyp_map.keys() - ref_map.keys())

    if not common:
        return CompareResult(
            n=0,
            missing=missing,
            extra=extra,
            median_ms=0.0,
            mean_ms=0.0,
            p95_ms=0.0,
            max_ms=0.0,
            within_150ms=0.0,
            within_500ms=0.0,
        )

    errors = np.array(
        [abs(hyp_map[key] - ref_map[key]) for key in common], dtype=np.float64
    )

    return CompareResult(
        n=int(errors.size),
        missing=missing,
        extra=extra,
        median_ms=float(np.median(errors)),
        mean_ms=float(np.mean(errors)),
        p95_ms=float(np.percentile(errors, 95)),
        max_ms=float(np.max(errors)),
        within_150ms=float(np.mean(errors <= TOL_TIGHT_MS)),
        within_500ms=float(np.mean(errors <= TOL_LOOSE_MS)),
    )


def compare_dbs(ref_db_path: str, hyp_db_path: str) -> CompareResult:
    """Загрузить две БД тайм-кодов и сравнить их через :func:`compare`.

    ``ref_db_path`` / ``hyp_db_path`` — имена чтецов, имена файлов или пути к
    ``.db`` (см. :func:`surah_cutter.data.references.load_reference`).
    """
    from surah_cutter.data.references import load_reference  # noqa: PLC0415

    ref = load_reference(ref_db_path)
    hyp = load_reference(hyp_db_path)
    return compare(ref, hyp)
