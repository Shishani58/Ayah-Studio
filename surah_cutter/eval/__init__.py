"""Метрики точности пер-аятных тайм-кодов (Phase-2 harness).

Чистая библиотека (stdlib + numpy). Без GUI/torch. Сравнивает гипотезу с
эталоном по абсолютной ошибке старта аята в мс.
"""

from __future__ import annotations

from surah_cutter.eval.metric import CompareResult, compare, compare_dbs

__all__ = ["CompareResult", "compare", "compare_dbs"]
