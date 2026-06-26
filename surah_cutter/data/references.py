"""Чтение вендоренных эталонных тайм-кодов чтецов (`data/references/*.db`).

Каждая эталонная БД содержит таблицу ``timings(sura, ayah, time)`` (6236 строк,
``time`` — старт аята в мс) и ``db_version(code, version)``. Модуль чистый
(только stdlib) и не тянет GUI/torch.

Соглашение по басмале подтверждено: в суре 1 басмала — это аят (1, 1) со
стартом 100 мс.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

__all__ = [
    "REFERENCES_DIR",
    "list_references",
    "load_reference",
    "validate",
    "find_reciter",
]

# Дефолтный каталог с эталонами: <repo>/data/references.
# references.py -> data/ -> surah_cutter/ -> <repo>/
REFERENCES_DIR = Path(__file__).resolve().parents[2] / "data" / "references"

EXPECTED_TOTAL_AYAHS = 6236
EXPECTED_SURA_COUNT = 114


def _refs_dir(references_dir: str | Path | None = None) -> Path:
    """Каталог с эталонными БД (дефолт — :data:`REFERENCES_DIR`)."""
    return Path(references_dir) if references_dir is not None else REFERENCES_DIR


def list_references(references_dir: str | Path | None = None) -> list[str]:
    """Имена доступных эталонов (чтецов).

    Имя берётся из ``db_version.code``, при недоступности — из имени файла
    (``<name>.db``). Результат отсортирован по алфавиту.
    """
    directory = _refs_dir(references_dir)
    names: list[str] = []
    for path in sorted(directory.glob("*.db")):
        names.append(_reference_name(path))
    return names


def _reference_name(path: Path) -> str:
    """Каноническое имя эталона: ``db_version.code`` или имя файла без ``.db``."""
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT code FROM db_version LIMIT 1").fetchone()
        if row and row[0]:
            return str(row[0])
    except sqlite3.Error:
        pass
    return path.stem


def _resolve_path(name_or_path: str | Path, references_dir: str | Path | None) -> Path:
    """Превратить имя чтеца или путь в существующий путь к ``.db``."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".db" and candidate.exists():
        return candidate

    directory = _refs_dir(references_dir)
    by_filename = directory / f"{Path(name_or_path).stem}.db"
    if by_filename.exists():
        return by_filename

    # Запасной путь: совпадение по db_version.code.
    for path in sorted(directory.glob("*.db")):
        if _reference_name(path) == str(name_or_path):
            return path

    raise FileNotFoundError(
        f"Эталон не найден: {name_or_path!r} (искал в {directory})"
    )


def load_reference(
    name_or_path: str | Path,
    references_dir: str | Path | None = None,
) -> dict[tuple[int, int], int]:
    """Загрузить эталон в отображение ``{(sura, ayah): start_ms}``.

    ``name_or_path`` — имя чтеца (``abdul_basit_mujawwad``), имя файла или путь
    к ``.db``. Бросает :class:`FileNotFoundError`, если эталон отсутствует, и
    :class:`ValueError`, если в БД нет таблицы ``timings``.
    """
    path = _resolve_path(name_or_path, references_dir)
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT sura, ayah, time FROM timings ORDER BY sura, ayah"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError(f"Невалидная эталонная БД {path}: {exc}") from exc

    return {(int(sura), int(ayah)): int(time) for sura, ayah, time in rows}


def validate(
    timings: dict[tuple[int, int], int],
    ayah_counts: dict[int, int] | None = None,
) -> list[str]:
    """Проверить корректность эталонных тайм-кодов; вернуть список проблем.

    Пустой список — всё в порядке. Проверки:

    - ровно :data:`EXPECTED_TOTAL_AYAHS` строк и :data:`EXPECTED_SURA_COUNT` сур;
    - ``time`` монотонно неубывает по аятам внутри каждой суры;
    - число аятов в каждой суре совпадает с ``ayah_counts``.

    ``ayah_counts`` — отображение ``{sura: count}``. Если не передано, счётчики
    лениво берутся из :mod:`surah_cutter.data.quran_text`; при его отсутствии
    проверка по числу аятов в суре пропускается.
    """
    issues: list[str] = []

    if len(timings) != EXPECTED_TOTAL_AYAHS:
        issues.append(
            f"ожидалось {EXPECTED_TOTAL_AYAHS} аятов, получено {len(timings)}"
        )

    suras = sorted({sura for sura, _ in timings})
    if len(suras) != EXPECTED_SURA_COUNT:
        issues.append(
            f"ожидалось {EXPECTED_SURA_COUNT} сур, получено {len(suras)}"
        )

    if ayah_counts is None:
        ayah_counts = _ayah_counts_from_quran_text()

    for sura in suras:
        ayahs = sorted(ayah for s, ayah in timings if s == sura)
        prev_time: int | None = None
        for ayah in ayahs:
            time = timings[(sura, ayah)]
            if prev_time is not None and time < prev_time:
                issues.append(
                    f"сура {sura}: time убывает на аяте {ayah} "
                    f"({time} < {prev_time})"
                )
            prev_time = time

        if ayah_counts is not None:
            expected = ayah_counts.get(sura)
            if expected is not None and len(ayahs) != expected:
                issues.append(
                    f"сура {sura}: ожидалось {expected} аятов, получено {len(ayahs)}"
                )

    return issues


def _ayah_counts_from_quran_text() -> dict[int, int] | None:
    """Ленивая попытка взять число аятов по сурам из ``quran_text``.

    Возвращает ``None``, если модуль ещё не реализован, — чтобы валидация
    оставалась слабо связанной.
    """
    try:
        from surah_cutter.data import quran_text  # noqa: PLC0415
    except ImportError:
        return None
    getter = getattr(quran_text, "surah_ayah_counts", None)
    if getter is None:
        return None
    try:
        return {int(k): int(v) for k, v in dict(getter()).items()}
    except Exception:  # noqa: BLE001 — quran_text не наша зона ответственности
        return None


def find_reciter(
    name: str,
    references_dir: str | Path | None = None,
) -> str | None:
    """Нечёткий поиск эталона по имени; вернуть каноническое имя или ``None``.

    Сопоставление нормализованное: регистр игнорируется, разделители
    (``_ - пробел``) схлопываются. Сначала точное совпадение, затем по
    вхождению подстроки в обе стороны.
    """
    available = list_references(references_dir)
    target = _normalize(name)
    if not target:
        return None

    normalized = {ref: _normalize(ref) for ref in available}

    for ref, norm in normalized.items():
        if norm == target:
            return ref

    matches = [
        ref
        for ref, norm in normalized.items()
        if target in norm or norm in target
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _normalize(value: str) -> str:
    """Нормализовать имя для сравнения: lower-case, разделители убраны."""
    return "".join(ch for ch in value.lower() if ch.isalnum())
