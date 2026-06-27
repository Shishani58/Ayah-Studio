# Задача: ставить черновой старт в середину паузы

## Цель

Черновая разметка по паузам не должна ставить начало аята в самый конец паузы:
так воспроизведение стартует резко, прямо на атаке следующей фразы. Для
pause-backed меток нужна середина паузы.

## Скоуп

- `surah_cutter/core/draft.py`: выбирать и писать pause-backed старт по
  `pause.center_ms`, включая вступительную басмалу.
- `tests/test_draft.py`: зафиксировать поведение регрессионным тестом.
- `surah_cutter/core/candidates.py` и `docs/STATUS.md`: записать правило, что
  пауза как старт аята означает центр, а не `end_ms`.

## Прочитанные доки

- `docs/project.md`: инварианты тайм-кодов, формат БД, forced alignment.
- `docs/STATUS.md`: текущее состояние fast/pause pipeline и web review.
- `docs/tasks/2026-06-28-pause-edge-refine/task.md`: предыдущий фикс краёв пауз.

## Решения

- Не менять детектор пауз и формат данных.
- В черновой ручной разметке считать `center_ms` канонической pause-backed
  меткой старта аята.
- Ранжировать ближайшую паузу к ожидаемой позиции тоже по `center_ms`, чтобы
  выбор и запись использовали одну точку.

## Чек-лист

- [x] Реализовать переход draft-разметки с `end_ms` на `center_ms`.
- [x] Добавить тест.
- [x] Запустить релевантные проверки.
- [x] Зафиксировать docs/status.

## Проверка

`.venv/bin/pytest tests/test_draft.py tests/test_candidates.py` — 7 passed.

`.venv/bin/ruff check surah_cutter/core/draft.py surah_cutter/core/candidates.py tests/test_draft.py tests/test_candidates.py`
— passed.

Docs updated: `docs/STATUS.md`, `surah_cutter/core/candidates.py`, этот task-файл.

## Блокеры

Нет.
