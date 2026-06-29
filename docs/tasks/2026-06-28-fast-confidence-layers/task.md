# Fast confidence-layer pause search

## Цель

Переделать fast-алгоритм под слоистый поиск пауз:
сначала все паузы классифицируются по достоверности, основной проход идёт по
достоверным, а менее достоверные подключаются только когда matcher доказал
перескок через аят.

## Скоуп

- `surah_cutter/core/align_fast.py`: слои пауз и recovery через менее
  достоверные паузы.
- `surah_cutter/webgui/server.py`: лог должен показывать слои/резерв.
- `tests/test_align_fast.py`: модельно-независимая проверка helper-логики, если
  потребуется.
- `docs/STATUS.md`: кратко зафиксировать актуальный алгоритм.
- `docs/initiatives.md`: журнал пользовательских инициатив.

## Доки

- `docs/STATUS.md` — прочитан ранее в этой сессии: актуальное состояние fast.
- `docs/tasks/2026-06-28-fast-sequential-pauses/task.md` — прочитан ранее:
  текущий sequential matching.
- `docs/project.md` — не перечитывался: формат БД не меняется.

## Решения

- Не делать все паузы равноправными: rejected/weak паузы слишком шумные.
- Первый проход использует только trusted pauses.
- При `future_gap` добавлять в активный список только менее достоверные паузы
  внутри пропущенного интервала.
- После добавления резервных пауз перезапускать sequential matching.
- Логировать `trusted/weak/rejected/activated_low_conf`.

## Чек-лист

- [x] Создать журнал инициатив.
- [x] Разделить pause candidates на слои.
- [x] Подключить low-confidence recovery после перескока.
- [x] Расширить trace/log.
- [x] Проверить ruff/pytest.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py surah_cutter/webgui/server.py`
- `uv run ruff check surah_cutter/core/align_fast.py surah_cutter/webgui/server.py`
- `uv run pytest tests/test_align_fast.py tests/test_candidates.py` — 31 passed.
- `uv run pytest tests/test_pipeline.py` — 2 passed.

## Docs

Docs updated: `docs/STATUS.md`, `docs/initiatives.md`, этот task-файл.

## Блокеры

Нет.
