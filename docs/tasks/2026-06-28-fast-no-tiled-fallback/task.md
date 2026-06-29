# Fast no tiled fallback

## Цель

Отключить автоматический `tiled` fallback после `mode='fast'`: если fast
детектирует drift, это проблема fast-алгоритма и диагностики, а не повод
подменять результат другим режимом.

## Скоуп

- `surah_cutter/core/pipeline.py`: убрать fast -> tiled fallback.
- `surah_cutter/webgui/*`: не показывать fast drift как переход в tiled.
- `tests/test_pipeline.py`: обновить ожидания.
- `docs/STATUS.md`: зафиксировать решение.

## Доки

- `docs/STATUS.md` — прочитан: содержит текущую ошибочную политику write-path
  guard через tiled.
- `docs/tasks/2026-06-28-fast-drift-fallback/task.md` — прочитан через поиск:
  описывает старое решение, которое сейчас отменяется.
- `docs/project.md` — не перечитывался: формат БД и архитектура хранения не
  меняются.

## Решения

- Fast всегда возвращает свои rows.
- `drift_detected=True` остаётся в trace/log/UI как сигнал слабого места.
- Env `SC_FAST_DRIFT_FALLBACK` удалить из runtime-поведения.
- `tiled` не использовать как лечение fast: его можно оставить как отдельный
  ручной режим для уже существующих сценариев, но не как fallback fast.

## Чек-лист

- [x] Убрать fallback из pipeline.
- [x] Обновить UI summary.
- [x] Обновить тесты.
- [x] Обновить docs/STATUS.md.
- [x] Проверить ruff/pytest.

## Проверка

- `python3 -m py_compile surah_cutter/core/pipeline.py surah_cutter/webgui/server.py`
- `uv run ruff check surah_cutter/core/pipeline.py surah_cutter/webgui/server.py tests/test_pipeline.py`
- `uv run pytest tests/test_pipeline.py` — 2 passed.
- `node --check surah_cutter/webgui/static/app.js`

## Docs

Docs updated: `docs/STATUS.md`,
`docs/tasks/2026-06-28-fast-no-tiled-fallback/task.md`,
`docs/tasks/2026-06-28-fast-drift-fallback/task.md`,
`docs/tasks/2026-06-28-fast-observed-pauses/task.md`.

## Блокеры

Нет.
