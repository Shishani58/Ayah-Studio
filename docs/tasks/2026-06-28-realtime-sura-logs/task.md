# Realtime sura logs

## Цель

Показывать во время расчёта суры понятный журнал пауз и нейро-чтений, чтобы было
видно, где алгоритм берёт паузу, что читает в окне и какой аят детектит.
Сохранять этот журнал для последующего анализа.

## Скоуп

- Fast/auto генерация через web GUI и SSE.
- Сохранение логов рядом с проектной БД без изменения схемы таймингов.
- Не трогать ручные `edited`/`verified` метки.

## Прочитанные доки

- `docs/project.md`: меняется пайплайн генерации и инварианты БД.
- `docs/STATUS.md`: текущие режимы web GUI, fast/tiled/auto и существующий
  debug-log.

## Решения

- Не менять формат данных: лог сохраняется отдельными файлами.
- Использовать уже существующую fast trace как источник деталей.
- Для realtime добавить callback из pipeline в aligner и прокинуть строки в SSE.
- Для web GUI сохранять файлы рядом с проектной БД:
  `<project>.logs/sura_NNN_<mode>.log` и `.json`.

## Чек-лист

- [x] Добавить realtime callback в pipeline/fast aligner.
- [x] Стримить строки в web GUI.
- [x] Сохранять `.log` и `.json` с трассой.
- [x] Обновить статус/доки.
- [x] Проверить линтер/тесты по затронутым файлам.

## Проверка

- `python3 -m py_compile surah_cutter/core/pipeline.py surah_cutter/core/align_fast.py surah_cutter/core/align_tiled.py surah_cutter/webgui/server.py`
- `uv run ruff check surah_cutter/core/pipeline.py surah_cutter/core/align_fast.py surah_cutter/core/align_tiled.py surah_cutter/webgui/server.py`
- `uv run pytest tests/test_align_fast.py tests/test_candidates.py`
- `node --check surah_cutter/webgui/static/app.js`
- `uv run python - <<'PY' ... from surah_cutter.webgui.server import _sse ...`

Docs updated: `docs/STATUS.md`, этот task-файл.
