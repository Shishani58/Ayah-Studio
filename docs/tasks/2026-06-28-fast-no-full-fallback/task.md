# Fast no full fallback

## Скоуп

Убрать автоматический fallback из fast-разметки на full при низком качестве
чтения. Вместо этого сохранять fast-результат и подсвечивать слабые места через
per-ayah confidence в `review_meta`.

## Доки

- docs/tasks/2026-06-28-fast-log-diagnostics/task.md — прочитан: новые поля trace
  уже содержат диагностику плохого чтения.
- docs/development/task-files.md — прочитан: изменение поведения генерации
  требует task-файл.
- docs/project.md — пропущен: схема БД и основной пайплайн не меняются.

## Решения

- При drift не запускать `align.align_surah`, а вернуть fast rows.
- В trace оставить `drift_detected`, `mean_conf`, `verify_sim`.
- Записывать confidence по каждому auto-аяту из trace в `review_meta.confidence`.

## Чек-лист

- [x] Убрать full fallback из fast.
- [x] Прокинуть per-ayah confidence в project DB.
- [x] Проверить синтаксис/тесты.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py surah_cutter/data/project_db.py surah_cutter/webgui/server.py`
- `uv run ruff check surah_cutter/core/align_fast.py surah_cutter/data/project_db.py surah_cutter/webgui/server.py`
- `uv run pytest tests/test_align_fast.py`

## Блокеры

Нет.

## Docs

Docs unchanged: пользовательские docs не обновляются, изменение локальное для режима fast.
