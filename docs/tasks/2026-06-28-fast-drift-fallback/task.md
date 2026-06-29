# Fast drift fallback

> Superseded 2026-06-28 by `docs/tasks/2026-06-28-fast-no-tiled-fallback/task.md`:
> fast drift must stay visible in fast logs/trace and must not trigger automatic
> fallback to tiled.

## Цель

Убрать корневую проблему: fast-разметка, которую сам fast пометил как drifted,
не должна записываться в БД как нормальный `auto` результат.

## Скоуп

- Исторически: `surah_cutter/core/pipeline.py` orchestration fast -> tiled fallback.
- Отменено: summary trace больше не должен показывать fallback как нормальный путь.
- `tests/test_pipeline.py`: unit-тесты без реального audio/model.
- `docs/STATUS.md`: зафиксировать изменение поведения.

Не трогать детектор пауз, локальный matcher и формат БД.

## Прочитанные доки и код

- `AGENTS.md` из сообщения пользователя: правила репозитория.
- Свежий `/Users/aabibulat1/Desktop/Badr_Turki/Badr_Turki.logs/sura_011_fast.json`:
  fast возвращает 123 строки при `drift_detected=True`, `verify_sim=0.407`.
- `surah_cutter/core/pipeline.py`: `_fast_align`, `_auto_align`, `_tiled_align`.
- `surah_cutter/webgui/server.py`: запись `db.write_timings(rows, source="auto")`
  происходит после `process_surah`.

## Решение

- Историческое решение ниже отменено: после `align_surah_fast(...)`
  `trace["drift_detected"]` остаётся диагностикой fast, но не запускает другой
  режим и не подменяет fast rows.

## Чек-лист

- [x] Реализовать fallback.
- [x] Добавить unit-тесты.
- [x] Запустить проверки.
- [x] Обновить STATUS.

## Проверка

- `uv run pytest tests/test_pipeline.py tests/test_align_fast.py` — 26 passed.
- `uv run ruff check surah_cutter/core/pipeline.py surah_cutter/webgui/server.py tests/test_pipeline.py tests/test_align_fast.py` — passed.
- `python3 -m py_compile surah_cutter/core/pipeline.py surah_cutter/webgui/server.py` — passed.
- `node --check surah_cutter/webgui/static/app.js` — passed.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
