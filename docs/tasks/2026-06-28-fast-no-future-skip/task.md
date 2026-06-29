# Fast no future skip

## Цель

Исправить нарушение пользовательского алгоритма: если после паузы слышен будущий
аят, matcher не должен перескакивать к нему, пока текущий аят не найден или явно
не признан wasl.

## Скоуп

- `surah_cutter/core/align_fast.py`: `_sequential_pause_match`.
- `tests/test_align_fast.py`: обновить регрессии под новое правило.

Не менять БД, UI и детектор пауз.

## Прочитанные доки и код

- `AGENTS.md` из сообщения пользователя.
- Свежий trace суры 11 Badr Turki: `future_anchor` перескакивает через
  текущие аяты, а потом восстанавливает цепочку постфактум.
- `surah_cutter/core/align_fast.py`: ветка `future_gap`.
- `tests/test_align_fast.py`: старые тесты закрепляли поведение с
  `future_anchor`.

## Решение

- При `future_gap` не принимать будущий аят.
- Сначала попытаться восстановить текущий аят из накопленных ранних/ожидающих
  пауз.
- Если текущий не найден — пометить именно его как `wasl`, но будущую паузу
  оставить на следующий шаг.
- Прогноз `expected_ms` считать от последнего найденного якоря, а не только от
  статического prior.
- Ввести отдельный `SC_FAST_CONFIRM_MATCH_SIM` (default `0.50`): слабый current
  match больше не считается найденным аятом и не может вернуться через recovery.

## Чек-лист

- [x] Изменить `_sequential_pause_match`.
- [x] Обновить тесты.
- [x] Запустить проверки.

## Проверка

- `uv run pytest tests/test_pipeline.py tests/test_align_fast.py` — 27 passed.
- `uv run ruff check surah_cutter/core/align_fast.py surah_cutter/core/pipeline.py surah_cutter/webgui/server.py tests/test_align_fast.py tests/test_pipeline.py` — passed.
- `python3 -m py_compile surah_cutter/core/align_fast.py surah_cutter/core/pipeline.py surah_cutter/webgui/server.py` — passed.
- `node --check surah_cutter/webgui/static/app.js` — passed.
- Fast-only практический прогон без записи:
  `/tmp/sura_011_fast_no_future_skip_confirm50.json`. Аяты 4 и 12 больше не
  `confirmed` по слабой паузе, а `wasl`; аят 19 остаётся на 377.34s, 26 на
  482.92s. Сура всё ещё `drift=True`, поэтому write-path fallback остаётся нужен.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
