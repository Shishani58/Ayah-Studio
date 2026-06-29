# Local future and micro shelf

## Цель

Исправить две причины неверных fast-меток на суре 11 Badr Turki:

- `future_gap` не должен срабатывать от далёкого совпадения с аятом вне
  локального lookahead.
- `micro-tempo` должен расширять найденную RMS-долину до тихой полки, а не
  оставлять сверхузкий seed.

## Скоуп

- `surah_cutter/core/align_fast.py`: `_sequential_pause_match`,
  `_find_micro_pause_valley`, trace micro-pauses.
- `tests/test_align_fast.py`: pure-регрессии.

Не менять формат БД, UI и ручные тайминги.

## Прочитанные доки и код

- `AGENTS.md` из сообщения пользователя.
- `/Users/aabibulat1/Desktop/Badr_Turki/Badr_Turki.logs/sura_011_fast.json`:
  свежий trace для аятов 12 и 18.
- `surah_cutter/core/align_fast.py`: global observed, future-gap, micro-scan.
- `tests/test_align_fast.py`: текущие helper/matcher тесты.

## Решение

- Разделить локальное observed для решений и глобальное observed для диагностики.
- `label_pause`/`future_gap` используют только `i..i+lookahead`; далёкие похожие
  аяты больше не останавливают текущий поиск.
- `_find_micro_pause_valley` расширяет seed долину до quiet shelf с отдельным
  лимитом ширины и возвращает seed/expanded diagnostics.

## Чек-лист

- [x] Добавить тесты.
- [x] Исправить matcher.
- [x] Исправить micro-shelf.
- [x] Запустить проверки.
- [x] Проверить практический прогон.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py`
- `uv run pytest tests/test_align_fast.py` — 30 passed.
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py`
- Практический прогон без записи в БД на
  `/Users/aabibulat1/Desktop/Badr_Turki/011.mp3`: trace сохранён в
  `/tmp/sura_011_fast_local_future_micro_shelf.json`.

Результат практического прогона:

- аят 12 остался на `205.479s`, но micro-pause расширена с seed
  `205.389-205.569s` до quiet shelf `205.029-205.929s`;
- аят 13 остался на исправленной ранней паузе `234.640s`;
- аят 18 перешёл с wasl `349.786s` на подтверждённую паузу `352.220s`;
- далёкие global observed совпадения больше не дают `future_gap`, а пишутся как
  диагностика `observed_global_boundary`.

Docs updated: `docs/STATUS.md`.

## Блокеры

Fast-прогон суры 11 всё ещё помечается `drift_detected=True`
(`mean_conf=0.403`, `verify_sim=0.331`). Это общий диагностический сигнал по
суре, не блокер для текущего исправления.
