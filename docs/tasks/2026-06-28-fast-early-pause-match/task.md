# Fast early pause match

## Цель

Уменьшить каскадные ошибки fast-разметки на суре 11 Badr Turki, где правильные
паузы видны в trace, но matcher выбирает более поздние/худшие паузы из-за
жёсткого prior-окна и фильтра `weak-dsp`.

## Скоуп

- `surah_cutter/core/align_fast.py`: только логика сопоставления пауз.
- `tests/test_align_fast.py`: модель-free регрессии для `_sequential_pause_match`.
- Не менять формат БД, UI и детектор пауз.

## Прочитанные доки и код

- `AGENTS.md` из сообщения пользователя: правила репозитория.
- `/Users/aabibulat1/Desktop/Badr_Turki/Badr_Turki.logs/sura_011_fast.json`:
  trace суры 11.
- `surah_cutter/core/align_fast.py`: `_sequential_pause_match`,
  `reject_match`, `weak_pause`, `far_weak_match`.
- `tests/test_align_fast.py`: существующие unit-тесты pure helpers.

## Наблюдения

- Аят 19: правильная сильная пауза №64 (`377.34с`) имеет sim `0.595`, но ушла в
  `early_pool`, потому что была примерно на 232 мс раньше нижней границы
  `prior_window`.
- Аят 26: пауза №82 (`482.91с`) имеет sim `0.500`, лучше выбранной №85
  (`503.15с`, sim `0.467`), но была отфильтрована как `weak-dsp` с плохим
  tempo-score.

## Решение

- Добавить небольшой grace к prior-окну сопоставления, чтобы near-early паузы
  оценивались акустически, а не выбрасывались до проверки.
- Не применять tempo-reject к `weak-dsp`, если sim уже достигает уровня
  `_PAUSE_TRUST_SIM`.

## Чек-лист

- [x] Добавить регрессии.
- [x] Внести точечную правку matcher.
- [x] Запустить unit-тесты/линтер по затронутым файлам.
- [x] Зафиксировать проверку и docs status.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py` — passed.
- `uv run pytest tests/test_align_fast.py` — 24 passed.
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py` —
  passed.
- Практический прогон без записи в БД:
  `process_surah('/Users/aabibulat1/Desktop/Badr_Turki/011.mp3', 11, mode='fast')`.
  Результат сохранён во временный `/tmp/sura_011_fast_after_fix.json`.
  Аят 19: `377.34s` вместо прежних `385.59s`; аят 26: `482.92s` вместо прежних
  `503.53s`. Сура всё ещё `drift=True`, `mean_conf=0.408`, `verify_sim=0.407`;
  аят 4 остаётся ошибочным.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
