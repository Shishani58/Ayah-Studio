# Internal repeat text match

## Цель

Усилить fast-анализ текста перед чтением пауз:

- учитывать не только одинаковые начала соседних аятов, но и повтор начала
  целевого аята внутри предыдущего/текущего текста;
- не принимать внутреннюю паузу за начало следующего аята только из-за высокого
  fuzzy-score;
- сделать сравнение decode↔text более привязанным к началу фразы.

## Скоуп

- `surah_cutter/core/align_fast.py`: text-aware анализ, similarity, reject guard.
- `tests/test_align_fast.py`: pure-регрессии.
- `docs/STATUS.md`: краткая запись о поведении.

Формат БД, UI и ручные тайминги не менять.

## Прочитанные доки и код

- `docs/project.md`: инварианты forced alignment и БД.
- `docs/STATUS.md`: текущее описание fast/pause.
- `surah_cutter/core/align_fast.py`: `_sim`, `_opening_shared_prefix_chars`,
  `prefix_mismatch`, `reject_match`.
- `tests/test_align_fast.py`: текущие pure-тесты.

## Решения

- `_sim` оставить fuzzy, но сделать score prefix-aware: общий префикс decode и
  opening влияет на итог, поэтому строка с хвостом предыдущего аята хуже, чем
  настоящий старт.
- Добавить анализ внутренних повторов: если opening целевого аята встречается
  внутри предыдущих локальных аятов, расширять окно/длину сравнения так же, как
  для одинаковых начал.
- Для таких рискованных мест добавить `internal_repeat_mismatch`: если decode
  похож на текст вокруг внутреннего повтора лучше, чем на реальное начало аята,
  пауза не принимается как граница.

## Чек-лист

- [x] Добавить helper внутренних повторов.
- [x] Подключить к window/sim sizing.
- [x] Усилить reject guard.
- [x] Добавить тесты.
- [x] Запустить проверки.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py`
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py`
- `uv run pytest tests/test_align_fast.py` — 36 passed.

Docs updated: `docs/STATUS.md`,
`docs/tasks/2026-06-30-internal-repeat-text-match/task.md`.

## Блокеры

Нет.
