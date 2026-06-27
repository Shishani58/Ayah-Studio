# Tiled: кэш эмиссий и empirical prior

## Цель

Ускорить и стабилизировать `mode='tiled'` без смены модели и формата данных:
- не пересчитывать ONNX emissions на перекрывающихся chunk-окнах;
- использовать empirical prior из `data/references/*.db` для оценки положения
  аятов внутри суры.

Тесты по указанию пользователя не запускать.

## Скоуп

- `surah_cutter/core/align_tiled.py`
- новый малый helper в `surah_cutter/data/`
- документация задачи и релевантный статус

Без изменений схемы БД, UI и ручной разметки.

## Прочитанные доки

- `docs/project.md` — инварианты пайплайна и формата.
- `docs/STATUS.md` — текущее состояние fast/tiled.
- `docs/tasks/2026-06-27-tiled-align/task.md` — известная проблема повторного
  расчёта emissions.

## Решения

- Кэшировать emissions на всю суру внутри `align_surah_tiled()` и резать их по
  frame-индексам для каждого chunk. Forced alignment остаётся тем же CTC Viterbi.
- Empirical prior строить лениво из эталонных БД: для каждого `(sura, ayah)`
  медианная доля `time / last_time_of_sura`, с fallback на текущий text-unit
  prior при нехватке данных.
- Fallback full сохранить для вырожденных окон.

## Чек-лист

- [x] Добавить helper empirical prior.
- [x] Подключить prior в tiled window sizing.
- [x] Добавить full-emissions cache в tiled.
- [x] Проверить без запуска тестов.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_tiled.py surah_cutter/data/reference_prior.py`
  — OK.
- Тесты/бенчмарки не запускались по указанию пользователя.

Docs updated: `docs/STATUS.md`,
`docs/tasks/2026-06-27-tiled-cache-prior/task.md`.

## Блокеры

Нет.
