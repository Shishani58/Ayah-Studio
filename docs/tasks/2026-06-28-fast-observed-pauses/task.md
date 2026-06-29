# Fast observed pauses

> Superseded 2026-06-28 by `docs/tasks/2026-06-28-fast-no-tiled-fallback/task.md`:
> `drift=True` remains a fast diagnostic and must not route the write path to
> tiled.

## Цель

Использовать каждую прочитанную паузу как наблюдение: если декод показал, какой
аят слышен после паузы, не выбрасывать это знание при движении по последовательности.

## Скоуп

- `surah_cutter/core/align_fast.py`: `_sequential_pause_match`.
- `tests/test_align_fast.py`: регрессии только при изменении поведения.

Не менять БД, формат выходных данных и ручные тайминги.

## Прочитанные доки и код

- `AGENTS.md` из сообщения пользователя: правила проекта и task-файлы.
- `docs/tasks/2026-06-28-fast-no-future-skip/task.md`: предыдущая правка
  запретила перескок на будущий аят, но ещё не сохраняла все наблюдения.
- `surah_cutter/core/align_fast.py`: `_sequential_pause_match` и trace events.

## Решение

- Кэшировать лучший распознанный boundary для каждой паузы.
- Добавлять `observed_boundary` и `observed_sim` в trace events для ранних,
  лишних, слабых и будущих пауз.
- Использовать глобальное наблюдение паузы при классификации, а не только
  локальный lookahead от текущего аята.
- Наблюдения позади текущего boundary явно помечать как `stale_observation`,
  чтобы они не могли увести matcher назад или в левый аят.
- Текущий аят с `min_sim`/reject-pass, но ниже `confirm_min_sim`, не принимать
  сразу: сохранить как `observed_current_candidate`, дочитать окно и либо
  восстановить его перед `future_gap`, либо принять лучший current-кандидат.
- В recovery ослабить влияние старого prior: среди кандидатов текущего аята
  выбирать преимущественно по услышанному `sim`, а не по близости к прогнозу.

## Чек-лист

- [x] Исправить `_sequential_pause_match`.
- [x] Запустить тесты и линтер.
- [x] Проверить практический прогон суры 11 без записи БД.

## Проверка

- `uv run pytest tests/test_align_fast.py` — 26 passed.
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py` — passed.
- `python3 -m py_compile surah_cutter/core/align_fast.py` — passed.
- Fast-only практический прогон без записи:
  `/tmp/sura_011_fast_recovery_content_first.json`.
  - Аят 4: pause 52.75s, row 53.33s, `match_action=recovered`.
  - Аят 12: pause 205.48s, row 205.29s, `match_action=recovered`.
  - Аят 19: 377.34s, `confirmed`.
  - Аят 20: recovery выбрал более сильный current-кандидат 418.86s
    (`sim=0.485`) вместо раннего 393.15s (`sim=0.438`).
  - Аят 26: row 482.92s остаётся wasl/interpolation, fast trace всё ещё
    `drift=True`; по актуальной политике это остаётся диагностикой fast, без
    fallback в другой режим.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Fast-only для всей суры 11 всё ещё drifted; это остаётся проблемой алгоритма
fast.
