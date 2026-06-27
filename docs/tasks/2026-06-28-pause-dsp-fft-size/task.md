# Задача: исправить падение DSP-метрики пауз

## Цель

При расчёте суры 4 в web UI worker падает с ошибкой:
`operands could not be broadcast together with shapes (1792,) (2048,)`.
Причина — `_dsp_pause_metrics()` выбирает `n_fft=2048` для сегмента короче
2048 samples и затем умножает короткий срез на окно Hann длиной 2048.

## Скоуп

- `surah_cutter/core/candidates.py`: выбрать FFT-размер не длиннее сегмента.
- `tests/test_candidates.py`: регрессия на неполную длину DSP-сегмента.
- Без изменения БД, GUI-контракта и текста Корана.

## Прочитанные доки

- `docs/STATUS.md`: текущий pause pipeline.
- `docs/tasks/2026-06-28-pause-edge-refine/task.md`: последняя правка DSP/edge
  pause path.

## Чек-лист

- [x] Исправить `n_fft`.
- [x] Добавить регрессионный тест.
- [x] Проверить тестами и реальным pause pipeline для суры 4.
- [x] Обновить docs/status.

## Проверка

`.venv/bin/pytest tests/test_candidates.py tests/test_align_fast.py` — 16 passed.
`.venv/bin/ruff check surah_cutter/core/candidates.py tests/test_candidates.py` —
passed.

Реальный проект из settings:
`/Users/aabibulat1/Desktop/Badr_Turki/004.mp3` →
`pause_pipeline_decisions(..., surah=4)` — 681 decisions, 655 accepted, без
исключения `operands could not be broadcast together`.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
