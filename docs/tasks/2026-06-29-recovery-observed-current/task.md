# Recovery observed current

## Цель

Исправить fast recovery: если при восстановлении пропущенного аята пауза уже
прочитана как другой аят, не использовать её для текущего аята только из-за
prior/refine. Для суры 11 Badr Turki аят 13 должен уходить на раннюю паузу,
где greedy-decode слышит начало 13-го, а не на позднюю паузу с чужим observed.

## Скоуп

- `surah_cutter/core/align_fast.py`: `_sequential_pause_match`, recovery выбор.
- `tests/test_align_fast.py`: pure-регрессия на early observed current против
  поздней observed-wrong паузы.

Не менять формат БД, UI, детектор пауз.

## Прочитанные доки и код

- `AGENTS.md` из сообщения пользователя.
- `/Users/aabibulat1/Desktop/Badr_Turki/Badr_Turki.logs/sura_011_fast.json`:
  свежий trace суры 11, аят 13.
- `surah_cutter/core/align_fast.py`: `_sequential_pause_match`,
  `observe_pause`, `recover_one`.
- `tests/test_align_fast.py`: существующие регрессии matcher.

## Решение

- В `recover_one` пропускать кандидата, если `observe_pause(j)` указывает на
  другой boundary и его similarity выше текущего similarity.
- Ранняя пауза, попавшая в `early_pool`, остаётся доступной для recovery.
- Если `observe_pause(j)` указывает на текущий boundary, recovery использует это
  observed similarity как доказательство и не отбрасывает паузу только из-за
  hard-reject от prior/prefix guard.

## Чек-лист

- [x] Добавить регрессию.
- [x] Исправить recovery.
- [x] Запустить тесты/линтер.
- [x] Проверить практический прогон.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py` — passed.
- `uv run pytest tests/test_align_fast.py` — 29 passed.
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py` —
  passed.
- Практический fast-прогон без записи в БД:
  `/tmp/sura_011_fast_recovery_observed_current.json`.
  - Аят 13: стал `234.640s`, пауза `234.40-234.88s`, `sim=0.529`,
    `match_action=recovered`.
  - Поздняя пауза `244.670s` больше не выбрана: событие
    `skip_recovery_observed_mismatch`, потому что для текущего аята `sim=0.378`,
    а observed указывал на другой аят с `0.486`.
  - Сура 11 всё ещё `drift=True`, `mean_conf=0.397`, `verify_sim=0.326`.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
