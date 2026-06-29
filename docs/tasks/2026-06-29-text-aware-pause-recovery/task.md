# Text-aware pause recovery

## Цель

Усилить fast-поиск границ:

- до декодирования анализировать начала аятов и давать больше аудио тем, у кого
  одинаковый текстовый префикс;
- при последовательном поиске по паузам явно восстанавливаться, если текущая
  пауза звучит слишком рано или слишком поздно относительно ожидаемого аята.

## Скоуп

- `surah_cutter/core/align_fast.py`: text-aware decode windows и
  `_sequential_pause_match`.
- `tests/test_align_fast.py`: модельно-независимые регрессии.
- `docs/STATUS.md`: кратко зафиксировать изменение fast-режима.

Формат БД, UI и ручную разметку не менять.

## Прочитанные доки и код

- `docs/project.md`: инварианты пайплайна и БД.
- `docs/STATUS.md`: текущая логика fast/pause.
- `docs/tasks/2026-06-28-fast-sequential-pauses/task.md`: sequential matching,
  weak-pass и micro-pause scan.
- `docs/tasks/2026-06-29-local-future-micro-shelf/task.md`: локальный lookahead
  и micro-shelf.
- `surah_cutter/core/align_fast.py`: decode windows, `_ambiguous_openings`,
  `_sequential_pause_match`.
- `tests/test_align_fast.py`: pure helper/matcher тесты.

## Решения

- Считать длину общего collapsed-префикса соседних openings до greedy-decode.
  Если общий префикс длинный, расширять окно пропорционально этому префиксу:
  одинаковые начала требуют услышать различающее слово, а не только общий старт.
- Для расширенного чтения писать причину `text-aware-opening` и метрики
  `shared_prefix_chars`/`decode_window_ms`/`sim_chars`, чтобы в trace было видно,
  почему окно длиннее.
- В sequential matcher при `heard_i < k` не просто логировать stale-паузу, а
  при наличии текущего кандидата подтвердить его и продолжить вперёд.
- При `heard_i > k` оставить прежний recovery назад через накопленные паузы, но
  логировать это как явный backtrack к пропущенному аяту.

## Чек-лист

- [x] Добавить text-aware анализ начал.
- [x] Подключить расширенные decode windows.
- [x] Усилить back/forward recovery в matcher.
- [x] Добавить тесты.
- [x] Запустить проверки.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py`
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py`
- `uv run pytest tests/test_align_fast.py` — 34 passed.
- `uv run pytest tests/test_candidates.py tests/test_align_fast.py` — 39 passed.

Docs updated: `docs/STATUS.md`,
`docs/tasks/2026-06-29-text-aware-pause-recovery/task.md`.

## Блокеры

Нет.
