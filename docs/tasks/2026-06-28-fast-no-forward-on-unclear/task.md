# Fast no forward on unclear decode

## Цель

Закрепить правило: fast не идёт дальше по аятам, если текущая пауза/окно
декодировались не как ожидаемый текущий аят. Сначала нужно понять, где мы:
прошлый аят, текущий, будущий или шум, и только потом двигаться назад/вперёд
или запускать локальный поиск.

## Скоуп

- `surah_cutter/core/align_fast.py`: sequential matcher не должен молча
  продвигать `k` при `no_confirmed_match`.
- `tests/test_align_fast.py`: добавить/обновить модельно-независимый тест.
- `docs/initiatives.md`: записать инициативу.
- `docs/STATUS.md`: кратко зафиксировать правило.

## Доки

- `docs/initiatives.md` — журнал инициатив.
- `docs/tasks/2026-06-28-fast-confidence-layers/task.md` — текущий слойный поиск.
- `docs/project.md` — не перечитывался: формат БД не меняется.

## Решения

- При `no_confirmed_match` возвращать событие `unresolved_current` и завершать
  текущий sequential pass, а не делать `k += 1`.
- В событие записывать лучший услышанный аят (`observed_boundary`) и sim, чтобы
  внешний слой мог понять, куда двигаться.
- Внешний слой уже имеет recovery-механику: low-confidence pauses и micro-scan
  должны получать такой unresolved-интервал как повод к локальному поиску.

## Чек-лист

- [x] Обновить журнал инициатив.
- [x] Изменить matcher.
- [x] Подключить unresolved к локальному recovery.
- [x] Добавить тест.
- [x] Проверить ruff/pytest.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py surah_cutter/webgui/server.py`
- `uv run ruff check surah_cutter/core/align_fast.py surah_cutter/webgui/server.py tests/test_align_fast.py`
- `uv run pytest tests/test_align_fast.py` — 27 passed.
- `uv run pytest tests/test_candidates.py tests/test_pipeline.py` — 7 passed.

## Docs

Docs updated: `docs/initiatives.md`, `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
