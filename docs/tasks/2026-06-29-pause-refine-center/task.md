# Pause refine center

## Цель

Если fast уже сопоставил паузу с аятом, а локальный refine нашёл начало первого
слова рядом с этой паузой, считать паузу достоверной и писать старт аята в центр
паузы, а не в начало слова после паузы.

## Скоуп

- `surah_cutter/core/align_fast.py`: логика Phase 2 для pause-backed boundary.
- `tests/test_align_fast.py`: модельно-независимая регрессия для правила выбора.

Не менять формат БД, UI, детектор пауз и ручные тайминги.

## Прочитанные доки и код

- `AGENTS.md` из сообщения пользователя: правила проекта и task-файлы.
- `/Users/aabibulat1/Desktop/Badr_Turki/Badr_Turki.logs/sura_011_fast.json`:
  свежий trace суры 11.
- `surah_cutter/core/align_fast.py`: Phase 1 pause matching, Phase 2 refine,
  Phase 2b repair.
- `tests/test_align_fast.py`: существующие pure-helper тесты.
- `docs/STATUS.md`: текущая политика fast.

## Решение

- Добавить правило: pause-backed refine подтверждает паузу, если вернул onset с
  ненулевой confidence рядом с центром паузы.
- В этом случае записывать `pause.center_ms`, confidence брать не ниже
  `conf_thresh`, чтобы repair не уводил confirmed pause-backed старт вправо к
  первому слову.
- Если refine не подтвердил слово, оставить старое поведение fallback на
  matched guess.

## Чек-лист

- [x] Добавить helper и тест.
- [x] Обновить Phase 2.
- [x] Запустить тесты/линтер.
- [x] Зафиксировать проверку и docs status.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py` — passed.
- `uv run pytest tests/test_align_fast.py` — 28 passed.
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py` —
  passed.
- Практический fast-прогон без записи в БД:
  `/tmp/sura_011_fast_pause_refine_center.json`.
  - Аят 4: было `53.331s` после repair, стало `52.750s` — центр паузы
    `52.36-53.14s`.
  - Аят 6: стало `82.710s` — центр pause-backed rough вместо word-onset
    `82.943s`.
  - Сура 11 всё ещё `drift=True`, `mean_conf=0.365`, `verify_sim=0.316`; это
    отдельная проблема общего fast-дрейфа.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
