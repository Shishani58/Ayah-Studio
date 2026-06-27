# Задача: визуальная проверка пауз

## Цель

Сделать так, чтобы пользователь видел не только цветные паузы на волне, но и
все решения алгоритма: принята/отклонена, strict/weak/VAD, длительность и DSP
метрики. Это нужно для движения к схеме `паузы → кандидаты → forced alignment →
review`, где спорные места видно глазами.

## Скоуп

- `surah_cutter/webgui/static/index.html`: добавить панель списка пауз под
  waveform.
- `surah_cutter/webgui/static/app.js`: отображать кандидаты видимого диапазона,
  переходить к паузе, показывать метрики и статистику.
- `surah_cutter/webgui/static/style.css`: компактная диагностическая таблица.
- `docs/STATUS.md`: зафиксировать поведение.

## Прочитанные доки

- `docs/STATUS.md`: текущая схема pause pipeline и web review.
- `docs/tasks/2026-06-27-pause-overlay/task.md`: предыдущая визуализация пауз.
- `docs/tasks/2026-06-28-pause-edge-refine/task.md`: последние изменения
  edge-refine.

## Решения

- Не вводить новый формат БД и не менять расчёт.
- Использовать существующий `/api/sura/{n}/pause-pipeline`.
- Список показывает только паузы в видимом диапазоне waveform, чтобы длинные
  суры не забивали экран сотнями строк.
- Клик по строке перематывает аудио в центр паузы и центрирует waveform.

## Чек-лист

- [x] Добавить панель в UI.
- [x] Связать панель с overlay/scroll/zoom.
- [x] Проверить lint/тесты и базовую работу API.
- [x] Обновить docs/status.

## Проверка

`node --check surah_cutter/webgui/static/app.js` — passed.
`.venv/bin/pytest tests/test_candidates.py tests/test_align_fast.py` — 16 passed.

Browser smoke на `http://127.0.0.1:8767`, проект из settings
`/Users/aabibulat1/Desktop/Badr_Turki`, сура 4:
кнопка «паузы» открывает `#pauseInspector`; summary:
`всего 681 · принято 655 · weak 158 · откл. 26 · видно 681`; строки таблицы
созданы, console errors нет.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
