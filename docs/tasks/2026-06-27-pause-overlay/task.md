# Задача: кнопка показа найденных пауз

## Цель

Добавить в экран ревью кнопку, которая визуально показывает паузы production
pipeline, включая новые слабые `weak-dsp`, чтобы можно было быстро увидеть,
что алгоритм нашёл и что отклонил.

## Скоуп

- Web-GUI: кнопка в панели волны.
- Использовать существующий API `/api/sura/{surah}/pause-pipeline`.
- Canvas-overlay поверх волны: принятые строгие, принятые weak, отклонённые.
- Без изменения схемы БД и алгоритма разметки.

## Прочитанные доки

- `docs/project.md`: инварианты разметки и БД.
- `docs/STATUS.md`: текущее состояние web-GUI и fast/pause pipeline.
- `docs/tasks/2026-06-27-pause-recall/task.md`: новая логика `weak-dsp`.

## Решения

- Не добавлять новый backend endpoint: `pause-pipeline` уже отдаёт `accepted`,
  `reason`, `start_ms`, `end_ms`, `center_ms`, метрики DSP.
- Overlay не меняет навигацию и тайм-коды; это read-only диагностический слой.
- Цвета: строгие accepted — синий, `weak-dsp` — янтарный, rejected — красный
  пунктир/полупрозрачный фон.

## Чек-лист

- [x] Добавить кнопку и легенду.
- [x] Загрузить/сбросить overlay в JS.
- [x] Нарисовать паузы на canvas.
- [x] Проверить lint/тесты.
- [x] Зафиксировать docs/status.

## Проверка

`node --check surah_cutter/webgui/static/app.js` — passed.
`.venv/bin/pytest tests/test_candidates.py` — 3 passed.
Browser smoke: `http://127.0.0.1:8766` загрузился, `#pauseOverlayBtn` и
`#pauseLegend` есть в DOM, console errors отсутствуют.

Docs updated: `docs/STATUS.md`, этот task-файл.

## Блокеры

Нет.
