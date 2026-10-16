# Расчёт вероятности точного времени

Цель: вычислять и показывать вероятность попадания метки в ±500 мс; numeric filter/sort, работа на FULL и FAST. Продолжение boundary-accuracy по прямому требованию пользователя.

Прочитаны: AGENTS/CLAUDE, RTK, project/STATUS ранее; timing_accuracy/onset_evidence; boundary-accuracy/report/audit; full-other-reciters и full-pause-windows сохранённые проверки; FAST baseline traces; web-ui-ux.

Смысл процента: эмпирическая вероятность |метка − packaged reference time| ≤500 мс на том же аудио; reference convention не переименовывается в физический speech onset. Классификатор использует только доступные до reference признаки границы/записи, не ayah/time lookup и не реальное расхождение.

Решения: обучить на сохранённых реальных прогонах нескольких чтецов, проверить на целых отделённых записях; не разбивать соседние аяты случайно. Записать ECE/Brier/ошибки/coverage и пригодность фильтра. Отдельные модели FAST/FULL; без новых acoustic calls и изменения тайм-кодов. Проценты приблизительные и с документированным benchmark target.

- [x] Corpus и reproducible split без повторных конфигураций в train/test
- [x] Features, fitting, held-out calibration и review metrics
- [x] Bundled probability inference и строгая provenance
- [x] UI percent/filter/sort и manual protections
- [x] Regression/browser/replay, документация

Блокеры: нет. Ограничения: external reference convention; число независимых записей; версии моделей. Не скрывать их при выпуске процентов.

Завершено 2026-10-16: эмпирическая Beta(1,1) калибровка шести групп, без recognition→percent. 2192 предсказания/59 записей/5 чтецов; Шатри2 исключена из обучения. FULL ≤90%: 33/34 расхождения при 60/286 ревью. 151 целевой тест, Node, Ruff/JS/diff/sync/browser/replay OK; общая suite 828 passed/9 skipped, 3 параллельных сбоя каталога моделей. Детали и пределы калибровки report.md. Не объявлять ECE гарантией для подгрупп.

Финальное решение: один и тот же глобальный DSP-интервал в обучении и inference; локальное позиционирование имеет другую конвенцию краёв. Замороженная выборка воспроизводит артефакт побайтово. Дополнительная сура67: 97,51% прогноз/96,67% факт. Блокеров реализации нет; только документированные статистические ограничения/область моделей.

Docs updated: task.md, report.md, docs/project.md, docs/STATUS.md.
