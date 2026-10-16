# Прогресс разметки

Цель: показывать реальную работу до появления границ аятов и оценку времени
текущего измеряемого этапа. Счётчик аятов не подменять процентом inference.

Прочитано: docs/project.md (пути FULL/FAST и инварианты), docs/STATUS.md
(текущее состояние), web-ui-ux/SKILL.md (существующий HTML/JS интерфейс).

Решения: отдельные структурированные события этапов; прогресс окон/кадров
по фактически завершённой работе; неизвестный объём — неопределённый индикатор.
ETA только текущего этапа после измерений, без выдуманного общего времени.
Контракт FAST, вычисления и сохранение ручных меток не меняются.

- [x] Инструментировать подготовку, FULL inference/CTC, FAST, сохранение.
- [x] Передавать события через SSE и восстанавливать после reconnect.
- [x] Показывать этап, единицы работы, время и неопределённое ожидание.
- [x] Проверить регрессии Python/JS и обновить STATUS.

Проверка: полный pytest — 772 passed, 8 skipped (9.60s); node
 tests/webgui_async.cjs — passed; scoped Ruff — passed. Новые регрессии:
прогресс до первого аята, неизменность emissions, ETA/throttle, вложенные
scope, отмена до сохранения, RPC, SSE eviction и восстановление интерфейса.
Рендер тех же функций на отдельном fixture проверен через браузер в
1280×720 и 390×844, измеряемый этап и загрузка без известного объёма.
Временный fixture удалён. Общий ETA намеренно отсутствует: этапы неоднородны.
Реальный длинный model-run и нативный Qt не проверялись/не изменялись.
Блокеров нет. Чужие изменения в workspace сохранены.

Docs updated: docs/project.md, docs/STATUS.md, этот task.md.

Изменённые файлы:
- core/progress.py — thread scope, этапы, throttle и ETA;
- core/pipeline.py, align.py, align_tiled.py, fast_executor.py — переходы этапов;
- core/emissions.py, full_windows.py, _ctc.py — фактические единицы работы;
- core/model_runtime.py, model_worker.py — промежуточный прогресс RPC;
- webgui/server.py, generation.py — SSE и snapshot;
- webgui/static/app.js, i18n.js, studio.css — отображение и перевод;
- tests/test_stage_progress.py, tests/webgui_async.cjs — регрессии.
