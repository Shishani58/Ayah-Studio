# Исправления ревью и аудит legacy

## Цель и scope
Исправить пять проблем общего ревью: идентичность FULL resume, оставшийся
auto/reference shortcut (авторежим по уточнению пользователя отсутствует),
ручные границы аудио, сохранение db_version и несовместимость FAST benchmark.
Дополнительно проверить legacy по реальным вызовам; массовое удаление вне
подтверждённого scope не выполнять.

## Прочитанные документы
- AGENTS.md / CLAUDE.md: инварианты, защита ручных меток, task workflow.
- docs/project.md: основной pipeline, FAST/FULL и сохранение provenance.
- docs/STATUS.md: актуальное состояние алгоритмов и воспроизведения.
- ~/.codex/RTK.md: запуск команд через RTK.

## Решения
- Пользователь явно поручил субагентам исправления; работа в общей папке,
  зоны файлов разделены. Общие docs и интеграцию ведёт основной агент.
- Существующие незакоммиченные изменения сохраняются; commit/push запрещены.
- Аудио, канонический текст и пользовательские БД не изменять.
- Auto как алгоритм отличается от статуса review_meta.status='auto': статус
  автоматических меток сохраняется.
- FULL resume без полной текущей execution identity пропускает только полностью
  edited/verified суры; совпадение имени модели и политики недостаточно.
- Удалены исполняемый auto и reference shortcut, параметры reciter расчёта и
  CLI --reciter. Названия чтецов в проектах и исторические source сохранены.
- Независимое ревью обнаружило риск отказа запуска при невозможности сохранить
  миграцию настроек. Исправлено: выбор FULL не зависит от сохранения; миграция
  best-effort. Добавлена проверка недоступной записи настроек.

## План
- [x] FULL resume: проверка записи и модели либо консервативный пересчёт.
- [x] Удалить доступность устаревшего auto/reference shortcut в рабочем pipeline.
- [x] Серверные и UI проверки ручной границы строго внутри аудио.
- [x] Не менять db_version существующего проекта при открытии.
- [x] Согласовать benchmark FAST с текущими видами evidence.
- [x] Проверить и описать кандидатов legacy с доказательствами использования.
- [x] Общий pytest, Ruff, четыре реальных аудиотеста FULL/FAST, diff/sync.

## Проверка и ограничения
До исправлений: 948 passed / 10 skipped; Ruff OK. Дополнительные реальные
112/114: FULL 2 passed, FAST 2 failed из-за старого pause contract checker.
Четыре ошибки хранения/маршрутизации воспроизведены на временных данных.
Финальный общий pytest: **971 passed / 10 skipped** (87.81 с).
После правки миграции также 101 целевой тест прошёл. Настоящие FULL/FAST 112/114:
4 passed; прежние ошибки FAST checker устранены. Ruff, JS syntax, Node UI
regressions, diff-check и синхронность AGENTS/CLAUDE проходят.

Роли: full_resume — resume/регрессии, db_manual_bounds — metadata и ручные
границы, auto_and_fast_checker — pipeline/CLI и checker. Основной агент —
API/Qt/JS интеграция, независимое ревью, docs и общий прогон.

## Изменённые файлы
- `surah_cutter/core/pipeline.py`, `surah_cutter/cli.py`: удаление auto/shortcut.
- `surah_cutter/data/alignment_identity.py`, `project_db.py`: resume и metadata.
- `surah_cutter/webgui/server.py`, `static/app.js`, `static/i18n.js`:
  bounds, API/defaults, миграция настроек и удаление auto-only UI текста.
- `surah_cutter/gui/dashboard_view.py`: актуальный default/API в Qt worker.
- `tools/bench_fast.py`: проверка актуального evidence FAST.
- Регрессии в tests: full_resume, project_db_safety, webgui_server,
  pipeline, cli_batch, bench_fast, fast_default, tarteel_fast,
  settings_concurrency, generation_write_choice, qt_generation_contract,
  alignment_commit_contract, generation_progress, generation_stream,
  stage_progress и webgui_async.cjs.

## Ограничения и блокеры
Блокеров нет. Native GUI/Windows и полное прослушивание всех сур не выполнялись.
FULL skip намеренно консервативный; точное переиспользование потребует отдельной
полной execution identity. Дополнительный legacy описан в [legacy.md](legacy.md),
без массового удаления optional entrypoints и исследовательских исходников.

## Документация
Docs updated: `docs/project.md`, `docs/STATUS.md`, этот task и `legacy.md`.
