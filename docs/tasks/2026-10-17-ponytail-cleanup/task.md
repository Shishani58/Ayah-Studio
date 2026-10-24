# Упрощения по Ponytail-ревью

## Скоуп
Применить шесть находок ревью: удалить старую коррекцию FAST и uniform fallback,
убрать soundfile из основного runtime/сборки, объединить FAST entrypoint и
дублирующее выполнение ffmpeg. Отдельный fingerprint и совместимость FAST-флагов
сохранить. Незакоммиченные изменения других задач не менять.

## Доки
- docs/project.md — текущий pipeline, отдельный fingerprint и контракты FAST.
- docs/development/task-files.md — формат записи решений.
- AGENTS.md / CLAUDE.md — защита ручных меток, отсутствие reference influence.

## Решения
- Удалить тесты снятой с runtime коррекции; проверить действующий fingerprint.
- Публичный experimental_fast и внутренний experimental остаются aliases,
  без отдельного выбора executor.
- Общий private ffmpeg runner сохраняет команды, отмену, ошибки и writable PCM.
- Soundfile в изолированных NeMo/MFA окружениях оставить: это их зависимости.

## Чек-лист
- [x] Применить все шесть находок.
- [x] Проверить fingerprint, FAST aliases, decode/отмену и полный pytest/Ruff.
- [x] Обновить STATUS и зафиксировать итог.

## Проверка
- Целевые проверки: 70 passed; после финальной чистки тестов — ещё 7 passed.
- Полный pytest: 1013 passed / 10 skipped (16.82 s). Удалены 10 тестов снятой
  коррекции; добавлены 1 тест действующего fingerprint и 6 проверок настоящего
  ffmpeg: точный writable PCM, ошибки с путём и отмена для full/range.
- Исправлено затенение параметра experimental внутри существующего теста:
  теперь обе параметризации действительно проверяют False/True.
- Ruff, git diff --check, побайтовое совпадение AGENTS.md/CLAUDE.md прошли.
- В затронутом runtime удалено 529 строк нетто (без тестов/доков); одна основная
  зависимость soundfile и её PyInstaller hiddenimport сняты.
- Реальный model inference и нативная сборка не запускались; пользовательские
  БД/аудио/модели не изменялись. Удалённые symbols не имеют рабочих вызовов.

## Блокеры
Нет.

## Docs
Docs updated: [docs/STATUS.md, docs/tasks/2026-10-17-ponytail-cleanup/task.md].
docs/project.md не менялся: действующие алгоритмы и контракты прежние.
