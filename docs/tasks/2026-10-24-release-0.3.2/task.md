# Выпуск 0.3.2

## Цель и скоуп
Проверить и опубликовать накопленные изменения; собрать публичные macOS/Windows установщики штатным tag CI.

## Прочитанные документы
- AGENTS.md / CLAUDE.md: защита данных, релизные ограничения.
- docs/STATUS.md, README.md, .github/workflows/build.yml: текущий статус и способ публикации.
- docs/tasks/2026-10-17-ponytail-cleanup/task.md: состав накопленных изменений.

## Решения
- Следующая patch-версия: 0.3.2. Обновить uv.lock после удаления soundfile из основного runtime.
- Публиковать через существующий GitHub tag workflow; пользователь разрешил push и деплой.
- Пользовательские данные и модели не добавлять в Git.

## Чек-лист
- [x] Проверки и независимое ревью.
- [ ] Commit и push.
- [ ] Установщики опубликованы и проверены.

## Проверка
1013 passed / 10 skipped (16.45 s); Ruff — OK. Независимое runtime-ревью: блокеров нет; 136 focused tests passed. Нативная сборка будет проверена CI.

## Блокеры
В процессе.

## Docs
Docs updated: docs/STATUS.md, README.md, README.en.md, .github/release-notes.md, этот task-файл.

## Исправление публикации
Первый CI успешен, но macOS spec содержал старую константу 0.3.1.
CFBundleShortVersionString теперь берётся из metadata установленного проекта,
как версия Windows-архива; требуется повторная сборка macOS.
