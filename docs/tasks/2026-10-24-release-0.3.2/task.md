# Выпуск 0.3.3

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
- [x] Commit и push GitHub (86bfcca).
- [x] Установщики 0.3.3 опубликованы и проверены.

## Проверка
1013 passed / 10 skipped (16.45 s); Ruff — OK. Независимое runtime-ревью: блокеров нет; 136 focused tests passed. Нативные сборки macOS/Windows и их self-check прошли в CI.

## Блокеры
GitHub: нет. GitLab main был на 16 коммитов позади; передача около 1.13 GB
истории ещё идёт в фоне. После неё запланирован обычный push актуального main
и v0.3.3; завершение GitLab пока не подтверждено.

## Docs
Docs updated: docs/STATUS.md, README.md, README.en.md, .github/release-notes.md, этот task-файл.

## Исправление публикации
Первый CI успешен, но macOS spec содержал старую константу 0.3.1.
CFBundleShortVersionString теперь берётся из metadata установленного проекта,
как версия Windows-архива; требуется повторная сборка macOS.

Версия 0.3.2 заменяется выпуском 0.3.3: нет авторизации для workflow_dispatch,
теги опубликованных выпусков не переписываются. Новый tag CI соберёт обе платформы
из исправленного spec.

## Результат
- Публичный latest: https://github.com/Shishani58/Ayah-Studio/releases/tag/v0.3.3
- CI: https://github.com/Shishani58/Ayah-Studio/actions/runs/40045074219 — success.
- Оба публичных архива полностью скачаны потоком: размеры и SHA-256 совпали
  с .sha256 и digest GitHub API (см. release-verification.json).
- macOS 187056256 bytes; Windows 195557879 bytes. Имена обеих платформ — 0.3.3.
- На пользовательской ОС приложения вручную не запускались; self-check,
  codesign/hdiutil macOS и Windows package checks выполнены релизным CI.
- GitHub main синхронизирован; GitLab остаётся отдельной незавершённой передачей.
