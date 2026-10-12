# Windows-сборка Ayah Studio

## Цель
Собрать и опубликовать Windows x64 ZIP с готовым EXE в существующем релизе.

## Прочитано
- AGENTS.md, RTK.md — правила репозитория.
- packaging.md, build.yml, surah_cutter.spec, launch.py, selfcheck.py — упаковка и проверки.

## Решения
- Portable ZIP: распаковать целиком и запустить EXE; без исходников и Python.
- Сборка на Windows GitHub runner; frozen self-check и проверка появления окна.
- Ручной запуск CI может выбрать только Windows и добавить assets к существующему релизу.
- Параллельные незакоммиченные правки иконки не включать.

## Чек-лист
- [x] Подготовить Windows упаковку и CI
- [x] Собрать и проверить на Windows
- [x] Опубликовать ZIP/SHA-256 и обновить страницу

## Проверка / блокеры
- 266 pytest passed; Ruff и diff check — OK.
- CI https://github.com/Shishani58/Ayah-Studio/actions/runs/38291457116 — success.
- Windows PyInstaller, frozen runtime self-check и нативное окно — success.
- ZIP опубликован в v0.1.0 вместе с checksum; DMG сохранён.
- Скачивание без авторизации: 176728578 байт; SHA-256
  `beebfdd47253e188b69e0aa03e150119578f7c729322da9ae9801cd75a0dd447`.
- ZIP CRC, EXE PE header и обе модели проверены. Скриншот: published.jpg.
- Ограничения: EXE без Authenticode; Windows x64 runner, не полный ручной
  прогон на пользовательском Windows 10/11. Для окна требуется WebView2.
- Исправление Windows старта находится в ee1cbc0; исходный тег v0.1.0
  сохранён, точный Windows build commit указан в описании релиза.

Docs updated: README.md, README.en.md, .github/release-notes.md, docs/STATUS.md, docs/development/packaging.md, этот task-файл.
