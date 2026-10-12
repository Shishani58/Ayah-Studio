# Ayah Studio: интерфейс и macOS DMG

## Цель / скоуп
Переименовать продукт в Ayah Studio; привести web-интерфейс в порядок;
собрать автономный macOS DMG с установкой переносом в Applications.
Алгоритм выравнивания, текст Корана и пользовательские БД не менять.

## Прочитано
- AGENTS.md / CLAUDE.md, RTK.md — правила и сохранение существующих изменений.
- docs/project.md — границы модулей и инварианты данных.
- docs/STATUS.md — основной GUI уже web, упаковка частично готова.
- docs/development/packaging.md — текущий PyInstaller и ассеты.
- web-ui-ux, better-layout — группировка, адаптивность и проверка рендера.

## Решения
- Пользовательское имя Ayah Studio; Python namespace, старые CLI и settings
  сохранить для совместимости, добавить новые команды.
- Сохранить тёмную палитру и золотые маркеры. Разделить настройку проекта,
  генерацию и каталог; добавить фильтры, клавиатурный доступ и пустые состояния.
- Использовать PyInstaller + системный hdiutil, все runtime-ассеты внутри .app.
- Сборка по архитектуре машины. Локально arm64; Intel требует отдельной сборки.
- Не трогать существующие изменения core, tests, bench и project.md.

## Чек-лист
- [x] UI и новое имя
- [x] Автономная .app, иконка, DMG и инструкция установки
- [x] Проверка браузера, API, сборки и запуска из DMG
- [x] Документация / STATUS

## Проверка
- `pytest tests/test_webgui_server.py tests/test_export_db.py -q`: 35 passed
  (включая новые проверки из параллельной работы в том же checkout).
- Ruff для launch/selfcheck/build/fetch tools; `node --check app.js`; `git diff --check`;
  `uv lock --check`; `cmp AGENTS.md CLAUDE.md` — успешно.
- DMG: 164 MiB, app 288 MiB, arm64. PyInstaller, codesign verify и hdiutil verify — успешно.
- Frozen `--self-check` из bundle и с read-only DMG, cwd `/tmp`: 114 сур / 6236
  аятов, ONNX inference, загрузка Silero, FFmpeg, статические ресурсы — OK.
- Нативное приложение запущено непосредственно из финального DMG: заголовок,
  иконка, восстановление настроек, каталог, системный Open dialog и отмена — OK.
- Browser: 1180×840, 820×600, 320×740; нет горизонтального переполнения на 320;
  открытие копии проекта, поиск 114, Enter на суре/аяте, проигрывание, арабский
  шрифт и волна — OK. Скриншоты рядом: native.jpg, dashboard.jpg, review.jpg.
- VoiceOver, 200% text zoom, Intel/Windows и чистый Mac не проверены.
- В checkout параллельно изменялись export_db, server/state, partial-статусы и
  тесты другим процессом. Их изменения сохранены; алгоритм здесь не редактировался.

## Блокеры
Нет Developer ID Application; публичная нотарификация не проверена.
Git remote — GitLab, существующий GitHub workflow сам по себе там не запускается.


## Результат / файлы
- UI: `surah_cutter/webgui/static/{index.html,style.css,app.js,mark.svg}`.
- Имя: `pyproject.toml`, `uv.lock`, `__init__.py`, CLI, оба GUI entrypoints,
  заголовок FastAPI. Старые команды, namespace и настройки сохранены.
- Упаковка: `surah_cutter.spec`, `assets/AyahStudio.icns`, `tools/create_icon.swift`,
  `tools/build_macos.py`, `tools/fetch_runtime_models.py`, `surah_cutter/selfcheck.py`,
  `.github/workflows/build.yml`.
- Артефакт: `dist/Ayah-Studio-0.1.0-macOS-arm64.dmg` + SHA-256.
- Docs updated: README.md, docs/development/packaging.md, docs/STATUS.md,
  этот task.md. Исторические записи с прежним названием не переписывались.
- Коммиты, push, публикация релиза и установка в /Applications не выполнялись.
