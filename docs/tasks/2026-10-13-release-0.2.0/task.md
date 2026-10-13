# Релиз Ayah Studio 0.2.0

Цель: по прямому разрешению пользователя закоммитить все накопленные изменения,
обновить публичную страницу и скриншоты, выпустить новый DMG и Windows ZIP с EXE.
Прочитано: AGENTS.md/CLAUDE.md, RTK.md; packaging.md, release-notes,
README RU/EN, предыдущая public-release task и build.yml — порядок релиза.
Решения: GitHub origin, версия 0.2.0; модели/пользовательские БД и build/dist
не входят в Git. Скриншоты реального текущего UI на изолированной копии проекта.
Базовая модель встроена, дополнительные модели загружаются отдельно по выбору.

- [ ] Проверки кода и frozen runtime.
- [x] Версия, описание, актуальные скриншоты.
- [ ] Commit/push, tag, CI DMG/Windows, release.
- [ ] Публичные assets и контрольные суммы.

Проверка: 399 pytest passed, 7 opt-in skipped; Ruff, JS syntax, diff-check,
AGENTS/CLAUDE sync — OK. Три новые JPEG просмотрены, ошибок JS нет.
Локальная .app: self-check 114/6236, ONNX/FFmpeg/assets, bundled uv 0.9.2,
codesign verify, hdiutil verify — OK. Финальная пересборка после исправления отмены.
Обновлён uv.lock: прежний CI падал из-за отсутствующего uv в lock.
Блокеры: gh/коннектор без авторизации; публикация через SSH tag и существующий CI.
Docs updated: README.md, README.en.md, .github/release-notes.md,
docs/development/packaging.md, docs/STATUS.md, этот task-файл.

