# Релиз Ayah Studio 0.2.0

Цель: по прямому разрешению пользователя закоммитить все накопленные изменения,
обновить публичную страницу и скриншоты, выпустить новый DMG и Windows ZIP с EXE.
Прочитано: AGENTS.md/CLAUDE.md, RTK.md; packaging.md, release-notes,
README RU/EN, предыдущая public-release task и build.yml — порядок релиза.
Решения: GitHub origin, версия 0.2.0; модели/пользовательские БД и build/dist
не входят в Git. Скриншоты реального текущего UI на изолированной копии проекта.
Базовая модель встроена, дополнительные модели загружаются отдельно по выбору.

- [x] Проверки кода и frozen runtime.
- [x] Версия, описание, актуальные скриншоты.
- [x] Commit/push, tag, CI DMG/Windows, release.
- [x] Публичные assets и контрольные суммы.

Проверка: 399 pytest passed, 7 opt-in skipped; Ruff, JS syntax, diff-check,
AGENTS/CLAUDE sync — OK. Три новые JPEG просмотрены, ошибок JS нет.
Локальная .app: self-check 114/6236, ONNX/FFmpeg/assets, bundled uv 0.9.2,
codesign verify, hdiutil verify — OK. Финальная пересборка после исправления отмены.
Обновлён uv.lock: прежний CI падал из-за отсутствующего uv в lock.
Публикация через SSH и существующий CI, без gh/коннектора:

- release commit `bd541b1`, тег `v0.2.0`, main и tag отправлены в origin;
- [релиз](https://github.com/Shishani58/Ayah-Studio/releases/tag/v0.2.0);
- [CI 38444979962](https://github.com/Shishani58/Ayah-Studio/actions/runs/38444979962):
  checks/models/macOS/Windows/release — success;
- Windows frozen self-check и отвечающее нативное окно проверены CI;
- DMG и Windows ZIP скачаны целиком публично; SHA-256 совпали с GitHub digest
  и опубликованными .sha256. ZIP CRC, EXE, uv, worker, model_text_targets,
  studio.css — OK. Проверенный CI DMG: hdiutil, codesign, frozen self-check
  и версия 0.2.0 — OK. Сборки скопированы в dist/;
- README и три новых изображения доступны публично и просмотрены на GitHub.
  `published-page.jpg` — подтверждение; `published-assets.json` — SHA/размеры.

После фиксации релиза начались новые задачи ayah-progress/english-localization
и codebase-review; их незавершённые изменения не входят в тег и не затрагиваются
финальным документационным коммитом этой задачи.
Отдельный дублирующий main CI 38444973821 остаётся на pytest; релизный
CI того же commit прошёл полностью. Нотарификация/Windows Authenticode
по-прежнему отсутствуют. Блокеров публикации нет.
Docs updated: README.md, README.en.md, .github/release-notes.md,
docs/development/packaging.md, docs/STATUS.md, этот task-файл.

