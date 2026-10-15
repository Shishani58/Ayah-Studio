# Релиз Ayah Studio 0.3.0

Цель: по явному запросу закоммитить все изменения и опубликовать новую версию.
Скоуп: накопленные FAST/FULL, confidence/review UI, тесты, исследовательские
отчёты; версия 0.3.0, GitHub origin, существующий tag CI macOS/Windows.
Прочитано: RTK.md, AGENTS.md/CLAUDE.md, packaging.md, build.yml,
release-notes, предыдущая release task, STATUS — правила и процедура выпуска.
Решение: включить все неигнорируемые изменения; модели/пользовательские БД,
build/dist исключены. GitHub CLI не авторизован; push через SSH,
контроль CI и релиза через публичный API.

- [x] Обновить версию и описание.
- [x] Ruff, pytest, JS и diff-check.
- [ ] Коммит, push main и тег v0.3.0.
- [ ] Дождаться CI и проверить опубликованные assets.

Ruff: архивные снимки docs/tasks исключены из линтинга; production/tools/tests
проверяются полностью. Исходные снимки сохранены для воспроизводимости.

Блокеры: пока нет.
Docs updated: README.md, README.en.md, .github/release-notes.md,
docs/development/packaging.md, docs/STATUS.md, этот task-файл.

Проверка: 734 passed, 8 opt-in skipped (17,57 с); Ruff, JS syntax,
node tests/webgui_async.cjs, diff-check, AGENTS/CLAUDE cmp — OK.

Первый commit 3dc2864 и tag v0.3.0 отправлены в origin. CI 38792455423
остановился на pytest: unit tests зависели от локальных ONNX/Silero и vocab.
На чистом snapshot Python 3.12 воспроизведены и устранены зависимости:
модельные загрузки подменены, vocab 377 байт сохранён как test fixture.
Также исправлено зависание очистки FFmpeg stdout в Python 3.12: communicate()
после kill вместо wait; существующий disconnect test ограничен 10 секундами.
Чистый snapshot после исправлений: 734 passed, 8 skipped (9,54 с).
Исправленный выпуск: 0.3.1; старый tag не переписывается.
Параллельные изменения после 3dc2864 в этот выпуск не включаются.
