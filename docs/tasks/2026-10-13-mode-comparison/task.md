# Сохранение и сравнение FAST/FULL

Цель: сохранять каждый успешный FAST/FULL-прогон и сравнивать два результата
одной суры: модель/время запуска, времена/уверенность, разницы и A/B-аудио.

Прочитаны AGENTS/CLAUDE, RTK, docs/project.md (pipeline, merge, sidecars),
предыдущая задача confidence-rerun. UI — ранее загруженный web-ui-ux.

Решения: неизменяемые JSON-снимки в `<project>.logs/runs`, без изменения
SQLite/экспорта. Сохранять исходные модельные времена до overlay ручных меток;
сравнение только читает данные. Каждый запуск — отдельный ID. Старые FAST/FULL
JSON-логи доступны как legacy-снимки (в них могли быть сохранённые ручные метки).
По умолчанию последние FAST и FULL, можно выбрать другие прогоны.
Разница не означает измеренную точность; confidence разных режимов не калибрована.

Проверка: [x] writer/history/rollback/legacy/API; [x] async/DOM/JS;
[x] реальное FAST→FULL→FAST и сравнение; [x] ru/en, темы, 390px/desktop,
прослушивание; [x] docs/STATUS. Ручные проекты для QA не изменять.

Блокеров нет. Рабочее дерево содержит предыдущие и сторонние изменения.

Завершено 2026-10-14. Файлы: data/alignment_runs.py,
data/alignment_result.py, core/pipeline.py, webgui/server.py,
static/app.js/index.html/studio.css/i18n.js; tests/test_alignment_runs.py,
tests/webgui_async.cjs. Общий writer сохраняет версии также для CLI/Qt;
Web generation и correction передают db_path/log_lines в этот writer.

Проверки: вся suite 547 passed, 7 opt-in skipped; Ruff всей базы, JS syntax,
diff-check. Тесты: три независимых версии с повтором режима и reopen;
raw-модельные времена с сохранностью ручной БД; rollback без нового снимка;
архивация legacy; symlink/traversal/чужая сура/повреждённый JSON; read-only API;
задержавшаяся пара не перерисовывает UI; clip boundaries/close отменяют playback.

Реальное аудио Шатри 114, Quran Phonetics INT8, default flags, lag 200 мс:
FAST→FULL→FAST = 1.866/3.934/1.147 с (с decode, первый с load).
Три версии; FULL и второй FAST записали 4/6 меток, edited/verified защищены.
В сравнении исходных предложений: 6 аятов, mean |Δ|=53 мс, max=90 мс.
Ранний и поздний FAST совпадают. [Результаты/модельные предложения](audio-check.json).

IAB: ru/light 1280×720 и en/dark 390×844. Выбор трёх версий, одинаковые
ID, фильтр без совпадений, реальные A/B currentTime/playback; Escape немедленно
останавливает и выгружает аудио. Таблица на 390px без горизонтального overflow;
JS ошибок нет. [Проверенный экран](comparison-ui.png).
Сверены SHA256 пользовательской БД/settings до/после: не изменились.

Ограничения: SQLite и JSON не общая транзакция; ошибка публикации после commit
может оставить записанную БД без снимка. Legacy-данные не восстанавливают
перезаписанную ранее историю и могут содержать ручные времена. Аудио-идентичность
проверяется метаданными файла. WKWebView/.app/Windows не проверены;
desktop binary не пересобирался. FAST-контракт, текст Корана, формат экспорта
не менялись.

Docs updated: docs/project.md, docs/STATUS.md, этот task.md.
