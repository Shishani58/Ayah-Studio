# Исправления по ревью — 2026-10-13

Все десять воспроизводимых замечаний F1–F10 из
[исходного ревью](../2026-10-13-codebase-review/review.md) исправлены.
Три специалиста работали параллельно: GPT-6 Astra — данные и независимое
ревью, GPT-6.1 Sol — сервер/настройки, GPT-6 Sol — поставка/происхождение.
Лид выполнил интеграцию, UI/core/runtime и итоговую проверку.

## Что исправлено

| Пункт | Результат | Основные файлы и регрессии |
|---|---|---|
| F1 · транслитерация | Воспроизводимая сборка согласует три известных объединения `baʿdamā` в 2:181, 8:6, 13:37. Проверяется каждый аят/слово, неожиданное расхождение блокирует загрузку. Cache пересобран кодом; Quran DB не менялась. | `data/transliteration.py`, `tools/build_transliteration.py`, `data/word_tr_cache.json`; `test_transliteration.py` |
| F2 · потеря ручной разметки | Чтение статуса и запись аятов/слов объединены в `BEGIN IMMEDIATE`; concurrent manual edit/verify защищены. Запрещён обход через `force=True`. | `data/project_db.py`; `test_project_db_safety.py` с двумя SQLite-соединениями |
| F3 · обратный порядок | Валидируется итоговый merge с ручными метками. Конфликт отклоняет всю транзакцию: timings, metadata, confidence/history. Ручная запись и Qt получают те же гарантии. | `project_db.py`, `gui/review_view.py`, `webgui/server.py`; `test_project_db_safety.py` |
| F4 · SSE/reconnect | Общая очередь заменена журналом 512 событий с независимыми курсорами, SSE ID, heartbeat и terminal replay. Status сохраняет job ID, итоговые счётчики и ошибки слов. UI восстанавливает задачу, сериализует callbacks и отменяет по ID. | `webgui/generation.py`, `server.py`, `static/generation.js`, `app.js`; `test_generation_stream.py`, `test_webgui_async.py` |
| F5 · настройки | Весь read/merge/replace защищён RLock, временные файлы уникальны. Выбор модели/проекта использует ту же запись. | `webgui/settings.py`, `model_store.py`; `test_settings_concurrency.py` |
| F6 · ложный ok | Пустой/blank/special-token decode и время вне записи дают `insufficient_evidence`, отдельно учитываемый как непроверенный результат. | `core/recheck.py`, `app.js`; `test_core_reliability.py` |
| F7 · устаревший UI | Общий context guard для recheck/reload/draft/correct/пауз; проверяются проект, сура, запрос и актуальность меток. Reload не откатывает ручную правку. Startup restore защищён epoch; смена суры освобождает кнопку пауз. | `app.js`, `server.py`; `tests/webgui_async.cjs` исполняет реальные функции с задержанными ответами |
| F8 · неполный wheel | Единый resolver для checkout/frozen/wheel; включены Quran DB, cache, targets, эталоны, шрифт/static. Веса внешние, поддерживается `SURAH_CUTTER_MODELS_DIR`. | `resources.py`, `pyproject.toml`, `surah_cutter.spec`, загрузчики; `test_packaged_resources.py`, чистая установка wheel |
| F9 · пропуск частичных сур в Qt | CLI/Web/Qt используют общие discovery/completeness. Полнота требует точного канонического набора целочисленных меток; наличие одного аята недостаточно. | `data/project_state.py`, `cli.py`, `gui/dashboard_view.py`; `test_project_state.py`, `test_cli_batch.py` |
| F10 · потеря происхождения | Общий writer сохраняет модель/revision/хэш артефакта, фактический backend/режим и lag; confidence атомарен с timings. CLI продолжает пакет после ошибки отдельной суры. Исправлены Torch и внешний runtime в tiled. | `data/alignment_result.py`, `model_store.py`, `align.py`, `align_tiled.py`, CLI/Web/Qt; `test_alignment_result.py`, `test_cli_batch.py`, `test_core_reliability.py` |

Пути приложения в таблице относительны к `surah_cutter/`, тесты — к `tests/`.
Исходные скрипты `codebase-review/reproduce*` остаются историческим
воспроизведением дефектов; актуальное правильное поведение проверяют регрессии.

## Надёжность, связность и повторная работа

- Общие операции фонетики выделены в `core/phonetics.py`, строгая проверка
  полного CTC-пути — в `_ctc.aligned_segments`. FULL/tiled/FAST/repair/words
  больше не расходятся в обработке неполных целей; исчезли импорты приватных
  текстовых helpers FAST из других режимов.
- Локальная фаза FAST выделена в `LocalAlignmentReads`. Поиск DSP-пауз
  для проверки onset кэшируется на один запуск. Бюджеты акустических чтений
  сохранены: до 6 секунд за вызов, покрытие до 65%, суммарное чтение до 200%.
  Автоматического перехода в FULL/tiled нет.
- CTC backtrace хранит переход в uint8 вместо int64: именно эта матрица
  занимает в 8 раз меньше памяти. Независимая сверка 672 synthetic случаев
  с прежним алгоритмом дала одинаковые paths/scores, включая ties/repeats.
  Это не утверждение о восьмикратном уменьшении всей памяти приложения.
- Acoustic windows материализуются только для текущего batch. Независимая
  проверка 18 конфигураций дала равные emissions. Точная PCM-длительность
  кэшируется по stat-идентичности файла с инвалидированием после изменения.
- Runtime deadline покрывает блокирующие pipe write/read, startup/probe,
  inference и установку. Зависший worker завершается с явной ошибкой,
  timer снимается перед повторным использованием. Проверены настоящие child
  processes с блокирующим вводом/выводом и cancellation.
- Optional runtime получил включённые в поставку universal hash-locks всех
  транзитивных зависимостей и `--require-hashes`; fingerprint меняется вместе
  с lock. Dry-run проверен для macOS arm64, Linux x64 и Windows x64.
- Из сервера выделены события генерации и writer результата; CLI/Web/Qt
  переиспользуют небольшие общие функции. Глобальная перестройка приложения
  и смена framework не выполнялись.

## Итоговая проверка

- Общий pytest текущего дерева: **520 passed, 7 skipped**, 6.70 секунды.
- Отдельный opt-in прогон настоящего аудио Шатри: **7 passed**, 24.37 секунды.
  Проверены FULL 112/114 и FAST 112/114/80/88/86, порядок/полнота,
  погрешность относительно независимого эталона, честная маркировка сомнений,
  реальные локальные чтения и отсутствие fallback.
- Ruff приложения, тестов, tools и текущей задачи — чисто. Полный `ruff check .`
  видит **22 ошибки в двух диагностических скриптах параллельной задачи**:
  `docs/tasks/2026-10-13-tarteel-models/api_probe.py` и `local_probe.py`.
  Эти чужие изменения сохранены, к исправлениям этого ревью не относятся.
- JS syntax, Node VM async-регрессии, `git diff --check`, AGENTS/CLAUDE sync — OK.
- Проверено 6236 аятов/77432 слова, непустые token spans для поставляемого
  phonetic vocabulary; builder воспроизводит те же байты cache.
- SHA-256 канонической DB сохранился:
  `c35272ed2a80916f5665f18e150686cfe258249cdf3ad50d3ac787f8cfe3fd66`.
- Offline wheel повторно собран и установлен non-editable в чистое окружение;
  запуск из `/tmp` без PYTHONPATH подтвердил 114/6236/77432, 46 эталонов,
  cache/font/static/new modules/hash-locks. ONNX в wheel не включены.
  Артефакт: `/tmp/ayah-wheel-integrated-20261013/ayah_studio-0.2.0-py3-none-any.whl`.
  SHA-256: `086d5ac1903cde39d712ee0c4921d65be9e3bbe8e989afef2c351c05a80dd1b3`.

Команды финальной проверки:

```bash
rtk proxy .venv/bin/pytest -q
rtk proxy env SC_TEST_AUDIO=1 .venv/bin/pytest -q tests/test_alignment_audio.py
rtk proxy .venv/bin/ruff check surah_cutter tests tools docs/tasks/2026-10-13-review-fixes
rtk proxy node --check surah_cutter/webgui/static/app.js
rtk proxy node --check surah_cutter/webgui/static/generation.js
rtk proxy git diff --check
rtk proxy cmp AGENTS.md CLAUDE.md
rtk uv build --wheel --offline --out-dir /tmp/ayah-wheel-integrated-20261013
```

## Границы результата

Пользовательские проекты/настройки, исходное аудио и каноническая DB не
использовались для записи; коммитов, push и релизов нет. Параллельные изменения
локализации, прогресса и Tarteel сохранены, их реализацию этот отчёт не присваивает.

Не проверены весь акустический корпус 114 сур, слуховая правильность всех
границ, реальная нагрузка длиннейших записей/peak RSS, native Qt/WKWebView,
готовые DMG/EXE и исполнение на Windows/Linux. Wheel smoke проверяет упаковку
ресурсов, а не устанавливает весь стек runtime или запускает native GUI.

FAST остаётся большим эвристическим модулем; server/app.js сохраняют глобальное
состояние одного desktop-приложения. Выделены конкретные общие границы,
полная архитектурная декомпозиция остаётся техническим долгом.
SQLite commit и замена диагностических файлов не атомарны совместно:
ошибка диска после commit может оставить старый sidecar, хотя происхождение
и confidence уже сохранены в БД. Settings-lock действует внутри одного процесса.
Это ограничения результата, а не обещание абсолютной надёжности всех сценариев.

Docs updated: `docs/project.md`, `docs/STATUS.md`, этот отчёт и `task.md`.
