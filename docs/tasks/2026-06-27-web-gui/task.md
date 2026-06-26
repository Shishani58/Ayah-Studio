# Web-GUI — замена PySide6/Qt на лёгкий web-стек

## Цель / скоуп

Заменить desktop-GUI (PySide6 1.2 ГБ + pyqtgraph) на лёгкий web-GUI, чтобы попасть
в целевой вес дистрибутива ~250 МБ (после выкидывания torch остался Qt как главный
драйвер веса). Переписать: дашборд (114 сур, статусы, запуск генерации с прогрессом)
и экран ревью (плеер, волна с перетаскиваемыми маркерами, текст Корана с подсветкой,
навигация по аятам, правка границ, статусы verified, экспорт).

## Стек (решение)

- **pywebview** — нативное окно через системный WebView (WKWebView на macOS, Edge
  WebView2 на Windows), без бандла Chromium. ~2 МБ + pyobjc (уже есть на маке).
- **FastAPI + uvicorn** (~10 МБ) — локальный сервер: JSON-API + SSE-прогресс
  генерации + раздача статики и аудио. Запускается в фоне-потоке, окно открывает
  pywebview (фоллбэк — браузер).
- **Фронт** — vanilla HTML/CSS/JS (без сборки). Аудио = HTML5 `<audio>`; волна =
  `<canvas>` по пикам, посчитанным в Python из PCM; маркеры = перетаскивание мышью.

Итого ~10 МБ против Qt 1.2 ГБ.

## Прочитанные доки/код

- `data/project_db.py` — `ProjectDB.open(path, code, version)`; `write_timings(rows,
  source, confidence)` (MERGE, не трогает edited/verified); `set_time` (→edited+
  edit_history); `set_status`; `get_timings`; `get_review`.
- `core/pipeline.py` — `process_surah(audio, surah, *, db_path, reciter, mode,
  project_db)` → `[(sura,ayah,start_ms)]`; mode hybrid/full.
- `data/quran_text.py` — `surah_ayah_counts`, `ayah_text`, `surah_text`, `surah_words`.
- `core/decode.py` — `decode(path)`→PCM16k, `duration_ms(path)`.
- `gui/dashboard_view.py` / `cli.py` — паттерны: `_DB_CODE`/`_DB_VERSION`,
  `_AUDIO_EXTS`, парс суры из имени файла, классификация статуса суры.

## Решения

- Не трогаю ядро (core/data) — web только обвязка поверх существующих API.
- Старый `gui/` (PySide6) оставляю как есть, но entry `surah-cutter-gui`
  переключаю на webgui. PySide6/pyqtgraph убираю из core-deps → в extra `qt`
  (для тех, кому нужен старый GUI). pywebview/fastapi/uvicorn → core.
- Один процесс: uvicorn в потоке + pywebview в главном потоке.

## Чек-лист

- [x] `webgui/state.py` — общее состояние проекта (audio dir, db path, sura→file).
- [x] `webgui/server.py` — FastAPI: /api/project, /api/suras, /api/generate(+SSE),
      /api/sura/{n}, /api/sura/{n}/waveform, /api/audio/{n}, /api/timing, /api/verify,
      /api/export.
- [x] `webgui/static/{index.html,app.js,style.css}` — дашборд + ревью.
- [x] `webgui/launch.py` — uvicorn-поток + pywebview-окно (фоллбэк браузер).
- [x] entry `surah-cutter-gui` → `webgui.launch:main`; pyproject deps (Qt → extra `qt`).
- [x] Smoke headless (curl): все эндпоинты, SSE-генерация, Range-аудио, правки. ✅
- [ ] Живой рендер окна на дисплее (computer-use отклонён) — за пользователем.

## Итог

Готово. Точность не затронута (web — только обвязка над core). Вес UI-deps ~10 МБ
(было Qt 1.2 ГБ). `surah-cutter-gui` → web-окно; `surah-cutter-gui-qt` — старый Qt.

Docs updated: `docs/STATUS.md` (ф.5 web-GUI), `README.md`, эта task-карточка.

## Проверка

- API: curl ключевые эндпоинты на реальном проекте (аудио Шатри + сгенерённая БД).
- Волна/аудио: длительность из decode совпадает с аудио; пики рисуются.
- Правка границы → `set_time` → перечтение показывает edited.

## Блокеры

- pywebview-окно требует GUI-сессии (на маке есть через дисплей); сам сервер/API
  тестируется headless через HTTP.

Docs: STATUS.md обновить по завершении.
