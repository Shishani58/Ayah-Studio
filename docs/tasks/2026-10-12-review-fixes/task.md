# Исправления по ревью проекта

## Цель и скоуп

Исправить пять находок ревью: безопасный экспорт, проверка состава и времени,
изоляция фоновой генерации, продолжение частичной разметки, модели для CI.
Существующие изменения FAST и Ayah Studio сохраняются. Без коммита и пуша.

## Прочитано

- AGENTS.md / CLAUDE.md и RTK.md — правила работы.
- docs/project.md — инварианты данных и пайплайна.
- docs/tasks/2026-10-12-project-review/task.md — воспроизведения ошибок.

## Решения

- Экспорт, сервер и CI выполняются отдельными агентами в разных файлах.
- Миграция схемы не нужна. Проверки используют временные БД.
- Исправления существующих пользовательских файлов только по месту.
- Экспорт читает consistent read-only snapshot и пишет временную БД рядом
  с назначением; `os.replace` выполняется только после успешного commit.
  `strict=False` оставлен для явного частичного экспорта, но не обходит
  защиту исходного файла.
- Задание получает неизменный снимок audio_map и пути БД; логи используют
  этот же путь. Open/generate сериализованы lock; частичные суры определяются
  по точному набору аятов. Draft/correct также фиксируют контекст до расчёта.
- CI экспортирует acoustic-модель из pinned HF revision, Silero скачивает
  по pinned URL с SHA256. Один Linux artifact используется обеими сборками.
  Готовый ZIP из AYAH_MODELS_URL остаётся optional, а не обязательной настройкой.
- Существующие переименование Ayah Studio, DMG, uv и пользовательские FAST
  изменения сохранены. AGENTS/CLAUDE синхронно уточнены под фактический стек;
  README больше не обещает NN только на коротких окнах.

## Проверка

- [x] 24 теста экспорта: source aliases, atomic failure, canonical IDs, valid times.
- [x] 10 новых server-регрессий: snapshot БД/аудио/логов, защита старта,
  продолжение partial, сохранение edited/verified, failed open без изменения STATE.
- [x] 3 теста model preparation: checksum, pinned export, failure сохранность.
- [x] Общий pytest: 252 passed. Ruff, `git diff --check`, `cmp AGENTS.md CLAUDE.md`.
- [x] `node --check` для app.js; отдельный Node smoke: processed + partial
  оставляет активную кнопку «Продолжить · осталось 1». Фильтр partial добавлен.
- [x] Read-only ревью server вторым агентом; итоговый diff проверен root.
- [x] Runtime `--self-check`: 114/6236, ONNX inference, FFmpeg, assets OK.
- [x] Оба workflow разобраны YAML parser; remote CI не запускался.
- [x] Реальный `python -m tools.prepare_models --output /tmp/surah-prepared-models`
  в отдельном Python 3.12 env с `tools/model-export-requirements.txt` завершился.
  ORT 1.20.1: acoustic outputs `(1,49,40)` / `(1,99,40)` для 16k/32k samples,
  все числа конечны; Silero inference также прошёл. Словарь 40 токенов совпал
  с существующим; Silero SHA256 совпал. Существующие `models/` не заменялись.

Новый acoustic ONNX: 121936019 байт, SHA256
`cdb047a63a443a653fddbecc2281067423092b69144ab40e2b41438516f7db94`.
Это результат конкретного smoke, не обещание битовой идентичности между ОС.
На двух синтетических сигналах argmax совпал с локальной моделью, но это
не заменяет оценку точности на реальной записи.

## Блокеры

Для пяти исправлений нет. Удалённые CI/macOS/Windows сборки в этой задаче
не запускались; GitLab native runners не настроены. Отдельная открытая задача
FAST 2:145 и полный визуальный GUI-прогон не входят в эти пять исправлений.

Docs updated: `AGENTS.md`, `CLAUDE.md`, `README.md`, `docs/project.md`,
`docs/STATUS.md`, `docs/development/packaging.md`, этот task-файл.
