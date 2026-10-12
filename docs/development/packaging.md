# Ayah Studio — сборка и установка

## Установка пользователем

Открыть `Ayah-Studio-0.1.0-macOS-arm64.dmg`, перенести `Ayah Studio.app`
на ярлык `Applications`, извлечь диск и запустить приложение из «Программ».
Python, исходники, терминал и скачивание моделей пользователю не нужны.
Все вычисления выполняются локально. Проекты сохраняются вне .app.

## Сборка разработчиком (macOS)

```bash
uv run --locked --extra build python tools/build_macos.py
```

Скрипт собирает PyInstaller `.app`, запускает её `--self-check`, проверяет подпись,
создаёт сжатый DMG с ярлыком `/Applications`, проверяет образ и записывает SHA-256.
`--skip-build` упаковывает уже собранную `.app`; для изменённого кода нужен полный
запуск. Версия берётся из Info.plist собранного bundle.

Результат: `dist/Ayah Studio.app`, `dist/Ayah-Studio-0.1.0-macOS-arm64.dmg`
и `.dmg.sha256`. Архитектура совпадает с Python сборочной машины: arm64 или
x86_64. Универсальная сборка и запуск на Intel в этой задаче не проверялись.
Зависимости зафиксированы в `uv.lock`; не включать extras `align` / `qt` в сборку.

Без uv: установить проект с extra `build`, затем выполнить
`python tools/build_macos.py`. Такой путь не использует lock зависимостей.

## Что включено

`surah_cutter.spec` упаковывает web-GUI, текст Корана (114 сур, 6236 аятов),
эталонные тайминги, QPC Hafs, FFmpeg, ONNX Runtime, словарь и обе модели.
Нужны локальные файлы (не коммитятся):

- `models/wav2vec2-quran-phonetics.onnx` — int8 acoustic model;
- `models/wav2vec2-quran-phonetics.onnx.vocab.json`;
- `models/wav2vec2-quran-phonetics.onnx.meta.json`;
- `models/silero_vad.onnx`.

При отсутствии файла сборка останавливается. FP32, torch и Qt не входят.
`data/`, `assets/`, `models/` находятся рядом с Python-пакетом внутри bundle:
этот layout используется runtime-загрузчиками, менять его нельзя.
`--self-check` не открывает пользовательский проект: проверяет число аятов,
статику/шрифт, загрузку моделей, реальный ONNX inference и запуск FFmpeg.

Иконка `assets/AyahStudio.icns` включена в репозиторий. Пересоздание:
`swift tools/create_icon.swift assets` (macOS AppKit + iconutil).
Имя Python-пакета — `ayah-studio`, команды — `ayah-studio` и `ayah-studio-cli`.
Внутренний namespace `surah_cutter`, старые CLI и `~/.surah-cutter/settings.json`
сохранены для совместимости с существующими проектами.

## Подпись и публичная раздача

Локальная сборка имеет ad-hoc подпись. Она не равнозначна Developer ID и не
гарантирует прохождение Gatekeeper на чужом Mac. Нужны сертификат
**Developer ID Application** в Keychain и профиль notarytool:

```bash
AYAH_CODESIGN_IDENTITY="Developer ID Application: …" \
AYAH_NOTARY_PROFILE="ayah-studio-notary" \
uv run --locked --extra build python tools/build_macos.py
```

PyInstaller подписывает вложенные бинарники; скрипт отправляет DMG в Apple,
прикрепляет билет и проверяет его. Секреты не записываются в репозиторий.
В этой задаче Developer ID отсутствовал; нотарификация и чистый Mac не проверены.
См. [PyInstaller](https://pyinstaller.org/en/stable/feature-notes.html#macos-binary-code-signing)
и [Apple](https://developer.apple.com/documentation/xcode/creating-distribution-signed-code-for-the-mac/).

## CI

`.github/workflows/build.yml` сначала запускает pytest/Ruff, затем готовит
модели в Linux-job и передаёт один artifact обеим платформам для сборки
macOS DMG + checksum и Windows ZIP. По умолчанию модели собираются из
зафиксированной Hugging Face revision и Silero URL с SHA256
(`tools/model_sources.json`); версии экспортера указаны в
`tools/model-export-requirements.txt`. Для подготовки локально используйте
отдельное Python 3.12 окружение, чтобы не менять runtime-зависимости:

```bash
python3.12 -m venv .venv-model-export
.venv-model-export/bin/python -m pip install -r tools/model-export-requirements.txt
.venv-model-export/bin/python -m tools.prepare_models
```

На Windows Python находится в `.venv-model-export/Scripts/python.exe`.
Скрипт сначала завершает загрузку и экспорт во временной папке, затем копирует
четыре runtime-файла в `models/`; FP32-экспорт не сохраняется. `--output PATH`
позволяет проверить рецепт без замены существующих моделей. Битовая идентичность
ONNX между ОС не гарантируется; CI передаёт один Linux-артефакт всем сборкам.

Опционально владелец может задать `AYAH_MODELS_URL` (HTTPS ZIP) и
`AYAH_MODELS_SHA256` (repository variables): тогда вместо экспорта скачивается
готовый архив. ZIP должен содержать четыре файла прямо в корне;
`tools/fetch_runtime_models.py` извлекает только эти имена после проверки checksum.
Для пути по умолчанию настройка внешнего архива не требуется.

Основной публичный репозиторий: [Shishani58/Ayah-Studio](https://github.com/Shishani58/Ayah-Studio).
GitHub Actions запускает проверки на main/PR и сборку на теге `v*`.
Release получает DMG, SHA-256, Windows ZIP и короткое описание из
`.github/release-notes.md`. Перед новым релизом обновите версию в
`pyproject.toml`, `surah_cutter/__init__.py`, `surah_cutter.spec`, README и release notes.

Старый GitLab remote сохранён как `gitlab`. Его `.gitlab-ci.yml` запускает
pytest/Ruff и создаёт artifact моделей для tag/manual pipeline, но нативные
macOS/Windows runners там не настроены. Для локального DMG CI не нужен.

## Лицензии

В bundle включён LICENSE. Данные, шрифт и модели сохраняют лицензии источников;
перед публичным релизом необходимо проверить условия их распространения.
