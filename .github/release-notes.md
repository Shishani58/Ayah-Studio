## Ayah Studio 0.3.1

Разметка аудио Корана по аятам и словам — локально на вашем компьютере.

- Единый FAST: поиск пауз и локальные проверки границ с журналом прочитанных участков.
- Улучшены проверки первого слова и разделённых ASR-слов; повторно используется RMS аудио.
- FULL: локальная проверка границ, выбор центра паузы и возобновление расчёта.
- Оценка уверенности начала аята и подсказки для ручного ревью.
- Выбор ASR-моделей, русский и английский интерфейс.
- Ручные и проверенные метки сохраняются при перегенерации.

**macOS Apple Silicon:** скачайте `Ayah-Studio-0.3.1-macOS-arm64.dmg`
и перенесите приложение в **Applications / Программы**.

**Windows x64:** скачайте `Ayah-Studio-0.3.1-Windows-x64.zip`, распакуйте
целиком и запустите **Ayah Studio.exe**. Папка `_internal` должна оставаться
рядом. Нужен [Microsoft WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).

Python устанавливать не нужно. Базовая модель включена; дополнительные
модели скачиваются отдельно по выбору. Доступность режима зависит от выбранной модели.
Файлы `.sha256` содержат контрольные суммы.

Автоматические тайм-коды проверяйте на слух. Сборка macOS без нотарификации
Apple ([инструкция запуска](https://support.apple.com/ru-ru/102445)); Windows EXE без цифровой подписи.

---

**Ayah Studio 0.3.1:** unified local FAST with improved head-word checks,
FULL boundary verification and pause centering, onset confidence review,
ASR model selection and Russian/English interface. Manual and verified timings
survive regeneration.

**macOS Apple Silicon:** open the DMG and drag Ayah Studio to Applications.
**Windows x64:** extract the complete ZIP and launch Ayah Studio.exe; keep
`_internal` alongside it. WebView2 Runtime is required. Python and the default
model are bundled; optional models download separately. The interface supports Russian and English.
These builds are not code-signed with distribution certificates. Review timings by ear.
