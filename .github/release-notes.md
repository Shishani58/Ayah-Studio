## Ayah Studio 0.3.3

Разметка аудио Корана по аятам и словам — локально на вашем компьютере.

- FULL: исправлены границы при слабом начале речи и перекрывающихся окнах.
- Исправлена обработка одинаковых CTC-оценок и порядок параллельных окон.
- GUI сохраняет активность во время генерации; улучшена очистка ресурсов декодера.
- Удалены устаревшие коррекции FAST и лишняя основная зависимость soundfile.
- Сохранены локальный FAST, выбор моделей, ручные и проверенные метки.

**macOS Apple Silicon:** скачайте `Ayah-Studio-0.3.3-macOS-arm64.dmg`
и перенесите приложение в **Applications / Программы**.

**Windows x64:** скачайте `Ayah-Studio-0.3.3-Windows-x64.zip`, распакуйте
целиком и запустите **Ayah Studio.exe**. Папка `_internal` должна оставаться
рядом. Нужен [Microsoft WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).

Python устанавливать не нужно. Базовая модель включена; дополнительные
модели скачиваются отдельно по выбору. Доступность режима зависит от выбранной модели.
Файлы `.sha256` содержат контрольные суммы.

Автоматические тайм-коды проверяйте на слух. Сборка macOS без нотарификации
Apple ([инструкция запуска](https://support.apple.com/ru-ru/102445)); Windows EXE без цифровой подписи.

---

**Ayah Studio 0.3.3:** improved FULL boundaries for weak speech onsets and
overlapping windows, deterministic CTC ties and parallel window ordering,
generation activity handling and simpler audio decoding. Manual and verified
timings survive regeneration.

**macOS Apple Silicon:** open the DMG and drag Ayah Studio to Applications.
**Windows x64:** extract the complete ZIP and launch Ayah Studio.exe; keep
`_internal` alongside it. WebView2 Runtime is required. Python and the default
model are bundled; optional models download separately. The interface supports Russian and English.
These builds are not code-signed with distribution certificates. Review timings by ear.
