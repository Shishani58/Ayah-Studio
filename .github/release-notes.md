## Ayah Studio 0.2.0

Разметка аудио Корана по аятам и словам — локально на вашем компьютере.

- Обновлённый редактор: навигация по сурам, компактные меню времени и статуса.
- Светлая и тёмная темы, настройки с автосохранением, список проектов.
- Скорость, остановка после аята/слова и повтор в плеере.
- Остановка разметки прерывает текущий расчёт, сохраняя завершённые результаты.
- FAST ищет паузы и проверяет небольшие локальные участки; скрытого полного прохода нет.
- FULL и разметка слов; каталог Quran Phonetics, Muaalem, FastConformer Arabic, Whisper Large-v3.
- Ручные и проверенные метки сохраняются при перегенерации.

**macOS Apple Silicon:** скачайте `Ayah-Studio-0.2.0-macOS-arm64.dmg`
и перенесите приложение в **Applications / Программы**.

**Windows x64:** скачайте `Ayah-Studio-0.2.0-Windows-x64.zip`, распакуйте
целиком и запустите **Ayah Studio.exe**. Папка `_internal` должна оставаться
рядом. Нужен [Microsoft WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).

Python устанавливать не нужно. Базовая модель включена; дополнительные
модели скачиваются отдельно по выбору. FAST доступен для базовой модели.
Файлы `.sha256` содержат контрольные суммы.

Автоматические тайм-коды проверяйте на слух. Сборка macOS без нотарификации
Apple ([инструкция запуска](https://support.apple.com/ru-ru/102445)); Windows EXE без цифровой подписи.

---

**Ayah Studio 0.2.0:** updated ayah/word editor, light/dark themes, project
management, playback controls and optional alignment models. Fast mode uses
pause detection and local acoustic checks. Manual/verified timings survive regeneration.

**macOS Apple Silicon:** open the DMG and drag Ayah Studio to Applications.
**Windows x64:** extract the complete ZIP and launch Ayah Studio.exe; keep
`_internal` alongside it. WebView2 Runtime is required. Python and the default
model are bundled; optional models download separately. The interface is in Russian.
These builds are not code-signed with distribution certificates. Review timings by ear.
