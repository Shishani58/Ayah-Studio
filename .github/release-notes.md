## Ayah Studio 0.1.0

Разметка аудио Корана по аятам — локально на вашем компьютере.

- Автоматические тайм-коды, волна и проверка на слух.
- Правка границ с сохранением ручной разметки.
- Поиск сур, фильтры и экспорт SQLite.

**Скачайте `Ayah-Studio-0.1.0-macOS-arm64.dmg` ниже** и перенесите приложение
в **Applications / Программы**. Для Mac с Apple Silicon. Python и отдельные
модели устанавливать не нужно. Файл `.sha256` содержит контрольную сумму.

Сборка без нотарификации Apple; при блокировке запуска следуйте
[инструкции Apple](https://support.apple.com/ru-ru/102445).
Автоматические тайм-коды проверяйте на слух перед публикацией.

**Windows x64:** скачайте `Ayah-Studio-0.1.0-Windows-x64.zip`, распакуйте
полностью и запустите **Ayah Studio.exe**. Папка `_internal` должна оставаться
рядом. Для нативного окна нужен [Microsoft WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).
EXE без цифровой подписи. Проверены модели, FFmpeg и запуск нативного окна
на Windows runner ([сборка](https://github.com/Shishani58/Ayah-Studio/actions/runs/38291457116), commit `ee1cbc0`).

---

**Apple Silicon macOS app.** Download the DMG below, drag Ayah Studio to
Applications and launch it. All runtime models are bundled. The interface is
in Russian. This build is not Apple-notarized; see
[Apple’s guidance](https://support.apple.com/en-us/102445) if the first launch
is blocked. Review generated timings by ear before publishing.

**Windows x64:** extract the entire ZIP and run **Ayah Studio.exe**. Keep `_internal` next to the EXE. Requires WebView2 Runtime. The EXE is not code-signed.
