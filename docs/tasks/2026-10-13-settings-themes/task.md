# Темы и настройки

Цель: выбор светлой/тёмной темы с сохранением и ясная группировка настроек.
Скоуп: HTML/JS GUI, валидация theme, API/UI проверки.
Прочитано: webgui settings/server, HTML/CSS/JS, STATUS.md, task boundary-studio,
web-ui-ux, better-layout/better-colors: существующее поведение и визуальные правила.
Решения: тёмная тема по умолчанию; preview до сохранения, отмена/Escape
возвращают сохранённый вид; разделить интерфейс и генерацию.
Архитектура/пайплайн/формат БД не меняются. Локальные изменения сохраняются.

- [x] Реализация.
- [x] Проверка API, сохранения, отмены, обеих тем и размеров.
- [x] Документация.
Блокеры: нет.


Проверка: полный pytest — 313 passed (40 webgui_server/desktop_projects), Ruff, node --check,
git diff --check проходят. 8 регрессионных проверок theme: сохранение обеих тем,
сохранность прочих настроек, default dark, отклонение неверных значений.
В реальном FastAPI с отдельным /tmp settings.json и временной SQLite/WAV:
preview, отмена и Escape, save/reload темы и 36px, вкладки разметки/проекта/импорта;
320×900 без горизонтального overflow (dialog 286px), 1280×820 обе темы и review.
Ошибок в консоли нет. Скриншоты light.png/dark.png.
Контраст объявленных токенов: text/panel ≥14.45, muted/panel ≥6.33,
muted/panel2 ≥5.54, primary/accent ≥4.70, wave/background ≥5.24.
Не проверены: native WebView Windows/macOS, VoiceOver, готовая DMG/EXE.
Пользовательские настройки/БД не записывались; коммит/пуш не выполнялись.
Docs updated: docs/project.md, docs/STATUS.md, этот task.md.


## Коррекция светлой палитры
По запросу пользователя холодные бело-серые цвета заменены тёплой бумажной
палитрой: крем/беж, графит, бронзовый акцент и приглушённая зелёная волна.
Меняются только light CSS tokens и миниатюра темы; тёмные токены прежние.
Проверено в браузере: настройки и рабочий экран на временных WAV/SQLite;
сохранение темы, console без ошибок. Контраст text/panel 12.67:1,
muted/panel 5.64:1, muted/panel2 4.57:1, primary/accent 5.64:1,
wave/bg 4.89:1. Граница управления затемнена до контраста >3:1.
Скриншоты light-warm-settings.png и light-warm-review.png.
git diff --check проходит. Тесты повторно не запускались: только цвета CSS.
Docs updated: docs/STATUS.md и этот task.md. Блокеров нет.
