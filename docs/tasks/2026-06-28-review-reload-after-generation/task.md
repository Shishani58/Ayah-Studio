# Review reload after generation

## Цель

После общей генерации, если пользователь держит открытую рассчитанную суру в
review-экране, сразу подтягивать свежие тайм-коды из БД без ручного
переоткрытия суры.

## Скоуп

- `surah_cutter/webgui/static/app.js`: обработчик SSE-события `sura`.
- Не менять алгоритм выравнивания, формат БД и серверный write-path.

## Прочитанные доки и код

- `AGENTS.md` из сообщения пользователя: обязательные правила работы.
- `surah_cutter/webgui/server.py`: событие `sura` отправляется после
  `db.write_timings(...)`.
- `surah_cutter/webgui/static/app.js`: текущий batch handler обновляет только
  dashboard через `refreshSuras()`, а per-sura handler уже делает `reloadSura()`.

## Решение

- В `handleGenEvent` на событии `sura` вызвать `reloadSura()`, если открыта та
  же сура.
- Оставить `refreshSuras()` для dashboard-статусов.

## Чек-лист

- [x] Внести точечную правку frontend.
- [x] Проверить JS-синтаксис.
- [x] Зафиксировать итог документации/проверки.

## Проверка

- `node --check surah_cutter/webgui/static/app.js` — passed.

Docs unchanged: поведение исправлено точечно в UI; пользовательская документация
и формат данных не менялись.

## Блокеры

Нет.
