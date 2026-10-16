# Компактные фильтры разметки

Цель: убрать постоянный блок фильтров, счётчика и пояснения над аятами;
собрать их в меню одной кнопки верхней панели.

Прочитано: STATUS.md (актуальные изменения интерфейса), task.md stable-workspace-header
(геометрия панели), web-ui-ux (существующие компоненты и проверка браузера).
Пайплайн и данные не затрагиваются; чужие изменения сохраняются.

Решение: переиспользовать toolbar-menu и обработчики меню; сохранить все
фильтры, сортировку, переход к следующей метке и пояснение внутри меню.

- [x] Перенести блок и оформить меню.
- [x] Проверить фильтрацию, клавиатуру и адаптивность.
- [x] Обновить STATUS и результаты проверки.

Блокеров нет.

Проверка: node tests/webgui_async.cjs — passed; pytest test_webgui_localization.py
— 19 passed; node --check app.js и scoped diff --check — OK; AGENTS/CLAUDE идентичны.
Chrome/Playwright fixture с настоящими HTML/CSS и функциями wireWorkspace,
applyTimingReviewFilter, positionWorkspaceMenu: 1440/1100/820/600/390/320 px.
Проверены границы меню, фильтр, Tab, Escape с возвратом фокуса и клик снаружи.
Визуально просмотрены desktop и узкое окно. Реальный pywebview не запускался.

Изменены: static/index.html, app.js, studio.css, i18n.js.
Docs updated: docs/STATUS.md, этот task.md.
