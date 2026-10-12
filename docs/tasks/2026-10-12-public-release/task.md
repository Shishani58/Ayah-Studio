# Публичный релиз Ayah Studio

## Цель
Запушить все накопленные изменения, оформить короткий README по примеру
Fluttios и опубликовать релиз с готовым macOS DMG.
Пользователь явно выбрал публичный GitHub вместо закрытого GitLab.

## Прочитано
- AGENTS.md, RTK.md — правила, разрешение пользователя на commit/push.
- docs/development/packaging.md, .github/workflows/build.yml, .gitlab-ci.yml — сборка.
- Fluttios README и release-notes — образец короткой страницы со скачиванием.

## Решения
- Новый публичный GitHub-репозиторий Ayah-Studio в аккаунте владельца Fluttios.
- Старый GitLab remote сохранить; основным сделать GitHub после создания.
- DMG публиковать как release asset, не в Git. Исходники моделей/проекты пользователя
  и папки build/dist не коммитить.
- Изображения публичной страницы без локальных путей пользователя.

## Чек-лист
- [ ] README, screenshot, release notes
- [ ] Тесты, финальная сборка DMG
- [ ] Репозиторий, commit/push, tag
- [ ] Release и проверка скачивания DMG

## Проверка / блокеры
В процессе.
