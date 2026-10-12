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
- [x] README, screenshot, release notes
- [x] Тесты, финальная сборка DMG
- [x] Репозиторий, commit/push, tag
- [x] Release и проверка скачивания DMG

## Проверка / блокеры
- Исходное состояние: 252 теста passed, Ruff без ошибок.
- Из GitLab добавлен ранее не полученный коммит 3d36fbf. Три конфликта
  состояли из добавлений удалённой ветки; они сохранены вместе с локальными
  исправлениями. После слияния: 265 passed, Ruff и diff check — OK.
- GitHub-репозиторий Shishani58/Ayah-Studio создан публичным по выбору пользователя.
- GitLab remote сохранён, GitHub — origin. Нотарификации пока нет.

- GitHub checks переведены на uv.lock после расхождения с pip latest;
  запуск 38290160435 завершился success.
- Релизный DMG пересобран после merge; self-check ресурсов/моделей, codesign
  и hdiutil verify — OK. SHA-256:
  `cecbebc2f20a22e2bf8bb4e01241962dd83f9b0b8f334ed930a51281232e22c2`.

- Релиз v0.1.0 опубликован: https://github.com/Shishani58/Ayah-Studio/releases/tag/v0.1.0
- DMG скачан целиком без авторизации: 171922782 байта; SHA-256 совпал
  с локальным файлом и опубликованной контрольной суммой.
- Публичная страница визуально проверена; `published-release.jpg` — подтверждение.
- Параллельные изменения задачи waveform-icon появились после фиксации релиза;
  в этот релиз и его коммиты не включались.

Docs updated: README.md, README.en.md, docs/attributions.md, docs/development/packaging.md, docs/STATUS.md, этот task-файл.
