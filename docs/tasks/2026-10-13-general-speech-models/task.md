# Обычные модели распознавания

Цель: открытые неспециализированные модели с арабским языком: Whisper Small/Medium/Large-v3/Turbo, NVIDIA FastConformer Arabic, MMS-1b-all с Arabic adapter.

Прочитано: model_catalog.json/model_store.py/models.js; официальные карточки OpenAI/NVIDIA/Meta на Hugging Face, manifests Hub; предыдущие задачи каталога и трёх статусов.

Решения: только multilingual Whisper без .en; pin revision/size/hash; один набор весов и сопутствующие tokenizer/config/Arabic adapter, без дублирующих форматов и без всех MMS языковых адаптеров. Новые записи добавляются к актуальному каталогу, сохраняя изменения другого чата.

UI остаётся компактным, с тремя статусами. Runtime новых моделей должен быть подключён отдельным чатом адаптеров; запрошено разрешение передать ему дополнительные модели.

Проверка: manifests, целевые тесты загрузок и браузер.

Готово: 6 новых entries, скачивание через общий менеджер; сохранена текущая схема трёх статусов и иконки.
Проверка: 50 tests (store/server), 2 новых tests (обязательные файлы, открытые repos, pinned hashes, multilingual Whisper и только Arabic MMS adapter), ruff. В браузере 13 строк, новые модели доступны для скачивания; снимок models.png. Большие веса не скачивались; качество на Коране пока не измерялось.

По ответу пользователя «Да, добавить их в работу этого чата» передано расширение задачи в 01a135c3-3389-7d11-a16b-2d1b9ff5d4df через send_message_to_thread. Runtime и UI здесь не редактировались во избежание конфликтов.
Docs updated: docs/STATUS.md, docs/project.md, этот task.md.
