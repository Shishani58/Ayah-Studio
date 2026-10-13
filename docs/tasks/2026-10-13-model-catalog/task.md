# Каталог моделей

Цель: вкладка моделей, лицензии, фоновые проверяемые загрузки, сохранённый выбор поддерживаемой модели.

Прочитано: docs/project.md (forced alignment, ручные метки), docs/STATUS.md (ONNX), core align/emissions/word_align/pipeline, настройки webgui; карточки и manifests Hugging Face.

Решения: сохранять revision и SHA256; загрузка атомарно, исходные веса не перезаписывать; gated доступ по токену без сохранения; несовместимые модели показывать честно, не заменять молча. Арабский CTC — экспериментальный полный/пословный режим, без FAST эвристик фонетики.

- [x] Каталог и загрузки
- [x] Выбор и runtime для Quran Phonetics INT8/FP32 и Arabic CTC; остальные adapters не реализованы
- [x] Вкладка: по обратной связи компактные строки, без лицензионных текстов/галочек
- [x] Тесты и визуальная проверка

Блокеры: отдельные runtime адаптеры Whisper/NeMo/Muaalem/Zipformer/MMS; gated доступ Quran-Lab.

Проверка: pytest 338 passed; ruff; загрузка Arabic (356182585 bytes, SHA256 совпадает), inference 49×51; словарь всех сур; FULL Шатри114 6 меток и words114:1 4 слова без записи в БД. FP32 smoke 49×40. Переключение через API/UI на временных настройках. Скриншот models-desktop.png. Пользовательские настройки выбора и БД не менялись.

Docs updated: docs/project.md, docs/STATUS.md, этот task.md. AGENTS.md/CLAUDE.md не менялись и идентичны.

Остаток исходного запроса: переключение MMS, Muaalem, NeMo FastConformer, Whisper, streaming Zipformer потребует отдельных адаптеров; для Zipformer внешний gated доступ. Эти ограничения явно показаны статусом.
