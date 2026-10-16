# Проверка способов чтения аудио моделями

## Цель и скоуп
Проверить, ограничивает ли качество FAST выбранный способ применения моделей:
подача локального контекста, decoding, использование известного текста и поиск
опор. Исследование, без изменения production, defaults и пользовательских БД.

## Прочитанные документы
- AGENTS/RTK, project.md, STATUS.md: инварианты и текущий путь.
- fast-full-execution и fast-review-fixes/report.md: парный legacy baseline.
- fast-systematic-research/research.md и fast-systematic-execution/{report,
  adapter-diagnosis}.md: проверенные и отвергнутые объяснения.
- fast-production-improvement/task.md: параллельная работа и выбор Base.
- Официальная документация Transformers Whisper, CTC segmentation, WhisperX:
  другие способы inference/alignment и их ограничения.

## Решения
- Старую и текущую whisper_local сравнить через AST; совпадение не доказывает
  равенство всех загруженных весов/конфигурации/окон.
- Новые inference только Tarteel Base, локальные фиксированные участки.
- Сравнить greedy, beam и оценку заданных конкурирующих текстов на одинаковом
  encoder output. Последняя — diagnostic score, не proof и не timing.
- Контроли: два размера контекста и нулевое аудио. Не подбирать параметры
  production по этим примерам. Входные интервалы, SHA, calls и время сохранить.
- Исторические результаты явно отличать от свежих измерений. Слуховой приёмки нет.

## Проверка
- [x] Сопоставлены старый поиск опор и текущий executor, AST whisper_local.
- [x] Выполнены локальные decoding/contrastive пробы и контроли:24encoder evaluation.
- [x] Сформулированы доказанные ограничения и альтернативы; n-best80:1 положительный.
- [x] Docs updated: task.md, probe.py, beam_probe.py, результаты, report.md;
  docs/STATUS.md. Docs unchanged: docs/project.md — production прежний.

## Блокеры
Нет независимой ручной разметки physical onset/cut; эксперимент оценивает
способы получения текста, не доказывает точность времени.

## Углубление и план по поручению пользователя
- [x] Проверить n-best относительно реальных plan_phrase/quality/continuation
  условий FAST: двухсловный пример80 сам по себе не доказывает потерю proof.
- [x] Отдельно проверить провалы frozen baseline и другие записи/чтецов на тех же PCM.
- [x] Зафиксировать типы отказов из transitions, не считать весь blocked хвост
  независимыми ошибками модели.
- [x] Проверить стоимость/качество чтения beam и различие canonical/ranked score.
- [x] Подготовить plan.md: этапы, файлы, причинные A/B, явные критерии,
  ограничение FAST, проверка регрессии legacy и независимый временной эталон.

Итог углубления:37новых encoder evaluation с greedy/beam,4чтеца; найден
адресный путь2:5 и identity-read2:4.454из477неподтверждённых frozen Base
строк являются blocked estimates. Подробности/ограничения findings-deep.md,
полная программа реализации plan.md. Это завершённое исследование и план,
не перенос прототипа. Исходные16audioSHA и3productionSHA неизменны.
Первая prepare попытка завершилась до inference на пустом списке historical
target reads; добавлена явная skipped запись Миншауи98:2, crop не подставлялся.
Ruff всех скриптов чистый, analyse replay37/37; pytest не запускался,
production прежний. Docs updated: findings-deep.md,plan.md,report.md,task.md,
docs/STATUS.md и исследовательские артефакты. Docs unchanged: project.md.
