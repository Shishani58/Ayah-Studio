# Аудит способов использования моделей

## Цель и скоуп
По уточнению пользователя проверить фундаментальные способы применения основной
Quran Phonetics и дополнительных моделей: обучение и входной контракт,
представление текста, доступные головы/декодеры, режимы alignment и выходы.
Не сводить исследование к отдельным отказам FAST. Production/defaults/БД не менять.

## Прочитано
- AGENTS/CLAUDE и RTK: локальность FAST, сохранность текста/ручных данных.
- project.md: pipeline, FULL windows, модельные runtime.
- STATUS и model-reading-audit: отделить новую проверку от старых опытов.
- fast-systematic-execution/adapter-diagnosis.md: прежние QPC context/OOV/int8 опыты.
- asr-model-selection/report.md: RNN-T уже испытывался отдельно от production CTC.
- Исходники emissions, align targets, full_windows, fast_adapters, model_worker.
- Первичные карточки QPC/Tarteel/FastConformer, документация Muaalem,
  Transformers, NeMo Forced Aligner, TorchAudio и исследование CTC label priors.

## Решения
Различать: ошибка интеграции; используемая возможность; неиспользуемая
возможность без доказанного выигрыша; другая постановка задачи; ограничение обучения.
Карточка модели и наличие API не доказывают улучшение на Коране.
Новые локальные пробы QPC исследуют распределение выходов/разделитель, не
качество границ. Входы фиксированы без reference; не читать моделью всю запись.
Сохранять source hashes, параметры, фактические интервалы и результаты.

## Чек-лист
- [x] Инвентаризация моделей и действующих путей.
- [x] Сопоставление с первичными источниками.
- [x] Локальная проверка QPC и экспортного контракта (input parity; logits parity не заявляется).
- [x] Матрица возможностей и приоритет опытов по способу применения.
- [x] Проверка артефактов, SHA, документация.

## Ограничения
Не получен независимый word/phoneme timing GT; новые runtime не подключать
и не объявлять неиспытанные методы улучшением. Закрытые веса не скачивать.

## Проверка и итог
Шесть завершённых QPC calls (4 real/2 zero) плюс один прежний вызов до ошибки
telemetry property: сохранён отдельный dispatch. Входы официального feature
extractor побайтово совпали на2 PCM. Для Muaalem показаны3 представления одного
канонического аята при разном madd, без acoustic inference. Ruff чистый.
Рабочие7 source SHA до/после прежние; полной suite/слуховой приёмки нет.
Выявлены оси проверки, а не доказанное улучшение production.

Docs updated: task.md, report.md, docs/STATUS.md, probe scripts/JSON/metadata.
Docs unchanged: docs/project.md — pipeline/defaults/production не менялись.
