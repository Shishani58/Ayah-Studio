# FULL по сильным паузам: 20 секунд ×3 по умолчанию

## Цель

Сохранить итог всего обсуждения/замеров в `docs/full-performance.md` и
перевести штатный FULL основной ONNX Quran Phonetics на рекомендованные
strong20×3. Пользователь явно разрешил включение, ранее путь был экспериментом.
Без изменения FAST, выбранной модели, формата/пользовательских БД и без commits.

## Прочитано

- `AGENTS.md`, `RTK.md`, `docs/project.md`: инварианты, пайплайн и docs workflow.
- `align.align_surah`, `_OnnxBundle`, `_emissions`, `bundle_source`,
  `align_tiled.align_surah_tiled`, `pipeline._full_align`: короткий/длинный FULL
  и отдельные вызовы FAST/tiled. Не менять общий `_emissions`, используемый FAST.
- `alignment_result.result_source`, `model_store.load_emitter`: параметры
  выполнения в сохранённом source, фактически выбранный emitter.
- Задачи full-pause-windows, full-strong-pause-matrix, full-15-seconds,
  full-10-seconds: измерения, рамки доказательств, предыдущие ограничения.

## Решения

- Выбор20×3 — итоговая рекомендация/прямое указание пользователя, не утверждение
  абсолютного минимума времени среди всех шумных измерений.
- Только настоящий QPC `OnnxEmitter` с phonetic targets. Другие модели/torch
  сохраняют собственный путь; нельзя переносить сетку QPC в Whisper/NeMo.
- Отдельный FULL helper вызывается в short/exact FULL и bounded FULL CTC;
  явно выбранный tiled и FAST не используют новую политику автоматически.
- Все внутренние cuts только в принятых строгих signal-паузах>=300мс с DSP
  gate;20с — ориентир, context2,batch1, нормализация всей записи, workers3.
- Сохранять реальные windows/паузы/параллельность в trace и параметры в source.
- Для production отказ до ONNX, если непрерывный участок между подходящими
  паузами>60с: не разрезать речь скрыто и не отправлять многоминутный участок
  в QPC с риском чрезмерной памяти. Benchmark без этого ограничения сохраняется.
- Сохранение edited/verified остаётся прежним. Отдельно проверить отмену,
  ограничение памяти, сохранённые flags/source и фактический real FULL.
- При реальной проверке исходный strong20×3 снова потерял предупреждения83:10
  и95:8. Добавлены изолированные от FAST консервативные FULL checks:
  min token confidence<.05; либо onset внутри принятой паузы>=700мс позже
  центра на>200мс. Времена/confidence не меняются, ID аятов/эталон не используются.

## Проверка

- [x] Сводный документ всех экспериментов и рекомендаций со ссылками.
- [x] FULL20×3 для short/bounded, FAST/tiled/другие модели изолированы.
- [x] Unit/integration: routing, provenance, cancel/error/long-gap, manual rows.
- [x] Реальные42 и известные83/95; качество и предупреждения документированы.
- [x] Ruff, релевантные тесты, STATUS/project/task, diff-check.

## Результат

Сводка всех обсуждавшихся вариантов и их ограничений: `docs/full-performance.md`.
Штатный QPC FULL20×3 подключён в `align.align_surah` для обычного и bounded CTC.
Общий emitter/FAST/явный tiled/другие модели не переключены. Контекст2с,
batch1, global normalization, только strong signal cuts. Trace/source фиксируют
параметры и фактические окна; длинный непрерывный участок>60с отклоняется
до ONNX, не режется скрыто. Ручная разметка/writer schema не менялись.

Нативный read-only pipeline smoke на42/83/95: все53/36/8 аятов строго в порядке.
Число окон48/10/3.42 совпала с прежним20×3 по всем тайм-кодам/confidence.
После дополнительных warnings все большие ошибки этих трёх сур с ревью:
42:1−810мс;83:10+825,83:13+924,83:31+823мс;95:8+840мс.
Оба прохода до/после warnings дали одинаковые времена/confidence; изменены
только флаги. Добавлено ревью для42:5/8/11/20/32/40/46/51,
83:8/10/15/16/18/33/34,95:1/7/8
(точный список —verification.json). Время pipeline соdecode42≈34.86с,
83≈7.13с,95≈1.66с; это smoke, не новый сравнительный benchmark.

Проверка:772 passed/7 skipped полный suite,217 целевых перед ним; Ruff;
native pipeline42/83/95; отдельно opt-in real FULL112/114:2 passed/5 deselected
(max error<=150мс), diff-check. Известные предупреждения восстановлены,
полная калибровка114 сур/других чтецов не заявляется. Артефакты:
`production-smoke.json`, `production-smoke-reviewed.json`, `verification.json`.
Блокеров выполнения задачи нет. Не коммитили/не пушили.
Docs updated: `docs/full-performance.md`, `docs/project.md`, `docs/STATUS.md`, этот task.
