# Tarteel Tiny: локальный FAST

Цель: подключить Tiny к существующему локальному адаптеру Base и измерить
FAST на тех же 37 сурах Шатри 78–114 (564 аята), сравнить с Base и Phonetics.

Прочитаны AGENTS/CLAUDE, RTK, docs/project.md, предыдущая задача tarteel-fast,
адаптер, runtime/worker guards, каталог, API/UI выбор режима, benchmark и тесты.
Serena недоступна (Session not found); discovery через rg и чтение файлов.

Решения: тот же алгоритм/thresholds без подгонки; собственные pinned веса Tiny
и её словарь. DSP/VAD перед моделью, локальные фразы 1.3/2.5/4 с, ≤6 с на вход,
union≤65%, input≤200%, никаких FULL/tiled/whole-recording вызовов. Неподтверждённые
границы — confidence=0 для ревью. Эталон читается только после inference.
Пользовательские данные не меняются; чужие изменения сохраняются, без коммитов.

Чек-лист:
- [x] Разрешить Tiny в адаптере/RPC/worker и API/UI FAST.
- [x] Проверить guards, отсутствие FULL, сохранение выбранного FAST/ручных меток.
- [x] Измерить 37 сур, ошибки, ревью, latency и фактические интервалы.
- [x] Обновить справку и документацию, записать проверки и ограничения.

Блокеры: нет на старте; качество Tiny пока неизвестно.

## Подключение

Tiny разрешена в существующем адаптере, локальном RPC/worker и supports_fast.
Цели строятся с реальным ID выбранной модели, source содержит Tiny revision.
Алгоритм/thresholds v1 не изменены; Base/Phonetics контрольные алгоритмы сохранены.
Выбор Tiny сохраняет fast/full/tiled/words; auto/hybrid по-прежнему отклоняются
до inference. UI разрешает FAST, внешние аятные клипы явно отклоняются.
Guard-тесты и API/selection параметризованы для Tiny/Base: 57 passed.

Все три модели запущены последовательно одним benchmark, первоначальные DSP/VAD
паузы общие. В процессе inference reference DB не читается.

## Результат (завершено 2026-10-14)

37 сур Шатри 78–114, 564 начала аятов, 3184.48 с аудио. Включены все
структурные priors, не только подтверждённые границы. Параметры до оценки
не менялись; эталон открыт после всех трёх inference-прогонов.

| Метрика | Quran Phonetics INT8 | Tarteel Tiny | Tarteel Base |
|---|---:|---:|---:|
| median, мс | 73.5 | 68 | 67.5 |
| p95, мс | 4465.85 | 1757.5 | 1628.2 |
| max, мс | 26762 | 6062 | 6062 |
| ошибки >500 мс | 107 | 72 | 64 |
| метки ревью | 288 | 170 | 153 |
| ошибки >500 без ревью | 6 / 276 | 8 / 394 | 9 / 411 |
| DSP+inference/matcher, с | 133.26 | 243.82 | 448.26 |
| inference/matcher, с | 130.80 | 241.36 | 445.80 |
| load/probe, с | 0.17 | 4.41 | 5.01 |
| decode, с (общий) | 3.35 | 3.35 | 3.35 |
| DSP/VAD, с (общий) | 2.46 | 2.46 | 2.46 |
| вызовы модели | 1650 | 1094 | 1104 |
| покрытие корпуса | 60.87% | 46.35% | 46.44% |
| вход с повторами | 140.37% | 71.61% | 70.98% |
| максимальный вход | 6 с | 4.2 с | 4.2 с |
| max покрытия одной суры | 65% | 57.53% | 57.01% |

Tiny: крупных ошибок на 32.7% меньше, ревью на 41% меньше, чем у Phonetics;
в 1.83 раза медленнее. Почти та же типичная ошибка. По сравнению с Base:
8 дополнительных крупных ошибок, 17 дополнительных ревью, в 1.84 раза быстрее.
На этом наборе Tiny — промежуточный вариант; основной выбор не меняется.
Без ревью Tiny ошибается >500 мс в 8/394≈2.03% случаев; максимальная такая
ошибка 672 мс. Confidence не является гарантией или калиброванной вероятностью.

Парное сравнение: Tiny исправила 76 крупных ошибок Phonetics, внесла 41 новую,
31 общая. По отношению к Base исправила 6, внесла 14 новых, 58 общих.
Результат Phonetics/Base и SHA initial-пауз полностью совпали с предыдущим
прогоном; разница времени — измерение этого запуска, не изменение алгоритма.
Сохранены две прежние метки Phonetics вне аудио (82:18/19, confidence=0);
они входят в ошибки. У Tiny/Base таких нарушений нет.

Tiny: emission_passes=0, 1094 реально прочитанных локальных интервала, denied=0;
не было FULL/tiled/teacher forcing. После benchmark изменена только подпись
логирования «FAST Base» на «FAST Tarteel», без изменений вычислений. Оба SHA
и причина сохранены в post-benchmark-log-change.json; comparison.summary.json
содержит SHA именно измеренного кода, не заменённые задним числом.

## Проверки, файлы и ограничения

- 547 passed, 7 opt-in skipped; после обновления справки 70 целевых passed.
  Ruff всего проекта, JS syntax, diff-check; AGENTS/CLAUDE идентичны.
- Реальный HTTP select/settings/generate/SSE с Tiny на временной .scproj:
  FAST сохранён, full/tiled запрещены spy, edited/verified время и metadata
  сохранены. Источник — local-asr-pause:whisper-tarteel-tiny с pinned revision.
- В интерфейсе проверены активная Tiny, доступный FAST и индивидуальная
  справка с результатами. Ошибок JS нет; снимок ui-tiny-fast.png.
- SHA canonical DB, read-only эталона и пользовательских settings совпали.
  Pinned Tiny веса/processor проверены; пользовательские проекты не менялись.
- Интеграция: align_fast.py, align_fast_whisper.py, fast_reads.py,
  model_runtime.py, model_worker.py, model_store.py; API server.py;
  app.js/index.html/i18n.js/model_catalog.json; test_tarteel_fast.py и
  test_model_runtimes.py. Собственные веса, без новых зависимостей.
- comparison.json — все 111 записи, интервалы и тексты; comparison.summary.json
  — сводка и исходные паузы/SHA; analysis.json — парные ошибки и бюджеты;
  reproducibility.json — модели, окружение и protected SHA;
  api-validation.json — настоящая интеграция. analyze.py воспроизводит анализ.

Блокеров нет. Проверен один чтец и суры 78–114; другие чтецы, 1–77,
Windows/Qt/.app не проверены. Все модели требуют слухового ревью.
Без коммита/push; чужие изменения сохранены.
Docs updated: docs/project.md, docs/STATUS.md,
docs/tasks/2026-10-13-model-shortlist/fast-model-selection.md и эта задача.

Повтор:

```sh
rtk proxy .venv/bin/python -m tools.bench_fast_models --models whisper-tarteel-tiny quran-phonetics whisper-tarteel-base --audio-dir /Users/ayub/Desktop/Abu_Bakr_Ash-Shatri --reference data/references/abu_bakr_ash_shatri.db --output docs/tasks/2026-10-13-tarteel-tiny-fast/comparison.json
rtk proxy .venv/bin/python docs/tasks/2026-10-13-tarteel-tiny-fast/analyze.py
rtk proxy .venv/bin/python docs/tasks/2026-10-13-tarteel-tiny-fast/api_validation.py
```
