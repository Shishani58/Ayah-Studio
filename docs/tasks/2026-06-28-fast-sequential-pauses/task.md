# Fast sequential pauses

## Скоуп

Доработать только fast-режим: сопоставлять паузы с аятами последовательно,
подтверждать старт аята коротким greedy-чтением после паузы, явно логировать
пропущенные/лишние паузы и метрики эффективности. Формат БД не менять.

## Доки

- docs/project.md — прочитан: инварианты тайм-кодов и схема БД.
- docs/STATUS.md — прочитан раздел fast/pause: текущее состояние режима и логов.
- docs/tasks/2026-06-28-fast-log-diagnostics/task.md — прочитан: текущий trace/log.
- docs/tasks/2026-06-28-fast-pause-match/task.md — прочитан: уже сделан старт
  greedy-чтения от конца паузы и локальный matching.
- docs/tasks/2026-06-28-fast-no-full-fallback/task.md — прочитан: fast не должен
  автоматически уходить в full.
- docs/development/task-files.md — прочитан: задача меняет пайплайн fast.

## Решения

- Вместо глобального «лучшего» прохода по паузам добавить последовательный матч:
  ожидаем текущий аят, читаем ближайшие паузы, если слышен будущий аят — сначала
  пытаемся восстановить пропущенные аяты на паузах между последним якорем и этой
  паузой.
- Считать и логировать бюджет пауз: границы аятов, waqf-знаки в mushaf-тексте,
  принятые/отклонённые паузы и покрытие.
- Если подтверждённой паузы нет, оставлять аят как wasl/local-align, а не
  назначать слабое совпадение как точный якорь.
- Для слабых `weak-dsp` пауз добавить prefix-guard: fuzzy similarity недостаточно,
  услышанный фрагмент должен начинаться как целевой аят. Это отсекает внутренние
  паузы, где декод начинается с хвоста предыдущего аята.
- Разделить точку чтения и точку таймкода: читать после паузы от
  `pause.end_ms - 50мс`, но pause-backed старт аята писать в `pause.center_ms`.
- Если локальный wasl-align нашёл старт рядом с сильной видимой паузой, repair
  не имеет права уводить старт дальше от этой паузы только ради большей
  confidence внутри слова; такой wasl-старт также снапится к центру сильной
  паузы.
- Добавить wave/tempo prior: сравнивать длину звучащих участков вокруг паузы с
  ожидаемой длительностью соседних аятов по текстовым единицам и waqf-знакам.
- Добавить micro-pause scan: при `future_gap` искать локальную RMS-просадку в
  speech-span около текстово предсказанной точки пропущенного аята и добавлять её
  как synthetic `micro-tempo` pause, если рядом нет уже найденной паузы.

## Чек-лист

- [x] Добавить helper последовательного сопоставления.
- [x] Подключить его в Phase 1c fast.
- [x] Расширить trace/realtime/fast-log метриками.
- [x] Добавить модельно-независимые тесты.
- [x] Проверить ruff/pytest/py_compile.

## Проверка

- `python3 -m py_compile surah_cutter/core/align_fast.py surah_cutter/webgui/server.py`
- `uv run ruff check surah_cutter/core/align_fast.py surah_cutter/webgui/server.py tests/test_align_fast.py`
- `uv run pytest tests/test_candidates.py tests/test_align_fast.py` — 29 passed.
- Диагностический fast-прогон без записи в БД:
  `/Users/aabibulat1/Desktop/Badr_Turki/093.mp3`, сура 93.

Результат диагностического прогона:

- elapsed: 2.26s
- rows: 11
- pause_budget: ayah_boundaries=10, waqf_marks=0, accepted=11, used=9,
  decoded=9, decode_reads=14, ambiguous_openings=[7,8,10,11]
- coverage: 0.842 (51.95s нейро-чтения на 61.73s аудио)
- n_pause=9, n_wasl=1, n_refine_skipped=9
- sequence_events: confirmed=8, future_gap=1, future_anchor=1,
  wasl=1
- аят 8 восстановлен на паузе 6 (`36.26-36.80s`, sim=0.375) после будущего якоря
  аята 9; затем extended decode включается только для общих префиксов 7/8 и
  10/11.
- drift_detected=False: verify_sim=0.597, mean_conf=0.597, pause_coverage=0.9.
  Остаток — аят 2 как единственный wasl.

Дополнительная проверка по ручным замечаниям на
`/Users/aabibulat1/Desktop/Badr_Turki/077.mp3`, сура 77:

- rows: 50
- coverage: 1.543 (364.52s нейро-чтения на 236.20s аудио)
- pause_budget: ayah_boundaries=49, waqf_marks=1, accepted=54, used=46,
  decoded=54, decode_reads=84
- n_pause=46, n_wasl=3, n_refine_skipped=37
- sequence_events: confirmed=43, extra=5, future_gap=3, future_anchor=3, wasl=3
- drift_detected=False: verify_sim=0.591, mean_conf=0.640, pause_coverage=0.939
- wave/tempo prior включён в trace/log: `tempo_score`, `prev_speech_ms`,
  `next_speech_ms`, `prev_span_ratio`, `next_span_ratio`, `waqf_before/after`.
- micro-pause scan включён в trace/log: `micro_pauses`. На 77-й найденные
  RMS-просадки для пропущенных участков оказались `near_existing`, то есть рядом
  уже были pause-кандидаты; новых synthetic pause не добавлено (`micro_pauses=0`
  в budget), таймкоды остались стабильными.
- аят 9: старт 27.40s, пауза 9, sim=0.723
- слабая пауза 10 (`28.16-28.36s`) отклонена как внутренняя: декод
  `samāufurija...` начинается с хвоста 9-го, а не с открытия 10-го
- аят 10: старт 30.75s, пауза 11, sim=0.750
- аят 11: старт 33.30s, пауза 12, sim=0.667
- аят 26: локальный wasl-align нашёл старт рядом с сильной паузой 89.76s и
  снапнул таймкод в её центр; repair-кандидат 90.15s отклонён, потому что уводил
  старт от видимой паузы. Wave/tempo подтвердил участок:
  `tempo_score=0.065`, `prev_span_ratio=1.087`, `next_span_ratio=1.148`.

## Блокеры

Нет.

## Wave/tempo prior

Реализован мягкий prior без эталона: аудио-структура сравнивается с известным
текстом.

- Для каждой пары соседних пауз считать `speech_span_ms`: длину звучащего участка
  между `prev_pause.end_ms` и `next_pause.start_ms`.
- Для каждого аята считать `text_units` через текущую `_unit_len`: обычные буквы
  + долгие гласные как более длинные темпо-единицы.
- Использовать локальный `tempo_ms_per_unit` из текстовой длины суры и длительности
  контента; следующий шаг для улучшения — заменить его скользящей медианой по
  последним надёжным якорям.
- Для каждой паузы логировать `prev_speech_ms`, `next_speech_ms`,
  `expected_ayah_ms`, `span_ratio`, `tempo_ms_per_unit`, `waqf_before/after`,
  `tempo_score`.
- Waqf-знаки использовать как структурный сигнал: пауза внутри короткого
  участка рядом с waqf чаще внутренняя остановка, а пауза после участка,
  совпадающего по длине с целым аятом, сильнее кандидат в границу аята.
- В `future_gap` recovery получает бонус, если `tempo_score` у паузы хороший;
  если пауза не подтверждается, участок всё равно уходит в локальный wasl-align.
- Использовать prior только как мягкий скоринг: `acoustic_score + sequence_score
  + tempo_score + pause_quality + waqf_score`. Greedy-чтение после паузы остаётся
  главным подтверждением.

## Micro-pause scan

Реализован поиск незаметных пауз, которых нет в обычном списке кандидатов:

- Первый sequential pass работает по обычным `dsp`/`weak-dsp` паузам.
- Если возникает `future_gap`, для каждого пропущенного аята берётся интервал
  между предыдущим подтверждённым якорем и будущим якорем.
- Ожидаемая точка внутри интервала считается по накопленным `_unit_len` между
  этими якорями, а не по глобальному prior всей суры.
- Вокруг этой точки сканируется локальное окно RMS; если находится достаточно
  глубокая просадка, она становится synthetic accepted pause с reason
  `micro-tempo`.
- Если просадка ближе `_MICRO_PAUSE_MIN_SEP_MS` к уже найденной паузе, новый
  кандидат не добавляется, а в trace пишется `near_existing`.
- После добавления хотя бы одной `micro-tempo` паузы sequential matching
  запускается заново: новая пауза читает аудио тем же greedy-decode от
  `pause.end_ms - 50мс` и должна пройти обычную проверку текстом.

## Docs

Docs updated: docs/STATUS.md, docs/tasks/2026-06-28-fast-sequential-pauses/task.md.
