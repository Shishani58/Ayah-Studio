# FAST v2: исполнение и проверка

## Итог

Единый экспериментальный executor и независимые акустические адаптеры реализованы.
Это **не завершённая production-миграция**: парный M5 coverage gate провален
для всех трёх моделей. Старые loops не удалены и не переключены. Все достижимые
кодовые этапы и реальные измерения выполнены; listening и независимые acoustic
subgroup проверки остаются открытыми. Условия плана не ослаблены.

## Статус этапов

| Этап | Реализовано / проверено / остаток |
|---|---|
| M0 | Исходная рабочая копия сохранена родителем отдельно от HEAD; фиксированы40сур, три установленные модели, hard/recovery/cache policy. Исторический37 baseline и свежий парный37 сохранены. P18 устранён, обычный pytest независим от import settings. |
| M1 | Общий LocalEvidenceReads с единым charged-at-dispatch ledger, exact-window bounded cache32MiB, failed/cancelled accounting, доказательствами и строгой валидацией. M1a сохранён. Legacy cache не получает eviction без quality gate. |
| M2 | CTC/Tiny/Base adapters получают только local PCM и scalar original length; независимый decode, canonical word map, captured revision/lag и per-unit quality. CTC timing по frames; ASR честно без word timing. Реальные модели измерены. |
| M3 | Один finite sequential experimental loop: DSP, minimal whole-word phrase, current/future/internal конкуренты, кандидаты/темп, bounded two-ayah recovery, identifiable whole-word wasl tail, strict stop и conf0 priors.46сценарных тестов и реальные40. Реальное качество недостаточно, это не скрыто статусом реализации. |
| M4 | Общий pipeline seam с explicit experimental flag, immutable run snapshot/key, transactional current manual merge, truthful commit/cancel/JSON outcomes, CLI/Web/Qt callers, historyv2 и failure sidecars. Временные БД/failure injection/e2e проверены. Production route activation gated; exact-key preflight для штатного нового GUI/CLI пока не включён. Qt native недоступен. |
| M5 | Выполнены реальные измерения и независимый dispatch spy; парные ворота вычислены по всем моделям. Coverage gate FAIL для каждой. Новая слуховая adjudication не проведена. |
| M6 | Не выполнен: production switch и удаление legacy запрещены непройденным M5. Experimental isolation сохранена; project.md/STATUS отражают фактическое состояние. |

## Реализация и исправления ревью

- Нормализация/представление не изменяют канонический Коран. CTC больше не
  сопоставляет отдельно collapsed слова с глобально collapsed строкой: raw
  canonical chars и обычный greedy CTC (повторы через blank сохранены).
- Phrase planner сохраняет whole-word boundaries, не обрезает multi-unit word,
  не имеет лимита6слов. Проверяет все канонические начала внутри суры без
  материализации всех длинных suffixes. Короткое целое слово разрешает
  уточнение, но не выдаётся за различающую фразу.
- Нет автоматического подтверждения следующего после провала текущего.
  Любой доказанный future останавливает поиск; next может участвовать только
  в атомарном recovery2аята, более далёкий — review.
- Wasl использует различимый whole-word tail предыдущего аята и настоящий
  onset CTC. Tarteel не выдумывает timing. Финальный validator сверяет и tail
  quality, и ownership, и услышанную различающую фразу.
- First onset не равен константе6000. Дополнительный bounded15s DSP/VAD поиск
  после обнаруженной речи восполняет пропущенные intro pauses, RMS сужает
  VAD-полку; source/ID/interval/center сохраняются отдельно от общего DSP.
- Estimates не получают акустический confidence; распределение учитывает
  длительность последнего подтверждённого аята, а не ставит следующую метку
  через1ms после его старта. Integer range/order проверяются до сохранения.
- Snapshot не зависит от последующего ACTIVE_ID. Импорт WebGUI не читает
  пользовательские settings; ASGI lifespan восстанавливает их при startup.
- Writer не доверяет устаревшему manual overlay из pipeline: повторно читает
  edited/verified под транзакцией и проверяет эффективную последовательность.
  Отмена после commit не скрывает сохранённую суру; диагностика имеет отдельный
  исход. Tuple-compatible return сохраняет существующих callers.

## Корпус и воспроизводимость

Audio: `/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{sura:03d}.mp3`.
Reference: `data/references/abu_bakr_ash_shatri.db`, только read-only ПОСЛЕ inference.
40сур:1,2,9,78–114;986аятов. Парный37:78–114;564аята. Одни PCM/audio hashes
и initial DSP decisions для всех адаптеров; дополнительные first hypotheses
нового executor явно отдельны. Модели/веса локальные, скачивания и FULL не запускались.

Артефакты:
- `corpus.json`/`.summary.json`: диагностический v1, не финальный результат.
- `corpus-corrected.json`/`.summary.json`: v2 после CTC seam/tail/prior fixes.
- `corpus-final.json`/`.summary.json`: frozen v3 после first-candidate/future fixes.
- `corpus-legacy.json`/`.summary.json`: свежий legacy37 на том же исходном DSP.
- `edges.json`: derived108 crop/no-intro candidate,12s silence,repeat,truncated8s,500ms.
- `analyze.py`, `paired-analysis.json`: воспроизводимый пересчёт парных gates.

Каждый фактический CTC/ASR dispatch перехвачен независимо от trace: проверяются
длина, SHA PCM-среза, соответствие заявленному окну и ledger. Whole file и >6s
отклоняются; union<=65%,sum<=200%, order/range и centers проверяются отдельно.
Source hashes до/после frozen run равны. Сохранены load/probe,decode,shared DSP,
executor elapsed и local model call seconds; дополнительный DSP выводится отдельно.
`ru_maxrss` — cumulative peak процесса/завершённых children на Darwin (bytes),
**не** peak конкретной суры/модели; retained trace JSON также занимает память.
Во время свежего legacy37 обнаружен чужой параллельный FULL benchmark42
(`tools/bench_full_windows.py`, процесс37040): подтверждённая конкуренция за CPU.
Он не запускался этой задачей, не останавливался и его файлы не изменялись.
Поэтому legacy wall times — наблюдённые значения при этой нагрузке, не доказательство
ускорения нового FAST; качество/PCM hashes/dispatch counters от этого не отменяются.
Изолированный сравнительный speed/RSS benchmark остаётся ограничением. Cache peak — собственный conservative accounting
контейнеров; это не изолированная RSS memory benchmark.

## Проверка

- `rtk proxy .venv/bin/python -m pytest -q`: финальный независимый запуск
  родителя на текущем workspace — **738 passed, 7 skipped, 16.88s**, включая
  новые внешние FULL-тесты. Обычный запуск без setup plugin/reset ACTIVE_ID.
  7 skip — opt-in реальные audio tests; настоящий FAST audio проверен отдельными
  frozen corpus/spy прогонами. Исторически до v4:726/7,8.60s; до P18 изолированный
  plugin run701/7 не был стандартным зелёным suite.
- `rtk proxy .venv/bin/python -m pytest tests/test_fast_executor.py -q`:
  **46 passed, 2.03s**. Это сценарии+реальные canonical maps, не замена corpus.
- Scoped Ruff всех наших Python/routes/tests/harness и `rtk git diff --check`
  чистые; родитель независимо проверил source, все111пар, replay120 и edges15.
- AGENTS.md/CLAUDE.md идентичны. Canonical/reference49 SHA совпадают с исходным
  snapshot. Пользовательские DB/settings и ручные метки не редактировались.

Команды реальных измерений (модели внутри каждого нашего прогона последовательны):

```sh
rtk proxy .venv/bin/python -m tools.bench_fast_models --experimental --models quran-phonetics whisper-tarteel-tiny whisper-tarteel-base --surahs 1 2 9 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 --audio-dir /Users/ayub/Desktop/Abu_Bakr_Ash-Shatri --reference data/references/abu_bakr_ash_shatri.db --output docs/tasks/2026-10-14-fast-full-execution/corpus-final.json
rtk proxy .venv/bin/python -m tools.bench_fast_models --models quran-phonetics whisper-tarteel-tiny whisper-tarteel-base --audio-dir /Users/ayub/Desktop/Abu_Bakr_Ash-Shatri --reference data/references/abu_bakr_ash_shatri.db --output docs/tasks/2026-10-14-fast-full-execution/corpus-legacy.json
rtk proxy .venv/bin/python -m tools.bench_fast_edges --audio /Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/108.mp3 --output docs/tasks/2026-10-14-fast-full-execution/edges.json
rtk proxy .venv/bin/python docs/tasks/2026-10-14-fast-full-execution/analyze.py
rtk proxy .venv/bin/python docs/tasks/2026-10-14-fast-full-execution/replay.py
```

## Результаты неизменяемых ворот M5

Парный набор78–114:564аята/37сур на модель. Подтверждение здесь = confidence>0 **и** отсутствие в review_required_ayahs. Отдельные positive_conf показатели сохранены в JSON; пересечений positive_conf+review в этих прогонах нет. Все111пар имеют одинаковые audio/DSP/reference SHA.

| Модель | Подтверждено: legacy → v3 | Ревью: legacy → v3 | >500ms без ревью: legacy → v3 | Первые аяты: legacy → v3 /37 | Coverage gate |
|---|---:|---:|---:|---:|---|
| CTC | 276 → 9 | 288 → 555 | 6 → 0 | 4 → 8 | FAIL |
| Tiny | 395 → 132 | 169 → 432 | 8 → 0 | 11 → 31 | FAIL |
| Base | 411 → 208 | 153 → 356 | 9 → 1 | 11 → 33 | FAIL |

У первых аятов >500ms без ревью0 у всех моделей/обоих путей. Наблюдаемое снижение числа уверенных расхождений не означает прохождения M5: покрытие упало у каждой модели. Единственный остаточный confirmed disagreement Base81:15,+607ms уже есть в legacy. Новых крупных no-review disagreements относительно этой пары нет; слуховая adjudication не выполнялась.

| Модель | Все >500ms (legacy → v3) | Median ms (legacy → v3) | p95 ms (legacy → v3) | Max ms (legacy → v3) |
|---|---:|---:|---:|---:|
| CTC | 107 → 516 | 73.5 → 3964.0 | 4465.9 → 8900.7 | 26762 → 14230 |
| Tiny | 70 → 390 | 68.0 → 2286.5 | 1627.2 → 8519.7 | 6062 → 13508 |
| Base | 64 → 323 | 67.5 → 1340.5 | 1628.2 → 8269.7 | 6062 → 13508 |

Дополнительные1/2/9:422аята, подтверждено1/0/0 (CTC/Tiny/Base), остальные421/422/422 — review. Fresh legacy для этих трёх не запускался; эта группа **не парная**. Полные40:986аятов, подтверждено10/132/208; >500ms среди всех меток935/810/743, median6110/5688.5/5379ms, max112206/99344/99344ms.

Новый путь:0 timeline/budget/whole-file violations на120model×recording. В legacy сохранены2 out-of-range estimates82:18/19 CTC; не замаскированы исправлением baseline. Другие local-budget проверки проходят.

### Время и локальное чтение нового40

| Модель | Model calls s | Supplemental DSP s | Executor other s | Executor total s | Calls | Sum/union s | Max union/sum % | Accounted cache peak bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CTC | 13.71 | 7.45 | 1.35 | 22.51 | 275 | 508.82/464.23 | 45.56/59.61 | 21860 |
| Tiny | 76.03 | 8.14 | 0.60 | 84.77 | 371 | 764.18/556.32 | 57.26/88.80 | 39820 |
| Base | 174.25 | 7.42 | 0.55 | 182.22 | 453 | 906.72/689.68 | 58.48/89.72 | 34572 |

Общие decode12.29s/DSP47.03s подготовлены один раз на40записей; load/probe CTC0.15s,Tiny4.24s,Base4.31s. Executor other включает matcher/validation/accounting/Python overhead, это не чистый forced alignment timer. Legacy37 executor elapsed141.48/311.92/1031.27s, при описанной внешней нагрузке; CTC model-time отдельно не instrumented (`null`), ASR cache bytes legacy не instrumented (`null`). Нулём эти неизвестные значения не считаются. Полные RSS/cache/read/deny счётчики — paired-analysis.json.

### Проверка последней правки без повторного акустического прохода

Acoustic corpus-final записан на frozen v3. Затем v4 исправил weak-tail acceptance (review вместо исключения), finite per-unit validator, добавил VAD asset/runtime и actual pause-pool identity, сохранил bundle до decode и безопасно пропускает optional Silero recall при отсутствии файла. Thresholds и стратегия чтений для имеющегося корпуса не менялись. `replay.py` заново выполнил executor/DSP/VAD с независимыми сохранёнными observations: **120/120** результатов и списков окон побайтово/численно совпали; SHA каждого реально подставленного PCM-среза совпал с original dispatch. Повторных text-model calls не было. `replay.json` фиксирует old/new executor SHA и текущий Silero hash/ORT1.27.0/TEN absent. Это replay доказательств, не новый acoustic benchmark v4.

### Реальные v4 edge probes

Все15 model×case запусков завершены; независимый dispatch spy и родитель
проверили canonical108:1–3, finite integer/range/order, фактические calls/ledger,
6s/65%/200% и запрет whole-file. Число подтверждённых границ из3:

| Производный вход108 | CTC | Tiny | Base |
|---|---:|---:|---:|
| Crop с6320ms (кандидат после intro) | 1 | 0 | 0 |
| +12s начальной тишины | 1 | 3 | 3 |
| Повтор участка6320–10460ms | 1 | 3 | 3 |
| Обрезка до8s | 1 | 1 | 1 |
| Очень короткие500ms | 0 | 0 | 0 |

Последний случай: **ноль acoustic dispatches** у каждой модели и все confidence0.
Это stress evidence контракта на настоящем аудио, без новой ручной ground-truth
и без слухового подтверждения точности полученных меток.

## Почему coverage остаётся низким

`final-diagnosis.json` отделяет исправленный candidate defect от оставшихся
ограничений доказательств. V2→v3: CTC7→10,Tiny71→132,Base114→208 confirmations
на986аятах; новых >500ms confirmations не добавилось. Паузы для78/79/88/93
появились после проверенного VAD/RMS recall, не после изменения quality threshold.
Например CTC78 теперь читает реальное окно5720–7820ms, но независимо выдаёт
`an` вместо канонического `ʿamma`. Подмена decode ожидаемым словом недопустима.

CTC:40stop случаев —15no_local_evidence,9unresolved_adjacent_block,
11future_outside_bounded_block,4unresolved_local_block,1incomplete phrase.
Все10 exact-prefix reads прошли .25 unit threshold; **ни одного** отвергнутого
только из-за слабого transition frame. Поэтому предложенный пересчёт качества
по максимуму CTC сегмента не внедрялся без доказательств пользы.
Tiny:13adjacent,4далёкихfuture,6unsupported timing,5incomplete,4ambiguous,1no evidence.
Base:11adjacent,4далёкихfuture,7unsupported timing,2incomplete,4ambiguous.
ASR timing в локальном контракте отсутствует; было бы неверно чинить coverage
выдуманным onset. Отказы с полным началом и низким token quality есть: Tiny50,
Base32 локальных чтения. ASR unit quality использует консервативный минимум всех
неслужебных generated tokens как нижнюю границу; это может отвергнуть сильную
начальную фразу из-за другого слабого слова. Локальная word-to-token attribution
пока не поставляется runtime и не имитируется средним confidence.

На1/2/9 Tiny/Base не получают первого подтверждения, CTC получает только1:1.
Длинные2/9 остаются review; похожие начала2:11/13/68–70 проверены canonical/fake
сценариями, но acoustic продвижение к ним в реальном корпусе не достигнуто.
Реальных подтверждённых wasl в final40 нет: CTC1first_onset+9pause,
Tiny132pause,Base208pause. Wasl capability/recovery проверены сценариями, не
объявлены качественно доказанными на этих записях. Независимо размеченных
acoustic subgroup ground-truth для wasl/похожих начал нет: fake cases и
trace evidence_kind не заменяют такие парные метрики. Listening и эти группы
не закрыты, даже несмотря на уже решающий общий coverage fail. Это открытые
ограничения M5.

## Ограничения и открытые P-ID

P01/P17 остаются открытыми на production: legacy loops не удалены до M5.
P10 exact independent decode ограничивает coverage; phonetic/ASR ошибки не
исправляются подстановкой ожидаемого канонического текста. P12 Tarteel wasl
остаётся review без доказуемого timing. P14 остаётся quality risk, пока новые
disagreements не проверены слухом. P06/P13 получили реализацию и тесты, но не
универсальное доказательство качества для других записей. P02–P09/P11/P15/P16
реализованы в экспериментальном маршруте; P18 исправлен и в штатном import.
P19 ручная explicit overwrite policy сохранена, schema не менялась.

Другой чтец локально не найден; derived108 — stress cases, не новые эталонные
записи. Реального playback/listening adjudication нет; расхождение с reference
не объявляется ошибкой эталона. Native Qt и собранные приложения не проверялись.
Крупные errors у conf0 хвоста честно показываются: отсутствие false confidence
не означает полезного качества результата. Для M6 нужен новый достаточный
набор доказательств M5, а не понижение зафиксированных ворот.

## Изменённые файлы относительно исходного snapshot

Полный список и внешние исключения — [changed-files.json](changed-files.json).
Новые core: `fast_executor.py`, `fast_adapters.py`, `alignment_identity.py`;
существующие: reader, model_worker, pipeline, transactional writer/ProjectDB,
CLI/Web/Qt callers и минимальные UI warnings. Новые четыре test modules,
`bench_fast_edges.py`; расширен `bench_fast_models.py` и Node async checks.
Артефакты/скрипты этой task-папки сохраняют diagnostic, final, paired и replay
доказательства; independent-review.md написан родителем.

В docs/project.md и STATUS.md атрибуция только нашим FAST/writer разделам.
Параллельные FULL файлы/задачи исключены; общий dirty diff против HEAD не является
списком этой работы. Docs updated: task.md/report.md, docs/project.md,
docs/STATUS.md, исходный fast-execution-contract/task.md (актуальный статус).
Proposal v2, канонический текст, SQLite schema и пользовательские данные не изменены.
