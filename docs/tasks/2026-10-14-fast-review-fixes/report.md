# Исправления второго ревью FAST

2026-10-14. Реализация локальных proofs, ownership и loaded identity готова к
независимой перепроверке родителя. Это экспериментальный opt-in FAST;
production/legacy не переключены, старые M5 ворота не переопределены. Пункт 5
закрыт частично: tail+opening recovery реализован, temporal ASR остаётся
измеренно некалиброванным. Качество CTC и общее coverage требуют дальнейшей работы.

## Семь находок

| ID | Изменение и validation | Оставшееся evidence |
|---|---|---|
| 1 | Exact greedy больше не единственный путь: targetless CTC beam с rivals/null; marginal prefix likelihood со всеми canonical word-start alternatives, null и unrestricted residual mass. Single forced path только указывает support. Probability ≥ .9 и margin ≥ ln(9), finite validator/captured policy. Noisy exact fragment/later word, posterior-vs-noise и измеренный incomplete speech edge разрешают конечные nested expansions. Tiny brute-force CTC, residual .74 vs expected .26, NaN/mutation/other-owner negatives; настоящие local probes. | Механизм исправлен, полезность CTC остаётся недостаточной. 108:2 p≈.26 не выдан за proof. Короткий terminal competitor как free-future prefix консервативно блокирует длинный expected; EOS/continuation proof пока не реализован. |
| 2 | Глобальный повтор разрешается только через различающий previous tail + current opening. Отдельный read заканчивается ровно у pause.left, opening начинается у right−200ms с bounded pre-roll. Stale/truncated/weak/wrong-owner/forged context отклоняются. Canonical 109:3/5, 55, 77 сценарии и реальные записи. | Реальный повтор с одинаковым локальным контекстом остаётся ambiguous/review. Порядок и tempo сами по себе не proof; второй чтец/listening subgroup не проверены. |
| 3 | Tarteel каноническое слово «الم» представляет как الف/لام/ميم; decoded compact alias раскрывается в те же spoken names. Minimal complete spoken word может быть частью canonical word, ownership хранит всё canonical rendering. Strong explicit/compact conflicting continuation отклоняется. 2/19/20/36/50 unit mappings и настоящие opening probes. | Не все открытия распознаются. Частичная различающая фраза не объявлена доказательством всех произнесённых букв. |
| 4 | Worker возвращает selected token IDs/log probabilities/raw bytes. Adapter восстанавливает UTF-8 token→word ownership, качество каждой proof unit — min её реальных токенов. Ненужный weak tail не отравляет начало. Split codepoint, один BPE token на два слова, nonreversible/missing/NaN mapping → review. | Средний confidence не используется вместо attribution. ASR lexical mistakes остаются возможны; per-word likelihood не является внешней калибровкой ошибок. |
| 5 | После подтверждённого current локальный tail+opening wasl proof не требует целого длинного аята в 6s и пробуется после future/blocked prefix. First short adjacent recovery сохранён. Отдельный prospective sparse anchor позволяет уже услышанный глобально уникальный future, честный conf0 gap любого размера; физический lookahead расширений=2, candidates=8, budgets неизменны. | **Частично исправлено.** CTC temporal proof доступен, Tarteel first-onset/wasl timing=False. Проведено 18 + 49 реальных charged temporal passes; оба DTW conventions не прошли калибровку. Независимой разметки word onsets и wasl ещё нет. |
| 6 | Полный DSP interval/ID/center отделён от короткого right-edge speech crop. Длинный interval не расходует весь 6s лимит на тишину. Decaying-tail RMS geometry gate требует review вместо неподтверждённого midpoint. Long silence, decay control и реальные derived/2 probes. | Историческое Tiny 2:10 +874 мс в local-sparse сохранено; причина — DSP interval 121380–124060 мс с затухающей слышимой речью. Gate закрывает долгие decaying intervals. Короткий 860 мс interval Base 79:26 остаётся известным ограничением; см. ниже. |
| 7 | Immutable loaded runtime snapshot: реально загруженные weights, config/generation/processor/tokenizer/vocab/blank/normalization/providers/version и generation knobs; loaded Python code/defaults/kwdefaults/module scalar fingerprint. Edit после load не переименовывает runtime, копии snapshot нельзя мутировать. Stub sessions не получают выдуманный weights SHA. | Benchmark spy wrapper консервативно отличается от uninstrumented GUI/CLI key. Это может лишить resume при тех же вычислениях, но не переиспользовать stale результаты. Private runtime mutation не является поддержанным пользовательским API. |

CTC posterior живёт только в bounded evidence cache. Trace хранит units,
positions, scalar alternative scores/provenance/SHA, без emissions матриц.
Recursive cache accounting считает вложенные контейнеры без повторного учёта
shared references. Eviction не возвращает acoustic budget. Existing transaction,
edited/verified protection и truthful after-commit outcomes не менялись.

## Temporal исследование

`temporal-probe.json`: 18 historical passes независимого ASR + attention на
108 crop от 6320 мс с добавлением 0/200/1200/2500 мс тишины. Crop уже начинается с речи;
reference 6020 мс — ayah pause-center, не разметка onset. Его расхождение с attention
само по себе не доказывает неправильный word timestamp.

`temporal-preroll-probe.json`: 49 новых реальных passes (14 independent ASR,
28 separate attention forwards для independently heard innā и wrong الحمد,
7 independent CTC). Каждый PCM slice, SHA/sample count и отдельный sum/pass ledger
проверен. Pre-roll от DSP left 5720 мс, варианты +200/+1200 мс silence и basmala negative.
Никакого нового production timing RPC; отдельный diagnostic subprocess перехватывает
DTW и получает обе query-row интерпретации с одного attention forward.

CTC независимо слышит innā целиком с quality > .998: onset 505/506/510 мс после
вычитания synthetic silence (loaded lag 200 мс). DSP 8% peak впервые 600 мс;
low thresholds .003/.01 слышат остаточный хвост уже у left edge. Whisper +1
даёт Tiny 1020/840/1020 мс и Base 1040/1040/1040 мс без synthetic silence. Разница
с CTC не является абсолютной внешней word-onset ground truth, но показывает,
что inherited OpenAI attention heads у Tarteel нельзя включить как калиброванный
FAST timestamp. Predicting-query alternative всегда 0, включая добавленную
тишину; shift check провален. Wrong target и basmala forced timing также дают
конечные времена и не являются identity proof. Модель не объявлена incapable;
непроверенная timing capability остаётся False. Shared FULL helper не изменён.

## Проверки и воспроизводимость

До финального acoustic freeze: обычный pytest **783 passed, 7 skipped**,
targeted executor/review tests **78 passed**, scoped Ruff и git diff --check чистые.
Skip — opt-in audio tests; настоящие audio проверены отдельными spy прогонами.
Родительское independent-* ревью — отдельная работа, не самопроверка исполнителя.

Audio read-only: `/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{sura:03d}.mp3`.
Reference read-only после всех inference: `data/references/abu_bakr_ash_shatri.db`.
Пользовательские DB/settings не менялись, canonical text не редактировался.

Команды:

```sh
rtk proxy .venv/bin/python -m pytest -q
rtk proxy .venv/bin/python -m pytest tests/test_fast_review_fixes.py tests/test_fast_executor.py -q
rtk proxy .venv/bin/python docs/tasks/2026-10-14-fast-review-fixes/temporal_preroll_probe.py
rtk proxy .venv/bin/python docs/tasks/2026-10-14-fast-review-fixes/ctc_onset_probe.py
rtk proxy .venv/bin/python -m tools.bench_fast_models --experimental --models quran-phonetics whisper-tarteel-tiny whisper-tarteel-base --surahs 1 2 9 19 20 36 50 55 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 --audio-dir /Users/ayub/Desktop/Abu_Bakr_Ash-Shatri --reference data/references/abu_bakr_ash_shatri.db --output docs/tasks/2026-10-14-fast-review-fixes/corpus-final.json
rtk proxy .venv/bin/python -m tools.bench_fast_edges --audio /Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/108.mp3 --output docs/tasks/2026-10-14-fast-review-fixes/edges-final.json
rtk proxy .venv/bin/python docs/tasks/2026-10-14-fast-review-fixes/analyze.py
```

Исторические iterations local-corpus/local-sparse/corpus-before-hint-fix сохранены.
Финальный sparse-4 freeze начат 2026-10-14 01:10 UTC; исходники между inference
не меняются. Новые acoustic paths не доказываются frozen replay.

## Frozen corpus и неизменённые M5 ворота

`corpus-final.json` — raw sparse-4: **138** настоящих model×surah запусков,
46 сур (1,2,9,19,20,36,50,55,77,78–114), **1475** аятов на модель.
Source hashes before/after равны; одна PCM/DSP подготовка на вход для всех моделей.
Каждый фактический PCM-pass перехвачен и сверён с ledger, 6s/65%/200% соблюдены,
whole-file/FULL/tiled/sequence нарушений 0. Матрицы posterior не сохранены в trace.

| Модель | Подтверждения full46 | Парный37: legacy → новый /564 | >500 мс без review: legacy → новый | Первые аяты парного37: legacy → новый | Coverage |
|---|---:|---:|---:|---:|---|
| CTC | 33 /1475 | 276 → 28 | 6 → 0 | 4 → 9 | FAIL |
| Tiny | 305 /1475 | 395 → 230 | 8 → 0 | 11 → 31 | FAIL |
| Base | 404 /1475 | 411 → 302 | 9 → 1 | 11 → 31 | FAIL |

Все 111 пар имеют одинаковые audio/DSP/reference SHA. Full46 относительно
предыдущей итерации 23/279/377 → 33/305/404; это улучшение этой итерации,
не прохождение исходного coverage gate. Всё >500 мс, включая conf0 priors:
1379/1087/997; median 6120/5800/3896 мс, max 108982/79350/87014 мс.
Низкое число уверенных расхождений не скрывает большие ошибки неподтверждённых
оценок. Первые canonical spoken-letter открытия подтверждены Tiny 2/20/36,
Base 20/36; 19/50 остаются review. Contextual repeated proofs: Tiny 12, Base 11;
109:1–6 подтверждены у обеих, max 119 мс. Реальных wasl confirmations в этом
корпусе 0. Posterior-only confirmations 0: новый механизм не объявлен доказанным
реальным выигрышем по CTC, beam/finite expansion сами по себе coverage не решили.

Base **79:26 +508 мс**, confidence≈1, новая no-review граница относительно legacy
(там confidence=0, +690 мс). Sparse anchor имеет локальное strong «إن في»,
DSP interval 115320–116180 мс, midpoint 115750 мс; reference 115242 мс.
20 мс RMS quarter medians .0164/.0098/.0082/.0043, decay ratio≈3.81.
Short interval автоматически принят исходным DSP; long >1000 мс gate здесь
не применяется. Полный proof/PCM SHA/nearby DSP/RMS сохранён в
`base79-26-diagnosis.json`. Порог не снижен ради единичного reference расхождения,
граница не скрыта выключением. Без listening/calibration остаётся новое крупное
unlistened disagreement и отдельный M5 blocker для Base. Production остаётся gated.

Actual model reads: 515/675/779; суммарный model time 47.93/164.74/445.30 с.
Максимальные local windows 6000/4200/4200 мс, union 56.35%/50.05%/51.28%,
sum 100.88%/75.50%/76.65%. Shared decode 17.61 с / DSP 58.55 с.
Это наблюдённые времена; isolated speed/RSS superiority над legacy не утверждается.
Legacy времена получены ранее при чужом FULL процессе. RSS cumulative,
cache peak — conservative accounting, не изолированный memory benchmark.

`edges-final.json`: 15 derived model×case, actual PCM/ledger/range/order/budgets
проверены. Confirmations (CTC/Tiny/Base): crop6320 1/2/2, +12 с тишины 1/3/3,
repeat6320–10460 1/3/3, truncated8s 1/1/1, 500ms 0/0/0 с нулём model calls.
Derived cropped/truncated inputs не считаются внешней word-onset разметкой.
`ctc-onset-probe.json`: 7 новых local passes на 78, pre-roll от 5520/5720 мс и
длительности до 5000 мс. Получено «an» / «amayatasā» / «lmayakasa» вместо
ʿamma; более ранний левый край и 4.8–5 с контекста не дали полного независимого
proof. Ошибка не объявлена исключительно ошибкой модели; это измеренный
оставшийся предел сочетания representation/decode/local context.

## Scalar confidence correction после freeze

После 138 + 15 + 7 запусков родитель выявил overstated contextual score:
opening .99 и обязательный tail .61 проходили оба gates, но публиковалось .99.
Sparse-5 теперь публикует min всех essential proof units; validator проверяет
confidence ≤ tail quality и для contextual pause, и wasl. Wasl producer уже
использовал общий min. Ни threshold, ни candidate/window/acceptance не изменены.
`source-4-to-5.diff` и SHA JSON фиксируют ровно этот scalar/version diff.

`corpus-score-replay.json` — **явно scalar-only**, 138 cases проверены текущим
validator; 19 scores уменьшены. PCM dispatches, observations, timestamps,
confirmed/review membership и historical sparse-4 keys неизменны. Это не новый
acoustic corpus и не rekey под sparse-5. Raw `corpus-final.json` не переписан.
`context-score-fresh109.json` — настоящие sparse-5 Tiny/Base: 6/6 каждый,
max 119 мс, по 13 actual reads. Targeted tests после correction: **79 passed**;
родитель отдельно сообщил **784 passed, 7 skipped** и 104 independent probes.
Independent review родителя оформляется отдельно.

[Независимое ревью родителя](independent-review.md) завершено: full suite
784/7, 104 адресных случая, raw 138, derived 15 и temporal 49 passes сверены.
Дополнительно [fresh verification](independent-verification.json) восстановила
126 actual PCM passes на 20/109 × Tiny/Base: 22/141 и 36/141 подтверждений,
positive errors >500 мс 0. Confidence contextual proof равен min двух reads;
границы и accepted/review decisions совпали с raw sparse-4 для этих четырёх пар.
Ссылки на parent-owned evidence: [artifact check](independent-artifact-check.json),
[derived check](independent-derived-check.json),
[temporal check](independent-temporal-check.json),
[fresh corpus](independent-fresh.json). Эти файлы не принадлежат исполнителю.
