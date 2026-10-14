# Исправления независимого ревью FAST

2026-10-14. Scope: семь находок fast-second-review в экспериментальном едином FAST. Production, legacy, канонические/пользовательские БД и настройки не меняются.

Прочитаны AGENTS.md/CLAUDE.md, RTK.md; docs/project.md (контракт/пайплайн), fast-execution-contract proposal/task (исходные ворота), fast-full-execution task/report/independent-review (история измерений), fast-second-review/review.md (находки). Исходный snapshot текущего поручения сохраняет родитель; чужие FULL-изменения не переписываются.

Решения: не повышать 6s/65%/200%; identity берётся от загруженного runtime. Canonical words сохраняют ownership при spoken-letter rendering. Качество ASR привязать к реально выбранным token IDs. Noise не разрешает подтверждение без доказанного различия; конечное расширение использует наблюдаемые локальные признаки. Повторные тексты разрешать только составным tail+opening с отрицательными контекстами. Unresolved blocks развиваются в отдельном prospective sparse контракте; старые strict результаты сохранены. Некалиброванный ASR timing не включать.

- [x] 1: локальное evidence/расширение и адресные negatives; CTC quality остаётся ограничением
- [x] 2: contextual ownership repeated ayahs
- [x] 3: spoken letter representation/canonical ownership
- [x] 4: ASR token→word attribution/per-proof quality
- [x] 5: tail+opening recovery, исследование ASR temporal proof; ASR timing частично открыто
- [x] 6: DSP interval отдельно от short speech crop; короткий decay interval остаётся ограничением
- [x] 7: loaded immutable runtime identity
- [x] Targeted regressions, scoped Ruff, pytest
- [x] Real local probes/spy, парные coverage/error metrics
- [x] report/changed-files, docs STATUS

Блокеры/непроверенное: калибровка temporal ASR, независимо размеченные wasl/repeat, второй чтец, native Qt. Самопроверка исполнителя не является независимым ревью.

Итог: raw sparse-4 corpus 138 запусков; подтверждено 33/305/404 из 1475,
paired37 28/230/302 против legacy 276/395/411 — M5 coverage FAIL у всех.
Base 79:26 +508 мс — новое unlistened no-review расхождение, сохранены PCM/DSP/RMS
и legacy сравнение. Production не включён. После corpus+edges+CTC probes scalar
confidence исправлен в sparse-5; raw corpus не переписан/rekeyed. Явный score
replay 138 cases меняет 19 aggregate scores, current validator passed; свежие
109 Tiny/Base 6/6, max 119 мс. Targeted 79 passed; parent full 784 passed/7 skipped.
7 skip — opt-in tests/test_alignment_audio.py; native Qt к ним не относится.
49 protected assets SHA unchanged. Docs updated: docs/project.md,
docs/STATUS.md, task/report/changed-files и reproducible artifacts этой задачи.

## Prospective decision after local probes

Адресные 30 model×surah: только 2/11/14 подтверждений на 824 аятах CTC/Tiny/Base. Exact independent beam не исправил CTC phonetic errors; ASR произносительные названия и compact decoded letter symbols расходятся. STRICT-stop усиливает один отказ до полного хвоста. Первая prospective версия разрешила sparse anchor после максимум двух unresolved аятов: только уже прочитанное, глобально уникальное independent opening и точная DSP ownership; пропущенные аяты остаются conf0/review. Повторные/внутренние future openings не могут быть anchors. Ни дополнительного scan, ни нового reading budget, ни переноса M5 ворот. Старый v2 FAIL остаётся историческим фактом.

### Sparse gap decision after frozen 138

В corpus-before-hint-fix только 79:1→4 и 83:1→4 у каждой модели остановлены исключительно gap=3. Уже прочитанное глобально различающее начало не становится ложным от размера непроверенного текстового блока. Prospective sparse-3 допускает такой anchor; пропуск остаётся conf0/review. Physical expansion lookahead=2, 8 candidates и 6s/65%/200% неизменны. Нет дополнительного scan для far owner. Current-prefix и provisional future hint раздельны: распавшаяся future гипотеза не блокирует поиск current после intro. Старые max2/strict результаты сохранены, M5 не переопределён.

### Final prospective sparse-4 freeze

2026-10-14 01:10 UTC. Измеренный speech edge (непустой короткий CTC decode,
речь по обе стороны правого края crop, нет полного конкурирующего owner)
разрешает конечное nested expansion, но не подтверждение и не current-prefix lock.
Проверки пустого/weak/ASR/competing/terminated edge; адресный настоящий 78/77/108/109.
Все thresholds acceptance сохранены. Implementation identity дополнительно
фиксирует function defaults/kwdefaults и scalar module knobs; детерминирован
между свежими процессами. Full pytest перед freeze: 783 passed, 7 skipped;
targeted: 78 passed; scoped Ruff clean.

Temporal исследование: 18 исторических charged calls и 49 новых (pre-roll 5 720 мс,
silence shifts, wrong-target/basmala negatives, независимый CTC onset). DTW
predicting-query alternative всегда 0, +1 показывает некалиброванный сдвиг от
независимого phonetic onset. DSP и reference pause-center не выдаются за word-onset
ground truth. Runtime timing=False; никаких правок shared FULL timing.
