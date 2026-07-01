# Shatri fast goal

## Цель

Итеративно улучшить fast-режим определения начала аятов на аудио Абубакра
Шатри без использования full-режима: прогонять суру, сравнивать с
`data/references/abu_bakr_ash_shatri.db`, анализировать промахи и править
алгоритм по фактическим ошибкам.

## Скоуп

- Использовать аудио `/Users/aabibulat1/Desktop/Abu_Bakr_Ash-Shatri/*.mp3`.
- Проверять fast-only результат против oracle-таймкодов Шатри.
- Не править сгенерированные `.db` руками и не трогать ручную разметку.
- Не использовать full-режим как fallback или источник результата.

## Прочитанные доки и код

- `AGENTS.md` / `CLAUDE.md`: инварианты Корана, запрет затирать ручную
  разметку, task-file policy.
- `docs/project.md`: назначение проекта, схема timing DB, references.
- `docs/STATUS.md`: текущее состояние fast, known gaps, oracle Шатри.
- `surah_cutter/core/pipeline.py`: `process_surah(..., mode="fast")`.
- `surah_cutter/core/align_fast.py`: текущий fast-алгоритм и незакоммиченные
  text-aware/internal-repeat изменения.

## Решения

- Первый проверочный прогон делать на короткой суре, начиная с 001, чтобы
  быстро получить trace и сравнение с oracle.
- Сравнение считать по per-ayah абсолютной ошибке в миллисекундах; отдельное
  внимание ошибкам, где старт попал в середину чтения вместо ближайшей паузы.
- Сура 001 показала, что DSP-паузы найдены рядом с oracle, но `sim=0.39`
  не проходил старый hard gate `0.40`; весь pause-match останавливался на аяте 2
  и дальше уходил в wasl.
- Для weak-but-local trusted-пауз черновой старт лучше брать правее центра
  паузы: на Шатри это уменьшило max error с 308мс до 252мс на суре 001.
- Сура 112 показала отдельную ошибку первого аята: anchor попал внутрь
  `intro-vad-tight` паузы, а oracle стоит ближе к правой части этой паузы.
- `drift_detected` был ложноположительным на уже хороших таймкодах: низкий
  verify-sim возникал от noisy коротких greedy-decode, а не от реального
  off-by-one дрейфа.
- Расширенная проверка 78-114 показала крупные каскады на 83/84/90/92 и
  несколько средних выбросов. На 84 корень был в sequential matcher: среди
  нескольких пауз, уже звучащих как текущий аят, structural prior выбирал
  более позднюю паузу и перескакивал следующую границу.
- Среди current-кандидатов теперь выбирается самая ранняя пауза в пределах
  0.05 от лучшего acoustic sim; это сохраняет начало аята и не даёт малому
  sim-преимуществу увести границу далеко вперёд.
- После принятия раннего current-кандидата `j_cursor` продолжает с этой паузы,
  а не с последней просмотренной future/stale-паузы; иначе следующая реальная
  граница пропускалась.
- Recovery теперь не отбрасывает local-current паузу из-за более сильного
  ложного global match в прошлый текст, но всё ещё отбрасывает будущий global
  match.
- Попытка поднять `SC_FAST_MAX_WASL_RUN_ALIGN_MS` до 25000 улучшала 84, но
  резко ломала 81; изменение отклонено и лимит оставлен 20000.
- Проверка `SC_FAST_MIN_MATCH_SIM=0.31` показала большой выигрыш на 83 без
  регрессии на проверенном subset: слабые короткие decode-подтверждения там
  были реальными pause anchors, а прежний 0.34 оставлял их в wasl.
- Сура 90 разобрана отдельно: ошибка 5-7 связана с повтором opening
  `ayaḥsabu...`; пауза oracle-старта аята 5 акустически ближе к аяту 7, поэтому
  matcher делает `future_gap`. Простое multi-ayah future правило не помогло и
  не оставлено.
- Recovery доработан для future-echo: если пауза рядом с текущим prior, но
  короткий decode чуть лучше похож на будущий похожий opening, matcher больше
  не выбрасывает её как `skip_recovery_observed_mismatch`. Это исправило 90:5-8.
- Prefix-check для onset стал терпим к выпавшему начальному артиклю
  (`l-ladhī...` vs `alladhī...`), не меняя основной fuzzy score. Это исправило
  каскад 92:16-18.
- Для weak/rejected пауз добавлена активация в `no_pause_in_window` tail-случае:
  если последний аят иначе остаётся без кандидата, слабая пауза рядом с expected
  возвращается в active set. Для последнего boundary разрешён очень слабый
  local-current match (`SC_FAST_FINAL_WEAK_MATCH_SIM=0.18`).
- Для near-prior recovery добавлены guarded случаи: current/future echo для
  immediate future, weak observed current и near-prior recovery в пределах slack
  от `SC_FAST_MIN_MATCH_SIM`. Это убрало основные хвостовые каскады 81/83.
- Для длинных low-sim DSP-пауз черновой timing берётся с левого края паузы, а
  не правее центра; это уменьшило часть поздних выбросов 84, но 84:6 остаётся.

## Чек-лист

- [x] Найти команду/скрипт для fast-прогона одной суры и сравнения с oracle.
- [x] Прогнать fast на Шатри без full.
- [x] Разобрать worst mismatches по trace/log.
- [x] Внести минимальное улучшение алгоритма.
- [x] Повторить проверку и зафиксировать результат.

## Проверка

- `uv run python ... pipeline.process_surah(..., mode="fast")` против
  `data/references/abu_bakr_ash_shatri.db`:
  - сура 112: max_abs 65мс, mean_abs 39.5мс, `drift=False`;
  - сура 113: max_abs 104мс, mean_abs 64.2мс, `drift=False`;
  - сура 114: max_abs 149мс, mean_abs 72.2мс, `drift=False`;
  - сура 001: max_abs 252мс, mean_abs 89.3мс, `drift=False`.
- Расширенный прогон 78-114 до второй итерации выявил worst: 84 max 19487мс,
  83 max 18208мс, 90 max 8143мс, 92 max 6890мс, 81 max 4186мс.
- После sequential/recovery-правок:
  - сура 84: max_abs 1143мс, mean_abs 192.6мс, `drift=True`;
  - сура 81: max_abs 4186мс, mean_abs 423.8мс, `drift=True` (регрессии нет);
  - сура 83: max_abs 18208мс, mean_abs 5752.5мс, `drift=True`;
  - сура 90: max_abs 8143мс, mean_abs 1582.6мс, `drift=True`;
  - сура 92: max_abs 6890мс, mean_abs 719.0мс, `drift=True`.
- После снижения `SC_FAST_MIN_MATCH_SIM` до 0.31:
  - сура 83: max_abs 2184мс, mean_abs 304.8мс, `drift=True`;
  - сура 84: max_abs 1143мс, mean_abs 192.6мс, `drift=True`;
  - сура 90: max_abs 8143мс, mean_abs 1582.6мс, `drift=True`;
  - сура 92: max_abs 6890мс, mean_abs 719.0мс, `drift=True`;
  - сура 81: max_abs 4186мс, mean_abs 423.7мс, `drift=True`;
  - суры 1/112/113/114 без регрессии.
- После recovery/prefix/tail/weak-current правок:
  - сура 90: max 8143мс → 180мс, <=500 100%;
  - сура 92: max 6890мс → 173мс, <=500 100%;
  - сура 81: max 4186мс → 892мс, <=500 97%;
  - сура 83: max 2184мс → 838мс, <=500 86%;
  - сура 84: max 1143мс, <=500 92% (84:6 остаётся 1142мс).
- Полный fast-only прогон Шатри 78-114 (`tools.bench_fast fast-only all`):
  aggregate med 78мс / p90 200 / p95 300 / <=150 80% / <=500 97% (n=564).
  Суры с max >500мс: 78=850, 80=851, 81=984, 82=5152, 83=843, 84=1142,
  89=1551, 97=755, 107=2637.
- После дополнительных trace-разборов остатки закрыты без full:
  - 78 max 274мс, 80 max 188мс, 81 max 255мс, 83 max 265мс, 84 max 435мс,
    89 max 287мс, 97 max 144мс, 107 max 313мс;
  - для 82 recovery теперь предпочитает раннюю near-current DSP-паузу 27.39s
    поздней micro-паузе 31.56s, когда обе локально звучат как текущий аят, но
    ранняя пауза существенно ближе к фактическому началу: 82 max 431мс.
- Финальный полный fast-only прогон Шатри 78-114:
  aggregate med 64мс / p90 160 / p95 210 / <=150 89% / <=500 100% (n=564).
  Все суры 78-114 имеют max <=500мс; worst max 431мс на суре 82.
- `python3 -m py_compile surah_cutter/core/align_fast.py`.
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py`.
- `uv run pytest tests/test_align_fast.py tests/test_pipeline.py -q` — 51 passed.

Docs updated: `docs/STATUS.md`,
`docs/tasks/2026-06-30-shatri-fast-goal/task.md`.

## Блокеры

- Блокеров нет. Цель по Шатри 78-114 в fast-only достигнута; full-режим не
  использовался.
