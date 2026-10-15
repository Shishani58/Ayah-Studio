# Узкие исправления FAST и открытые ограничения

14 октября 2026. HEAD37f3309, без commit/push. Исправление **частичное**:
механика recovery исправлена; реальный1:2 и надёжный Tiny/Base word timing
не закрыты. Canonical Quran/пользовательские БД/настройки не записывались.

## Что изменено в runtime

Единственный production файл этой задачи — `surah_cutter/core/fast_executor.py`.
Новые проверки — `tests/test_fast_algorithm_gaps.py`. Все прочие изменения
рабочей копии (FULL, его writer, STATUS/project и другие задачи) — чужие.

1. **Повреждённое минимальное CTC начало:** добавлено конечное перепланирование
   identity proof: независимо decoded остаток первого слова после одной потерянной
   начальной единицы + дополнительное целое слово, максимум два следующих слова.
   Остаток проверяется против всех других владельцев и внутренних word starts,
   в том числе с тем же допуском одной начальной потери. При совпадении rival
   recovery консервативно отклоняется даже если более позднее слово различает их.
   Все intervening words должны совпасть; качество — минимум всех услышанных
   essential units. Нельзя подтвердить только остатком/forced path/слабым extra
   word/чужим или вставленным текстом. Canonical phrase остаётся неизменной,
   proof явно `identity_anchors_only`, отсутствующий onset не восстановлен.
   Эта ветка разрешает только pause center, не first_onset/wasl timing.
   Validator заново строит это доказательство из сохранённого independent decode.
   Отдельного акустического вызова для posterior/forced recovery нет.

   После такого конкретного наблюдения дальнейшие speech crops —1.7/1.9/2.1s,
   вместо скачка с1.3 сразу к2.5/4s. Это минимальное конечное расширение,
   **не** проход окнами по записи. Policy и версия7 участвуют в run key.
   Реальные короткие расширения сохраняют `lḥamdu`, тогда как2.5/4s добавление
   соседней речи меняет decode на `lfadu…`/`lḥadu…`. Дополнительное exact
   canonical слово на проверенном1:2 всё ещё не доказано, поэтому time/review
   не превращены в ложное подтверждение.

2. **Похожие начала на паузе:** после реально неполного opening пробуется более
   короткий `previous-tail + current-opening` независимо от статического
   `ambiguous` и наличия word timing. Opening должен быть действительно услышан;
   previous-tail отдельно читается у левого края той же паузы, через общий ledger.
   Более короткий seam — альтернатива: если previous-tail слабый, неверный или
   truncated, восстанавливаются исходные observation/plan/phrase/beam/noisy
   proofs, затем исполняются original future-owner/finite opening expansion
   на той же паузе. Неудачный tail не блокирует полную различающую фразу.
   Future/sparse provenance всегда использует opening_id, не последний tail-ID.
   Два read-ID, минимальное качество обеих частей и canonical seam повторно
   проверяет общий validator. Для wasl требование timing сохранено.
   Новый маршрут доказан synthetic/fake regression; прирост real coverage этим
   тестом **не** доказан.

3. **Геометрия/честность уверенности:** удалён bypass≤1000ms. Для каждого DSP
   interval измеряются уровни начала/конца, robust variation, stationarity и
   радиус исходного interval вокруг midpoint. Радиус — геометрия интервала,
   **не** оценка фактической ошибки. `temporal_probability=None`,
   `timing_status=uncalibrated_pause_midpoint`; text score не назван процентом
   точности времени. Validator требует измеренную разрешённую geometry для
   коротких пауз тоже. Старый4×audible-decay gate применён ко всем длительностям.

   Экспериментальный statistical-decline gate отверг2/3 обычных пауз на matched
   controls и **не оставлен** в runtime. Без слуховых labels падение RMS от эха
   нельзя объявлять неверной паузой. Final DSP controls:1 accepted5/6,
   79 accepted54/54,108 accepted4/4. Спорный79:26 проходит исходный4×gate;
   его историческое+508ms расхождение **не исправлено и не проверено слухом**.
   Новый calibrated temporal gate и UI отображение двух независимых оценок
   остаются открытыми. `confidence_semantics` — диагностика, не калибровка.

4. **Tiny/Base word timing:** runtime/adapters/worker не менялись;
   `timing=False` сохранён. Новые методы проверены отдельными диагностическими
   скриптами, не подключены к FAST и не выданы за implemented local timing.

## Свежие реальные доказательства

- `local-verification.json`: CTC1:2, inputs1.5/2.7/4.2s, точные PCM hashes,
  dispatch spy и ledger. Целое canonical дополнительное слово не подтверждено.
  Geometry здесь относится к отвергнутому statistical gate; это intermediate
  evidence, **не** final runtime behaviour.
- `minimal-extensions.json`:1.7/1.9/2.1s speech +200ms pre-roll:
  `lḥamdulilāhi`, `lḥamdulilāhira`, `lḥamdulilāhiram`. Сохраняется independently
  heard first-word remainder; дополнительных labels `lillahi`/`rabbi` нет.
  Здесь сохранены final4×gate DSP controls.
- `separate-word-crops.json`: попытка отдельно прочитать уже локализованное
  следующее слово (7054–8654ms), результаты `…lillāhi…`, не `lillahi`.
- `isolated-opening-crops.json`: первый-word crops6420–7250/6220–7350/
  6120–7450ms дают `lḥam`/`lḥadu`/`lḥadu`; missing-a не устранён.
- `repeated-separate-word-crops.json`:3 вызова предыдущих separate crops
  непреднамеренно повторены из-за ошибки замены списка окон в probe script.
  Они учтены и сохранены, не названы новыми controls или успехом.

Обучающее представление проверено по первичным источникам:
[model card](https://huggingface.co/TBOGamer22/wav2vec2-quran-phonetics/raw/main/README.md)
указывает обучение на одиночных словах и риск coarticulation; это основание
**гипотезы** про влияние соседнего слова, не доказательство физической причины
missing-a. [Dataset viewer](https://huggingface.co/datasets/Buraaq/quran-md-words)
подтверждает1:2 `word_tr=lillahi` и `rabbi`. Cache/adapter следуют training labels;
право заменитьa→ā или удалить gemination не доказано. Blanket aliases и пороги
не ослаблены. Почему именно модель ошибается на физическом звуке, слухом не
установлено.

### Fresh ASR temporal experiments

`temporal-candidates.json`, `exact-temporal-candidates.json`,
`single-word-exact-temporal.json`: Tiny и Base, соседние crops5720/6320ms,
leading silence0/200/400ms. Проверены token-axis/time-axis normalization,
DTW с prefix absorber и без, attention peak, exact independently selected BPE
IDs вместо повторной tokenization. Все encoder/ASR forwards отдельно charged;
никакие дополнительные forwards не скрыты в runtime capability.

На5720ms time-axis с absorber успешно сдвигался200ms, но на6320ms Tiny
140→900ms (сдвиг760ms). Exact BPE на том же crop440→900 (460ms).
Для отдельно independently heard первого слова на5720ms exact-BPE/time-axis
Tiny1220→980 (отрицательный сдвиг240ms), Base720→920 (правильный относительный).
Original token-axis тоже имеет несовпадающие absolute times. Text-only DTW
принудительно выдаёт0 первому слову из-за края global path, это не onset.
Wrong forced targets закономерно получают times и не являются доказательством.

В ранних двухсловных диагностических targets второе слово не всегда полностью
распознано: соответствующий `independently_heard=False` сохранён. Эти результаты
не принимаются за доказательство ownership; именно поэтому добавлен отдельный
single-word/exact-selected-BPE control. Ни один tested method не прошёл общие
temporal controls Tiny/Base/оба crops. Независимой слуховой onset разметки нет.
**Capability не включена; надёжный local time-producing stage не реализован.**

### Небольшая парная реальная регрессия

`real-regression.json`: HEAD37f3309 executor против изменённого executor,
одни и те же PCM, adapters и реально закреплённые installed weights.

| Случай | Подтверждено до/после | Изменённые метки | Вызовов на каждый run | Union |
|---|---:|---:|---:|---:|
| CTC108 |1/1 (из3)|0|7|42.80%|
| Tiny109 |6/6|0|13|41.78%|
| Base109 |6/6|0|13|41.78%|

У109 подтверждения3/5 сохраняют separate previous-tail read-ID8/11 и минимум
качества обоих essential reads. У Tiny5 score≈.854 не заменён сильным opening.
Timestamp/proof/read intervals существующих подтверждений сохранены. Spy SHA
совпадают с ledger. Это regression control, не новый корпусный benchmark,
не слуховое доказательство точности и не доказанный прирост покрытия.

Real regression записан до последнего дополнительного ужесточения min-quality
короткого intervening word; эта recovery ветка в трёх controls не используется.
Source SHA каждого real artifact сохранён; hashes нельзя подменять final SHA.

## Проверка и воспроизводимость

Все tests/probes выполнялись в заключительной фазе после реализации.
Первая suite616 passed/7 skipped. После связанных final recovery/geometry правок
общая suite625 passed/8 skipped (параллельно добавлены чужие FULL tests).
После последнего quality tightening — адресный FAST набор96 passed.
Ruff и diff-check passed. Suite counts — текущее общее дерево, не число тестов
или результатов, принадлежащих этому агенту.

Всего сохранено15 CTC диагностических forwards (включая3 непреднамеренных
повтора),50 ASR/attention диагностических forwards и66 model calls шести real
paired runs. Inputs адресные, максимальный FAST input≤6s; бюджеты6s/65%/200%
и32MiB не менялись. GT не подавался алгоритму. Диагностические teacher-forcing
скрипты не выполняют FULL/wasl fallback и не меняют capability приложения.

Команды: `.venv/bin/python -m pytest -q`;
`.venv/bin/python -m pytest -q tests/test_fast_executor.py tests/test_fast_review_fixes.py tests/test_fast_algorithm_gaps.py`;
`env PYTHONPATH=. .venv/bin/python <local/real_regression probe>`;
`env PYTHONPATH=. ~/.surah-cutter/model-runtime/bin/python <temporal probe>`.
Проверки запускались черезRTK. Для temporal scripts108 PCM предварительно
декодирован app environment в `/tmp/fast-gap-timing-108.npy`: isolated runtime
не имеет imageio_ffmpeg. Только crops подавались модели.

## Следующий проверяемый шаг

1. Независимо разметить слухом missing-a/границы1:2,79:26 и matched quiet/echo
   controls; не принимать reference как безусловную истину.
2. Подтвердить phonetic target convention на нескольких словах/чтецах, отделив
   модельную ошибку/CTC gemination от mismatched auxiliary labels. Новые aliases
   допустимы только с общим обоснованием, симметричными rivals и negatives.
3. Для ASR timing использовать exact generated BPE/word ownership, но выбирать
   estimator только после общего silence-shift/crop-shift/known-onset/negative
   acceptance gate. Ни один исследованный estimator пока его не прошёл.
4. Измерить новый similar-opening recovery на реально неполной различающей
   фразе и отдельно оценить прирост coverage; fake branches этого не доказывают.

## Исправление независимой проверки альтернативного seam

Независимый reviewer воспроизвёл regression: shared короткое начало + weak
previous-tail раньше блокировали более длинное полное original opening.
Исправлено сохранением исходного состояния и проверкой альтернативного tail
до acceptance. Добавлены weak/wrong/truncated positives с полным distinguishing
opening в расширении1.3→2.5s и sparse/future opening-ID regression.
Ранние negative tests продолжали возвращать только shared и этой потери
не обнаруживали; они не переобозначены как достаточная предыдущая проверка.
Final FAST100 passed; актуальная версия sparse7. После narrow repair
полная suite: **631 passed,8 skipped**; Ruff/diff-check passed. Старые paired real artifacts сохранены со
своими исходными SHA, не выдаются за новые model calls этой narrow поправки.

## Проверка сильного противоречащего продолжения

Независимый reviewer затем воспроизвёл ложное подтверждение: при корректном
previous-tail после паузы полностью распознавалось различающее начало третьего
аята, но короткий общий `shared` seam ошибочно подтверждал второй.

Исправлено: более короткий contextual proof не может скрывать независимо
распознанное продолжение, противоречащее текущему аяту с качеством не ниже
порога. Проверяется только текст до канонического конца текущего аята:
соседний следующий аят сам по себе не является противоречием. Отсутствующее
или слабое продолжение не объявляется сильным противоречием. Проверка действует
при выборе альтернативного seam, обычном contextual acceptance, wasl acceptance
и повторной валидации. Исходные opening/future/finite expansion маршруты
сохраняются при отказе от короткого seam.

Добавлены четыре проверки: correct-tail + полный future opening, сильное
неизвестное wrong continuation, слабое continuation и подделанное contextual
доказательство для validator. FAST: **104 passed**. Родительский независимый
`fast-fix-supervision/regression_probe.py` не редактировался; его повторный запуск
показал `regression=false` во всех четырёх случаях. При contradictory future
baseline и новый код одинаково оставляют второй аят estimate 10684 ms / conf 0;
weak/wrong/truncated tail с полным original opening подтверждают 6100 ms / .99.

Новых модельных вызовов этой поправки нет. Действие дополнительного строгого
contradiction gate на реальный coverage отдельно не измерено; прежние real
артефакты не переобозначены как новые прогоны. Статус остаётся частичным:
реальный CTC 1:2, Base 79:26 и надёжный Tiny/Base word timing открыты.

Финальная общая suite после contradiction repair: **635 passed, 8 skipped**.
Ruff и diff-check прошли. Новых real model calls не запускалось.
