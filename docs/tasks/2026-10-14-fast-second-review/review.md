# Независимое ревью FAST и следующий план

14 октября 2026. Проверена текущая рабочая копия, не только отчёт и не только
git HEAD. Исходный snapshot предыдущей задачи доступен; changed-files.json
использован для границ работ. FULL-изменения в worker и отдельные FULL-модули
не приписываются FAST. Production и данные не изменены.

## Находки по важности

### 1. P1 — успешный greedy-prefix стал обязательным входом в recovery

[fast_executor.py:387](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:387),
[fast_executor.py:423](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:423),
[fast_executor.py:515](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:515).

Подтверждение требует точного префикса от нулевой позиции независимого decode.
Расширение разрешено лишь после точного совпадения **всего первого канонического
слова**. Ошибка одного звука CTC или первого ASR-слова означает переход к другой
паузе, даже если текущий короткий crop мог быть недостаточен. Отсутствует
состояние «локально правдоподобный, но шумный префикс». Найденный future
останавливает поиск, а неудачный recovery переводит весь остаток в estimates.

Адресная fake-acoustic проба: на коротком окне первое слово неверно, на том же
месте при >=2s ответ был бы верным. Реально выполнены только 1.5s на паузе и
1.3s на first-onset; EXPAND_LOCAL не случился, подтверждений 0. Это проверка
перехода, не обещание улучшения настоящей модели. В corpus CTC78 при покрытии
2.96% прекращает работу на аяте1 после future; ни одного budget-denial.
CTC2 останавливается на аяте1 при покрытии0.188%, также без budget-denial.

Точный greedy — достаточное положительное свидетельство, но не единственно
возможное акустическое доказательство и не корректный тест отсутствия текста.
Необходимо сравнение локальных акустических альтернатив с отрицательной
гипотезой. Одного forced-path, среднего confidence или свободной similarity
для подтверждения недостаточно. Простое снижение .25/.6 не исправит этот дефект:
в frozen40 все10 точных CTC-префиксов уже проходили .25.

Остановка всего хвоста — осознанная политика исходного плана, а не случайный
баг. Но она усиливает один локальный отказ до потери сотен аятов. Поэтому план,
который одновременно требует не меньшего покрытия и оставляет эту политику без
развитого recovery, пока не решает исходную цель полезного FAST.

### 2. P1 — глобальная уникальность принципиально исключает повторные аяты

[fast_executor.py:46](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:46),
[fast_executor.py:123](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:123).

plan_phrase сравнивает начало со всеми word-starts всей суры; даже полный аят
остаётся ambiguous, если повторяется позже. Предыдущий доказанный контекст,
порядок и локальные конкуренты не могут снять флаг: валидатор снова вызывает
тот же глобальный planner. Акустическое улучшение само по себе этого не исправит.

На каноническом арабском представлении: ambiguous у33аятов суры55, у10суры77
и у2суры109. Последний случай уже есть в настоящем корпусе: Tiny/Base подтверждают
109:1–2, затем останавливаются на109:3 с ambiguous_phrase. Это повтор109:5,
а не плохое распознавание. Старый Whisper planner тоже имел глобальную
неоднозначность; новый общий план унаследовал ограничение вместо его решения.

Нужен составной proof владельца: различимый локальный соседний текст + порядок
и проверка повторов/пропусков. Удаление конкурентов только по предполагаемому
темпу было бы небезопасным. Если контекст не различает владельца — ревью.

### 3. P1 — Tarteel FAST не представляет начальные буквы как произнесённые названия

[fast_adapters.py:103](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_adapters.py:103),
[model_runtime.py:48](/Users/ayub/Developer/surah-cutter/surah_cutter/core/model_runtime.py:48),
[model_worker.py:41](/Users/ayub/Developer/surah-cutter/surah_cutter/core/model_worker.py:41).

FAST берёт model_text_targets через inference_words и нормализует raw «الم»
как одну единицу. Для этого же канонического word FULL-helper представляет
«أَلِفْ لَامْ مِيمْ». Аналогично19/20/36/50. Независимое ASR, правильно выдавшее
названия букв, не может пройти точный FAST-prefix. Multi-unit canonical mapping
у executor предусмотрен, но эти произнесённые единицы не поставляются.

Проба сравнила оба представления без изменения канонического текста. В настоящем
корпусе2 короткие ответы Tiny/Base дополнительно искажены («الر تُبْكِرْ»,
«الْإِفْلَى»), поэтому нельзя утверждать, что одна замена представления уже даст
покрытие. Но правильный ответ сейчас также был бы отвергнут. FULL-helper —
внешняя параллельная работа; находка относится к несовпадению FAST-контракта,
а не к приписыванию этого FULL-изменения прошлому исполнителю.

### 4. P1 — качество любой слабой части ASR-окна приписывается различающей фразе

[model_worker.py:395](/Users/ayub/Developer/surah-cutter/surah_cutter/core/model_worker.py:395),
[fast_adapters.py:111](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_adapters.py:111).

Worker вычисляет минимум всех generated non-special tokens; adapter копирует
его в каждую единицу. Это безопасная нижняя граница, но не качество слова.
Правильное уверенное начало отклоняется из-за слабого, вообще ненужного хвоста.
Расширение окна может дополнительно ухудшить этот минимум. Проба с mean=.98 и
min=.1 показывает .1 у всех пяти слов. Frozen40:50 Tiny и32 Base чтения имели
полный exact-prefix, но низкое качество; из имеющегося scalar нельзя определить,
какое слово было слабым. Следовательно, **не все** эти отказы доказанно ложные.

Нужно сохранить selected token IDs/log-probabilities и отображение token→decoded
word, затем проверять только доказательную фразу, включая её различающую часть.
Средний confidence не заменяет это отображение. Находка уже упомянута в старом
отчёте, но её исправление должно быть этапом качества, а не общим ограничением.

### 5. P1 — adjacent recovery требует весь текущий аят, а ASR вообще не даёт timing

[fast_executor.py:438](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:438),
[fast_executor.py:481](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:481),
[fast_adapters.py:92](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_adapters.py:92).

При future=current+1 recovery читает максимум6s, но ищет groups[current] целиком
+ next_phrase; после уже подтверждённого аята требуется ещё и previous_tail.
Если текущий аят физически длиннее окна, такой блок неразрешим независимо от
качества CTC. Wasl-ветка с tail+phrase не пробуется после future/blocked_prefix.
У Tiny/Base timing=False, поэтому их adjacent recovery никогда не подтверждает
пару; ASR onset без паузы всегда unsupported. Frozen40:0 подтверждённых wasl
у всех моделей, adjacent-stop9/13/11 для CTC/Tiny/Base.

Это обоснованный отказ от выдуманного времени, но недостаточная реализация
возможностей для исходной цели. Нужен локальный proof стыка tail+opening,
не весь аят в одном crop, и отдельно проверенный механизм временной привязки
Whisper. Заимствование local forced timing допустимо только внутри guard,
после независимой проверки идентичности; факт существования CTC/attention-path
не подтверждает границу. Если ASR timing не удаётся откалибровать — capability
честно остаётся неподдержанной; нужна явная продуктовая оценка такого FAST.

### 6. P2 — длинная DSP-пауза расходует размер окна на ненужную тишину

[fast_executor.py:345](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:345),
[fast_executor.py:374](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:374),
[fast_executor.py:144](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:144).

Окно начинается в pause.start, заканчивается после pause.end. Пауза>=6s
не даёт прочитать даже1.3s следующей речи. Валидатор требует read.begin<=pause.left,
хотя центр полной DSP-паузы уже известен без текстовой модели. Fake-проба с7s
паузой: post-pause identity могло бы быть доказано возле8s, но этот crop не
запрашивается, результат insufficient_local_context. Этот выбор прямо зафиксирован
исходным планом; его следует пересмотреть как геометрию proof, не повышать6s.

Хранить полный DSP interval/owner отдельно; text crop читать возле правого края
паузы с малым pre-roll. Подтверждение центра потребует DSP-проверки интервала
и доказанного следующего начала; новый validator должен проверять именно эти
условия. Широкая/ложная VAD-полка должна оставаться для ревью.

### 7. P2 — run key не идентифицирует весь реально используемый текстовый runtime

[fast_executor.py:288](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_executor.py:288),
[fast_adapters.py:95](/Users/ayub/Developer/surah-cutter/surah_cutter/core/fast_adapters.py:95),
[model_runtime.py:167](/Users/ayub/Developer/surah-cutter/surah_cutter/core/model_runtime.py:167),
[model_worker.py:152](/Users/ayub/Developer/surah-cutter/surah_cutter/core/model_worker.py:152).

Snapshot включает ORT/numpy для DSP и revision весов, но не worker.runtime_versions,
generation_config/processor/tokenizer identity Whisper. GenerationConfig загружается
из отдельного catalog; изменение её содержимого при прежнем weights revision и
adapter.version не меняет key. CTC revision — SHA ONNX, но vocab/blank ID и
emitter._do_normalize в snapshot отсутствуют. «normalization=adapter.version»
не фиксирует эти реальные значения. Часть runtime pinned на диске; это уменьшает
вариативность, но не делает ключ полной идентичностью.

Следствие: будущий exact-key resume способен принять иной runtime/config за
тот же запуск. Сейчас Web/Qt expected-key не передают и пересчитывают auto,
поэтому это **не воспроизведённый production stale-resume**. Захватывать identity
при загрузке bundle/worker, включая фактически загруженные assets и значения,
не перечитывать изменившийся файл как будто он описывает уже живую session.

## Проверка прежних выводов и границы доказательств

Независимый пересчёт исходных JSON:120 уникальных новых model×surah,111 legacy,
по37пар/564аята на модель. Audio/initial-DSP/reference SHA каждой пары равны.

| Модель | Подтверждения legacy→new | Расхождения>500ms без ревью |
|---|---:|---:|
| CTC | 276→9 | 6→0 |
| Tiny | 395→132 | 8→0 |
| Base | 411→208 | 9→1 |

Coverage FAIL подтверждён. Сохранённое Base81:15,+607ms было и в legacy;
слуховая причина не установлена. Ноль новых численных расхождений на этой паре
не заменяет проверку уверенной ошибки, а уменьшение их числа при столь резкой
потере покрытия не доказывает улучшение точности алгоритма.

Replay код действительно использует сохранённые independent observations,
проверяет PCM SHA/размер каждого dispatch и совпадение окон/rows. SHA текущего
executor совпал с replay.json; среди13 frozen source hashes отличается только
executor (ожидаемый v3→v4). Но initial DSP decisions **в replay подставляются
из summary**, а не пересчитываются; заново выполняются supplemental VAD/RMS
и первый onset. Поэтому фраза предыдущего ревью «DSP/VAD пересчитан» слишком
широка: replay не тестирует текущий полный pause_pipeline_decisions, adapters,
worker или реальные версии моделей. Он достаточен для детерминированного
acceptance-only изменения на известных observations, не для нового matcher,
новых окон, token attribution или timing. Все15 edges — реальные acoustic
контрактные пробы, но без независимых меток не доказывают точность crop/repeat/wasl.

Scoped suite в этом ревью: **186 passed,2.66s** — executor/read accounting,
identity/commit/DB safety, CLI batch, Qt contract, generation stream/write choice.
Пробы выполняются read-only, не запускают модель и не меняют ACTIVE_ID.
Новая причина дефекта не требует ещё одного дорогого полного корпуса.
Сохранены probes.py/probes.json с парными счётчиками и реальными stop-примерами.

Writer заново читает current edited/verified под BEGIN IMMEDIATE, валидирует merge
и откатывает конфликт. Последняя отмена — до commit; после commit writer возвращает
saved или saved_with_diagnostics_error. FAST Web/CLI/Qt callers учитывают этот
исход. Нового воспроизведённого дефекта защиты ручных меток или честности commit
здесь нет. Native Qt не проверен: контрактные тесты его не заменяют.
Эксперимент изолирован, legacy маршруты сохранены; это правильное решение при FAIL.
Production-resume пока консервативно пересчитывает auto; включение exact-key preflight
не закончено. Cached observations ограничены32MiB, но retained ledger/events —
отдельная память; текущие счётчики не доказывают общий RSS-предел.

## Улучшенный план: новая версия, без изменения старых ворот задним числом

1. **Зафиксировать диагностику до правок.** На имеющихся чтениях считать причины
   отказа до каскадной остановки: first-candidate recall, mismatch первого слова,
   неполная фраза, phrase-quality, глобальный повтор, unsupported timing, forecast
   exclusion, бюджет. Сохранить пары «последний proof→первый unresolved», не считать
   сотни blocked_by_unresolved отдельными акустическими ошибками. Критерий: каждый
   отказ воспроизводим по observations и PCM provenance; source/inputs frozen.

2. **Исправить представление/качество, сохранив strict-stop.** Spoken-letter mapping
   с canonical ownership; selected-token probabilities и точная word attribution;
   missing/ambiguous attribution→review. Добавить regression strong-prefix/weak-tail,
   EOS, subword/multiword и начальных букв. Критерий: канонические words/ayahs
   неизменны; слабое различающее слово не компенсируется сильным хвостом;
   посторонний слабый хвост не понижает доказанное начало. Отдельно получить
   реальные local observations для2/19/20/36/50; не запускать чтение всей суры.

3. **Сделать локальное распознавание доказательным без exact-greedy монополии.**
   Для CTC сохранить только локальные posterior/emissions в bounded cache;
   сравнить текущую/соседние/внутренние/intro/null гипотезы на одном crop.
   Для ASR использовать наблюдаемые token/word alternatives, не исправленный
   ожидаемым текстом decode. Вероятности и margins калибровать на размеченных
   локальных positives/negatives, включая неверное похожее начало. Критерий:
   подтверждение содержит различающее акустическое свидетельство и исключённого
   конкурента. Expansion для noisy partial требует конкретного acoustic evidence,
   причины и конечного набора окон. Бюджет/6s/запрет сплошного прохода неизменны.

4. **Разделить geometry и ownership.** После длинной паузы crop возле её конца,
   полная DSP-пауза отдельно. First-onset: intro/no-intro, буквы, длинная тишина;
   для ASR либо верифицированный локальный temporal proof, либо review.
   Wasl/recovery — различимый tail+opening, а не весь аят; повторный текст —
   составной proof с соседним контекстом. Критерий: реальные размеченные wasl,
   repeated109/55/77, false-VAD и длинный current дают правильный owner/время
   либо явное ревью, без synthetic confident onset. Одни fake-тесты не достаточны.

5. **Только затем отдельно решить политику продолжения.** Сначала измерить пункты2–4
   со strict-stop. Если coverage ещё не проходит, подготовить отдельную перспективную
   версию контракта: unresolved block сохраняется с conf0, более поздний независимый
   anchor не объявляет пропуск resolved. Нужны явные индексы аятов, конкуренты
   повторного чтения, конечный поиск вокруг конкретных DSP-гипотез и запрет
   последовательного text scan. Нельзя просто разрешить любой future или выбрать
   поздний повтор после неясного раннего. До реализации зафиксировать negative
   tests и критерии точности/покрытия; старый эксперимент по-прежнему FAIL.

6. **Закрыть identity и реальные ворота.** Runtime identity от загруженного bundle,
   vocab/tokenizer/config hashes и фактические runtime versions; изменение любого
   влияющего поля меняет key. Temporary DB tests: edited/verified race, cancellation
   до/после commit, ошибка sidecar, same/different key. Сохранить текущую transactional
   защиту. Новые crop/decoder/timing требуют настоящих model calls и spy; replay
   оставлять для решений на неизменных observations. Сначала адресные пробы, затем
   полный парный корпус со всеми тремя моделями, одинаковыми исходными inputs.

7. **Допуск после качества.** Прежние M5: ноль контрактных нарушений; не меньшее
   покрытие по каждой модели; ноль новых крупных уверенных ошибок на слухово
   проверенных случаях; разбивки first/long/short/wasl/repeat. Дополнительно заранее
   выделить holdout и независимо размеченные сложные границы, измерять ошибку estimates
   и количество ручных исправлений. Второй чтец обязателен для вывода о переносимости;
   без него область вывода ограничить одним чтецом. Скорость/RSS сравнить без чужой
   нагрузки при равном достигнутом качестве и числе обработанных границ. Native Qt,
   packaging и реальные GUI/CLI resume проверить перед M6. Ни production switch,
   ни удаление legacy до прохождения ворот.

Не доказано: выигрыш предложенных acoustic методов, temporal calibration Tarteel,
точность wasl/repeats, причина81:15, скорость при сопоставимом качестве,
переносимость на другого чтеца и native/packaged runtime. Это задачи следующей
реализации, а не основания объявить текущий эксперимент завершённым.
