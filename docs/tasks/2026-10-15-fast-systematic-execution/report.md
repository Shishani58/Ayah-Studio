# Системное исследование локального FAST: измеренный результат

Поправка к статусу: этот отчёт завершает только измеренный этап. Закрытие по нему
общей цели улучшения FAST было преждевременным; выполнение продолжается в
[задаче production-улучшения](../2026-10-15-fast-production-improvement/task.md).

15октября2026. Исследовательский цикл M0–M7 завершён с отказом от production
переноса прототипа. **Системное исправление FAST не доказано; production-прирост0.**
Выявлена воспроизводимая зависимость распознавания короткого CTC opening от
контекста. Узкий прототип даёт по2прямых подтверждения INT8/FP32 на Шатри112/114,
но не имеет положительного межчтецового случая. Согласно поручению он оставлен
изолированным. Полные emissions/FULL/tiled fallback не запускались.

## Что мешает

FreshA: INT8/FP32/Tiny/Base =2/2/105/130 из607,48/48rows+dispatches точно
повторяют исторический sparse12. У каждого CTC10/12сур без первой опоры;
в остальных остановка на2аяте.605неподтверждённых строк CTC — потенциально
блокированный хвост, а не605независимо диагностированных model failures.
Бюджет первые остановки не объясняет.

Новые локальные CTC опыты80:1 и98:2 показывают отсутствие нужных phones уже
в independent decode. FP32 и короткий контекст этого не исправляют. На112
сокращение того же PCM-origin с1500до700ms возвращаетq: `ul'mualā`→`qul'hū`.
На положительном1:1 слишком короткий crop обрезает `bis'mi` до `bis'm`.
Значит, безусловное сокращение всех чтений также не обосновано.

Статический аудит77432слов/114сур:0OOV,0non-NFC; pinned vocab/blank37/config
согласованы. Это не доказательство правильной фонетики каждого target.
Обучение на отдельных словах согласуется с чувствительностью к контексту,
но без независимо размеченных word boundaries не доказывает её физическую причину.
Подробности: [диагноз адаптеров](adapter-diagnosis.md),
[аудит представлений](representation-audit.json), [первичные источники](primary-source-manifest.json).

Аудит ASR strong unique openings не нашёл пропущенного подтверждаемого future:
неиспользованный1:1 имеет текст, но не поддержанное первое время. Named-letter
conflicts32 и рефрены55 не являются безопасными опорами. Переписывать recovery
или ослаблять essential quality на этом основании нельзя.

## Прототип и причинный эффект

Только CTC, только полное минимальное opening из3phones. Сильный уникальный
остаток после одной потерянной начальной единицы разрешает одно чтение500msречи
у той же паузы. Target/rivals симметричны по пропуску; новый полный exact greedy
proof с обычным качеством и прежней DSP geometry обязателен. При неудаче короткого
чтения используется исходное observation. Время из остатка не выводится.

| Случай | Подтверждения до→после | Прямой прирост | Каскад | Calls | Вход модели |
|---|---:|---:|---:|---:|---:|
|INT8/FP32,112|0→1|+1|0|6→9|10,0→18,9s|
|INT8/FP32,114|1→2|+1|0|8→12|13,0→26,1s|

Каждая строка таблицы одинакова по счётчикам у двух вариантов weights.
Новые метки112:1=6020ms и114:1=6010ms; ошибки к reference74/29ms.
Прежние положительные строки не потеряны и не изменены. Дополнительные calls
после первой опоры пытаются продолжить, но новых downstream proofs не получают.
Стоимость прироста не скрыта за словом «ускорение».

НаA CTC2→3, наE INT8 9→10 иFP32 10→11. Tiny/Base A105/130 неизменны,
их24pairs имеют точные rows/dispatches. Все остальные A/E CTC различия
перечислены в [построчном сравнении](comparison.json).

## Межчтецовый gate и границы корпуса

Гамиди, Миншауи, Хусари и Абдуль-Басит mujawwad:80/98/112, дваCTCварианта,
24before/after пары. Всеrows/dispatches идентичны; новый cue не возник ни разу.
Это regression без положительной проверки применимости ветки. **Gate переноса
не пройден.** Шатри112/114 — разные записи, но один чтец и одно слово `qul`;
выдавать их за обобщённое исправление модели нельзя.

Итоговый paired corpus:86пар/172modelcases. A48пар четырёх моделей, B–D24CTCпары,
E14CTCпар. Не затронутые ASR варианты B–E исключены по зависимости branch от
`ctc_greedy_probability`; A дополнительно проверяет их dispatch equivalence.
F114не запускался: условие принятия A–E не выполнено, общего production изменения нет.
Это явное ограничение вывода, не114-сурная проверка. Полный длинный файл2 пройден
в обоих A вариантах с обычным локальным ledger.
[Корпус и scope](corpus-manifest.json), [контракт измерений](measurement-contract.md).

## Цена и отказы

Три тёплых повтора каждого112/114 на каждомварианте повторяютrows/dispatches точно.
Медиана executor, before→prototype:

| Модель |112|114|
|---|---:|---:|
|INT8|0,488→0,745s|0,538→0,990s|
|FP32|0,588→1,009s|0,736→1,355s|

Load/probe,decode,DSP,inference,matcher/other отдельно сохранены в
[baseline](baseline.md), [измерениях](measurement-audit.json) и rawsummary.
Coldload на A:0,405/0,869/5,540/4,733s соответственно четырём моделям.
Whisper startup дополнительно выполняет по одному synthetic1s-zero probe
(encoder+decoder); это не recording read и входит в load/probe. Неучтённых
timing forwards нет. [Аудит startup](startup-probe-audit.json).

TreeRSS измерен как одновременная сумма main+живыхworkers с номинальным шагом200ms:
A baselinepeak2171,4MiB; CTCrepeat baseline800,9 иprototype1290,0MiB.
Это абсолютные пики процессов, не доказанный causal memory delta метода;
короткие пики между samples могут быть пропущены. IndividualPIDs и фактический
cadence сохранены в [memory summary](memory-summary.json) и *-process.json.
Cachelimit32MiB сохранён; per-case actualcache в rawtrace.

Все recordingdispatches сверены независимым PCMspy с ledger; ошибки вызовов,
union/sum/cache иoutcomes сохранены. В A/E prototype maxunion49,23%,
maxinput81,89%; полные лимиты6s/65%/200% соблюдены во всех прогонах.
6реальных negativecalls: нули, чужой и поздний non-targetcrop; ни один не дал
`qul`proof. [Негативы](negative-audio.json), [risk/coverage](risk-coverage.json).
Wrong-owner и физическая temporalrisk не выдуманы из ошибки к reference:
они остаются отдельно неоценёнными без независимой разметки. Порог не подбирался.
На новых подтверждениях нет >500ms; прежний Base79:26+508ms безreview воспроизведён.

## Identity, cue и время

[Контракт доказательств](evidence-contract.md) разводит searchcue, полный identity
иtemporalproof дляgreedy/beam/posterior/seam/noisy/split/body/sparse/wasl.
Новая короткая гипотеза не превращает remainder вproof. First-entry probability
не названа вероятностью любого позднего вхождения: проверенные offsetmaxima
не восстановили обычные crop80/98/112 и не приняты как новое posteriorevent.
Representation/tempo coupling известен, но без новой доказанной representation
поправки разделение runtime структур не выполнялось.

Подготовлены [протокол](annotation-protocol.md) и18WAV/SVG с
[пустыми labels](annotation-package/labels.csv). **Прослушивания нет.**
Humanlistener должен отдельно отметить physicalonset, previousspeechend и
допустимый playbackinterval, проверитьowner и достаточностьcrop. Для других
чтецовDSP locator не удостоверяетowner. [Временная оценка и блокер](temporal-evaluation.md).
Старые неудачные midpoint/attention методы не повторялись. Timing=False сохранён.

## Воспроизводимость, чужие изменения и M7

Исходный HEAD37f3309; грязнаякопия сохранена, ничего не сброшено. Baseline/prototype
исполнялись в независимых roots; actual__file__/cwd/SHA/policy/runtime закреплены
в *-imports.json и *-command.json. Originalsource ZIP, measured-short-prototype
ZIP/manifests и все rawlogs остаются рядом. [Все попытки](attempts.md),
[checkpoint](checkpoint.json), [исходный status](initial-status.log).
Первыйbaseline exit1 до модели и ошибка генерацииannotation также сохранены.

Во время исследования другая работа добавила progress и FULL/UI изменения.
Они не перезаписаны. Для FAST/existing CTC/read logic AST точно совпадает после
удаления толькоprogresscalls/imports. Заморожен отдельный currentsource;
freshsmoke4модели×112/114 завершён, все6сопоставимыхcases точно повторяют
baseline rows/dispatches; ASR114—два дополнительныхsmoke без historicalpair.
[Concurrentdiff/AST/smoke](concurrent-audit.json). FrozenA не переименован
в полный свежий currentA. Файлыdata98DB+126MP3(224guard) неизменны.

Заключительная текущая suite: **772passed,8skipped**,
[pytest.log](pytest.log). Подготовленный isolatedprototype contract: **10passed**,
[log](prototype-contract.log); предупреждение pytestcache о /dev не относится
к проверяемому коду. [Ruff](ruff.log), [diffcheck](diff-check.log), AGENTS/CLAUDEsync
успешны. Persistence/resume/manual/history/FULL/UI проверены существующей suite;
пользовательские/referenceDB не использовались для записей.

Из production этой задачей ничего не изменено. В общей копии добавлены только
исследовательские артефакты этой папки и записьSTATUS. project.md этой задачей
не менялся: production поведение не изменилось. Эта задача не выполняла commit/push.
HEAD общей копии параллельно изменился до3dc28641907d81c3ce5d2aec06ac23605206b794;
final-audit.json фиксирует его и отсутствие drift относительно проверенногоcurrentroot.
Итог — измеренный предел выбранных методов, локальный положительный результат
без разрешённого общего переноса и конкретный пакет для временной зависимости.
Низкое покрытие FAST и слуховая временная неопределённость остаются открытыми.
