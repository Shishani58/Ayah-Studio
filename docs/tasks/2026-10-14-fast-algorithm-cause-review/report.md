# Проверка причин и исполнения алгоритма

2026-10-14. Прежние общие объяснения «модель путает», «повторы» и «паузы
внутри аята» были недостаточны для ревью исполнения12шагов. Ниже конкретные
ветви, воспроизведения и изменения, необходимые для исправления. Код приложения
не менялся; исходные7 свежих model calls сохранены раздельно от fake/history.

## 1. Почему CTC теряет a и что из-за этого делает исполнитель

3 свежих локальных чтения1:2:6420–7920,5780–7920,4980–9120мс.
На первых двух greedy=`lḥamdulilā`; максимальный posterior a перед l равен
0.002213/0.003504, blank в этих frames≈0.99774/0.99643. Это выход акустической
модели, до matcher. Полная canonical prefix probability=`alḥamdu` равна
0.002235/0.003461, без-a prefix=`lḥamdu` —0.51023/0.54403.
Поддерживающий forced prefix path может вставить a, но его качество0.0022–0.0035
не превращает его в услышанный звук. Более широкий4.14с crop содержит предшествующий
текст и меняет другие фонемы; first-entry prefix для текущего аята из начала такого
crop не является корректным доказательством позднего вхождения текста.

Адресный2×2 A/B того же1.5с crop: INT8/FP32 × local/global-DSP normalization.
Все4 дают `lḥamdulilā`. Значит, на этом случае missing-a не устранили ни отказ
от INT8, ни смена нормализации; нельзя приписать дефект квантованию или выбору
RMS без этих проверок. Global normalization — только DSP arithmetic, никаких
full acoustic passes. Причина отсутствия символа на уровне акустического выхода
доказана. Физическое произношение a, его слышимость и причина ошибки обучения
не установлены слухом; не выдаются за доказанные факты.

Отказ executor: `_prefix` должен покрывать всё запланированное начало
(`fast_executor.py:733`); posterior proof требует probability≥.9 и каждый
essential unit≥.25. Для минимальной фразы из одного слова missing first unit
не даёт exact-first-word/noisy-later-word expansion: поздних слов в выбранной
phrase нет (`:154–190`), а incomplete-speech-edge исключён, поскольку наблюдение
уже длиннее phrase (`:197`). Recovery добавляет жёсткие требования полного tail
+ opening и снова требует слабое a. Это объясняет отказ на имеющейся паузе,
а не доказывает отсутствие аята в аудио.

Исправление: при повреждённом минимальном начале нужно перепланировать локальный
identity proof на дополнительный различающий контекст/слово, сравнив с wrong-owner
и internal occurrences; слабую onset единицу оценивать отдельно для timing.
Одно совпадение forced path, автоматическое удаление a из канонического
представления или снижение порога не являются исправлением. Для pause timestamp
нужна проверенная geometry; для wasl — independently supported onset.
Критерии: real1:2 плюс missing/wrong first-word negatives, внутренние совпадения,
uncertain onset отдельно; подтверждение должно опираться на реально услышанный
различающий материал. Канонический текст остаётся неизменным.

## 2. Повторные начала: доказанный пропущенный recovery

Шаги5/8/9 требуют различающий фрагмент/расширение/совместный соседний контекст.
Planner действительно ищет минимальный opening среди всех word starts, и
plan_seam умеет искать tail предыдущего + более короткое current opening.
Но `fast_executor.py:663` вызывает seam для pause только когда
`plan[ambiguous] and confirmed`. Когда целое начало различимо далее по тексту,
ambiguous=False. Если различающее слово не уложилось в1.3/2.5/4с, tail recovery
для pause не пробуется. Последующая ветка recovery (`:885`) gated timing, поэтому
Tarteel даже доступный contextual pause proof не читает.

Fake-сценарий с настоящим набором112 и явно synthetic representations:
opening2=(shared,a2w5); shorter contextual opening=(shared), previous tail unique.
Первый аят подтверждён;3 nested reads текущего слышат shared. Результат2conf0,
previous-tail reads=0; reason unsupported_wasl_timing. Это воспроизводимая ветвь
исполнителя, не заявление о real ASR accuracy112.

Исправление: после конкретного incomplete/noisy opening на реальной pause пробовать
plan_seam независимо от статического ambiguous флага; separate charged read
previous tail + current opening, same validator, минимум обеих essential qualities,
без word timing как предусловия для pause-center. У wasl timing по-прежнему нужен.
Проверить wrong/truncated/low-quality tail и глобальные повторные seams.

## 3. Внутренние паузы и центр: уточнение прежнего вывода

Сам факт внутренних пауз не означает нарушения алгоритма: `plan_phrase` проверяет
внутренние word-start occurrences, executor проверяет слышимую различающую фразу.
Центр назначается прямо по шагу10. Прежняя общая формулировка не была конкретной
находкой несоблюдения алгоритма и отозвана.

Конкретный остаток: `pause_geometry(:278)` автоматически принимает interval≤1000мс.
В сохранённом Base79:26 proof аят определяется по сильному «إن في» после паузы,
но pause860мс принимается как short_dsp_interval без дополнительного geometry
measurement. Confidence≈1 относится к тексту; timestamp midpoint имеет508мс
расхождение с reference. Это исторический unlistened disagreement, не доказанная
на слух неправильная пауза. Политика 4×decay также не отклонила бы measured
ratio3.81 просто при снятии1000мс gate — нельзя обещать такой механический fix.

Исправление: независимо проверить реальные края/тишину и временную неопределённость
этого interval и matched controls, затем prospective geometry rule. Сохранить
центр для доказанной pause. Не компенсировать неопределённость времени высокой
ASR word probability и не подгонять правило к одному508мс примеру.

## 4. Почему чтение Tiny/Base пока не возвращает границу внутри crop

Это реализуемая задача шага9, а не принципиальная невозможность локального FAST.
`model_worker.py:385` запускает recognize_local с return_timestamps=False;
он возвращает текст/selected tokens и probabilities, без word positions.
`fast_adapters.py:160` объявляет timing=False; observe возвращает positions_ms=None.
Executor вместо локального wasl read выдаёт unsupported_wasl_timing (`:884–924`).
Для CTC эта же ветвь реально читает только прогнозный local5.5с crop и ищет
tail+opening с frames. Для Tiny/Base time-producing stage не подключён.

Имеющиеся FULL attention-DTW heads нельзя просто включить: сохранённый настоящий
temporal probe Tiny crop5720 даёт innā1020мс; при +200мс leading silence1040мс
(ожидался сдвиг≈200, получился20). Wrong target также получает конечный time.
Это historical measured evidence, не новый inference данной задачи и не внешний
word-onset ground truth.

Исправление: после independently heard локальной фразы подключить отдельное
local word alignment в том же bounded crop, вернуть positions/word ownership,
сверить seam/onset, учитывать все model forwards через общий ledger. Проверить
silence-shift invariance, known onsets и wrong-text controls; если inherited heads
не проходят — исправлять метод локальной привязки, а не объявлять текст временем.
Запрещённый FULL по всей записи не нужен. timing=True допустим после проверки.

## Проверка

1 fake recovery case +3 fresh CTC calls (`results.json`) +4 fresh INT8/FP32×normalization
calls (`runtime-ab.json`). Все реальные inputs local1.5–4.14с, spy PCM/SHA/ledger
проверены. Первый real run: sum7780мс,union7.02%; A/B:1500мс на каждую из4 попыток.
Global normalization считается DSP, не full inference. Production/DB/settings
не записывались. Ruff/diff-check passed. Полный suite не повторялся: runtime
код не изменён, проверки адресные и были объединены в заключительную фазу.
Следующий шаг — узкие recovery/local-timing исправления с указанными negatives;
это ревью не выдаётся за выполненную починку.
