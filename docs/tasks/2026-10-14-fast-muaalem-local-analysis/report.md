# Muaalem: независимые локальные phones и незавершённый wasl crop

14 октября2026. Исследование продолжается; Muaalem не активирована в GUI/CLI FAST,
default model/weights/зависимости не менялись. Новый wasl-edge код пока существует
только в isolated sparse-9 prototype. Общая цель остаётся active, pytest не запускался.

## Причина исходной проблемы

После двух исправленных ошибок подключения/счётчика выполнены3 проверенных
локальных чтения1:2 через общий LocalEvidenceReads. Muaalem независимым greedy
декодирует **полное** ожидаемое `ءَلحَمدُ` в6420–7920/8520 и6120–9120.
First-entry probabilities=.712/.915/.980. Сопоставляемый текст не передаётся
акустическому inference; полные аяты используются только static phonetizer.

Quran Phonetics на первом том же PCM выдаёт `lḥamdulilā`, initial-a posterior
около.002–.003. Алфавиты двух heads различаются, их token probabilities нельзя
сравнивать как одно и то же событие. Тем не менее альтернативная модель даёт
полный независимый first-word decode там, где первичная не даёт его. Это
обоснование исследования нового акустического адаптера, не повод вставлять
отсутствующий a в первичный decode или изменять transliteration cache.
Человеческое прослушивание не выполнено.

## Контроль времени/тишины/чужого текста

`controls.json`: при+200ms leading silence onset сдвигается на161.23ms,
ошибка38.77ms (примерно один frame). Это один local control, не общая калибровка.
На silence полного слова нет, вероятность correct prefix7.19e-19.
Wrong target(первое слово3) на исходном crop имеет9.40e-38 вместо correct.915.
Все настоящие acoustic inputs bounded; source/runtime/PCM SHA и ledger сохранены.

## Почему initial pipeline всё ещё остановился на1:5

Experimental adapter строит штатные Muaalem targets и проверяет, что word spans
являются полным неперекрывающимся partition. На1/32/112 эта проверка прошла.
Не менялись phoneme labels/waqf/madd; используются runtime hafs/madd4/waqf=ayah.

На sparse-8:1 имеет4 proofs,112 —4,32 —0;8/41 подтверждений,0confident>500ms.
1:5 не читается на принятой паузе (geometry gate), native wasl forecast читает
17332.625–22832.625,5.5s. Greedy содержит полный previous tail
`ددِۦۦۦۦن` и6/9 нужных opening units `ءِييَا`. Сильного противоречия нет;
DSP speech fraction по обе стороны crop edge=[1,1]. Алгоритм не расширяет
такой recovery crop, хотя до6s остаётся500ms. Это отдельный доказанный пропуск
native recovery, открывшийся благодаря лучшему acoustic output.

## Конечный контроль расширения

Read-only `wasl_extension.py`: тот же start, end+500ms, строго6s. Новый decode
содержит **полный** tail+opening `ددِۦۦۦۦنءِييَااكَ`, min essential quality.880,
onset21692.625 противGT21536 (+157ms). Ничего не достраивалось forced path.

Изолированный prototype разрешает ровно одно такое расширение при единственном
полном tail,≥3 совпавших opening units до самого конца decode, достаточном
качестве, unfinished DSP speech edge, отсутствии противоречия и наличии word
positions. End ограничен6s/длительностью/началом provisional future. Новый полный
proof снова проходит прежний validator; partial не становится подтверждением.

На1/32/112:1 теперь7/7,112 по-прежнему4/4,32 по-прежнему0/30.
Новые1:5/6/7 имеют ошибки157/130/73ms; max всех11proofs308ms.
Прежние8proofs точно сохранены, новых confident>500 нет (`edge-comparison.json`).
Это маленький prospective результат, ещё не основание для production activation.

## Ошибки эксперимента и честный учёт

1. Phonetics constructor требует `weights_sha256`; Muaalem immutable identity
   хранит SHA в `assets[muaalem/model.fp32.onnx]`. Первый запуск остановился
   до audio reads. Standalone adapter теперь использует реально loaded asset SHA.
2. `dispatch_spy` выбирал ASR по `hasattr(recognize_local)`, хотя все RuntimeEmitter
   имеют этот метод. Следующая попытка выполнила3 CTC reads, но spy записал0;
   verification упал до export. Их raw arrays утрачены, три входа учитываются
   отдельно как невалидированные повторные измерения, не как доказанные результаты.
   Исправлен tools/bench_fast_models.py: Whisper определяется по model family;
   CTC counts emissions. Два unit checks подготовлены, не запускались.
3. Hardcoded output pipeline.json перезаписал initial raw control при prototype
   запуске. Candidate сохранён edge-pipeline.json, stdout initial pipeline.log
   сохранён, fresh control заново выполнен в pipeline-control.json. Он явно
   назван повторным контролем, а не восстановленным старым measurement. Added
   output env prevents repeating the overwrite. Прежний источник успешного
   локального probe сохранён attempt-3-success-source.py.

В scripts теперь сохраняются checkpoints до verifier. Никогда не считать
сбой instrumentation ошибкой acoustic модели или успешным измерением точности.

## Полный prospective corpus завершён: activation запрещена результатом

Isolated corpus-edge завершился exit0:согласованные10 сур+полная2+регрессионная79,
12случаев/607строк/344акустических вызова.116proofs,491review. Inference+matcher
157.55s, DSP38.11s, decode8.82s. Source/adapter архивированы edge-source.zip/
edge-manifest.json. Process terminal и summary сохранены; повторять этот прогон
из-за прежней пометки running не нужно.

Confident errors:55:14−50310ms,79:26+508ms. Muaalem/prototype **не готовы к
activation**, несмотря на7/7 Фатихи. Default model не менялась.

Причина55:14 установлена:time11890 относится к3(GT11918), а не14(GT62200).
Actual decode=`خَلَقَلءِںںںسَاانَعَللَمَ...`, реальные3→4. Canonical первые два
слова3/14 побайтово одинаковы:«خلق الإنسان». Phoneme-only planner считает их
различными:для3 предполагается waqf/madd4 без конечной фатхи,для14 —connected
speech с конечной фатхой. В аудио3 connected→4, его sound form совпадает с
модельной14. Beam/null margin325 означает речь против blank, но не исключает
настоящего владельца3 в альтернативном wasl form. Это ошибка semantic owner
discrimination при pronunciation variants, не50сек акустического смещения.
`false-55-14-proof.json` содержит phones, одинаковые canonical words и raw read.

Следующий необходимый шаг: canonical word identity должна задавать нижнюю
границу различающей фразы независимо от acoustic spelling/waqf/madd.14 требует
третьего слова«من», которого real3→4 не содержит. Для3 нужны left-tail или
допустимые wasl/waqf variants; нельзя объявить shared prefix уникальным.

## Canonical floor: новый широкий prospective контроль

`identity_floor_probe.py` проверяет нижнюю границу по **каноническим словам**,
сохраняя прежнюю phonetic проверку; исключения noisy opening тоже не могут
обойти minimum. Это только process-local prototype, source runtime не изменён.
На55:14 требуются3слова, а реальные3→4 содержат«خلق الإنسان علم», поэтому
ошибочный14 больше не подтверждается. Реальный3 остаётся review: его left-tail
2 ожидает `قُرءَاااان`, тогда как raw local tail=`قُرءَاان`. Нельзя вставить
недостающие long-vowel units в услышанное; требуется допустимый вариант модели.

Свежие12cases/607rows/343reads:115proofs вместо116. Потеря только ошибочного
55:14; **все115 остальных подтверждений сохранены**. Новых confident
outliers нет,79:26+508ms остаётся. All>500478→474, median9290→6319ms из-за
исправленного sparse anchor/interpolation; это не универсальная accuracy score.
Inference+matcher157.93s против157.55s в прежнем одиночном замере: ускорения нет.
`identity-floor-comparison.json` хранит per-ayah losses/additions; всё raw/ledger
сохранено отдельно, output overwrite не повторялся.

Следующий этап не заменяется одним безопасным отказом: canonical floor перенести
в общий planner/validator после парного контроля существующих моделей; затем
реализовать вариантные wasl/waqf/madd контексты и минимальные named-letter proofs.
Muaalem/default activation остаются закрыты, pytest по-прежнему отложен.

Отдельно оценить internal/pause-restored pronunciation rivals и
intro(Basmala) перед production Muaalem;32 требует минимальные доказанные
letter-name границы, а не чтение всех начальных букв одним длинным crop.
Расширение native recovery отдельно сравнить на существующих поддержанных
FAST моделях и длинной записи, прежде переноса из prototype в runtime.

Ruff изменённых путей/diff-check проходят. Docs updated: этот каталог, STATUS.
Основной FAST остаётся sparse-8 с проверенным empty-region recovery предыдущей
серии; Muaalem, prefix proof и native extension — ещё экспериментальные.
