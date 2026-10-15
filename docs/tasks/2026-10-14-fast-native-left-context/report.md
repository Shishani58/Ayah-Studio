# FAST: previous-tail evidence у12experimental native onset

Production unchanged. Pytest не запускался. Previous turn —progress, не blocker.
Все12native_onset из frozen sparse15 корпуса проверены, включая3успешные поGT.
Source guard67файлов подтверждён после экспериментов. Ни один timestamp не записан
в пользовательскую/reference БД, и ни одна новая boundary не принята.

## Попытки

1.12fresh независимых2,5s previous-context reads через ledger. Полный canonical
previous last word найден у7/12. Raw greedy runs сохраняются scalars, не матрицами.
2.Full-word/minquality≥.25 + right margin≤2stride разрешил200ms справа у
2:10/18 и55:26.3fresh2,7s reads: endpoints стабильны (сдвиг<1,5ms), всеquality≥.25.
Начальная близость token support к краю сама по себе не доказывает обрезание слова.
3.Exact≥4unit previous-word suffix у левого края/minquality≥.25 найден только
1:5/2:9.2fresh4s reads восстановили оба полных слова. Здесь crop действительно
обрезал начало. Итог full previous proof9/12; статические тексты/пороги неизменны.

17fresh dispatches, только конечные локальные интервалы2,5/2,7/4s.
Остальные3missing tails:2:4 omitted vowel вyُنفقون,2:25 mismatch finalN→M,
32:16 mismatch возлеtail начала. Нет достаточного exact left-edge partial proof,
поэтому автоматическое расширение для них не запускалось. Не исправлять
огласовки/согласные илиcanonical targets по output decoder.

## Временная гипотеза и отрицательный контроль

Mixed-run counterfactual использует fresh previous argmax support end и retained
полный current opening native support. Это не результат prospective новогоexecutor,
а отдельная проверка гипотезы «середина между двумя независимыми опорами».

2:9 error1012→151ms;2:23 1335→462;32:20 1063→312.
2:10 2217→855,2:18 1476→565,32:11 1892→640: ошибки остаются>500.
Удачные1:5 145→-856ms,55:26 251→-971ms;77:30 337→-465ms.
Следовательно, глобальная замена native_onset на CTC-gap midpoint отвергнута.
В двух удачных случаях эталон расположен около90% gap, у ряда неудачных18–29%.
Это показывает отсутствие единойmidpoint привязки в этих примерах, но не
доказывает ошибку эталона или физическуюspeech-end точностьCTC. Human listening
не заявляется. Argmax support end не равен окончанию речи/реверберации.

## Следующий шаг

Время нельзя вычислять одним правилом, основываясь только на полномtext proof.
Нужно исследовать физическое начало текущего слова и затухание предыдущего,
с отдельной temporal assurance. Не активироватьMua/native prototypes по числу
text proofs. Зафиксирован crop дефект left context1:5/2:9; prospective helper
должен расширять только exact suffix у левого края и сохранять удачные timestamps.
Общие temporal replacements отложены как неподтверждённые, не как разрешение
использовать некалиброванное время.

## Артефакты и проверки

probe.py,edge_extension_probe.py,left_context_expansion_probe.py + JSON/log
каждой попытки; partial-left-audit, counterfactual, all-attempts-comparison.
Каждый прогонdispatch_spy/verify_dispatch, model worker released; no fullpass.
Scoped Ruff3скриптов clean. Тесты только при общем завершении цели.
Docs updated: task.md,report.md,docs/STATUS.md. Docs unchanged: docs/project.md —
production behavior не изменено; только экспериментальные анализ/контроли.
