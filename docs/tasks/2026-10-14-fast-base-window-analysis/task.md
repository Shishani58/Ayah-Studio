# FAST Base: причинный эксперимент размеров локальных окон

Продолжение активной цели: сначала измерять и анализировать, затем менять код.
Предыдущий goal turn — progress:158 валидных прогонов и проверенная оптимизация.
Новая гипотеза: Base на поднаборе получает больше proofs с1.0/2.0/4.0s;
неизвестно, сохраняется ли выигрыш на полном согласованном наборе и какие
прежние доказательства теряются. Не переносить настройку на Tiny/CTC.

Прочитано: RTK/AGENTS/CLAUDE, Serena manual, project/STATUS, предыдущий
iterative-analysis report, актуальный gap-fixes report (contradiction repair).
Source authoritative: текущая sparse-7 с последним contradiction guard.

Решения: fixed source archive, полный factorial начального1.0/1.3 и
следующего2.0/2.5s; третий4.0s, прежние thresholds/budgets/геометрия.
Набор1,32,55,59,65,73,77,80,98,112. Reference только после inference.
Не запускать pytest до завершения требуемых изменений всей активной цели.

- [x] Четыре свежих серии и source/PCM/runtime log.
- [x] Поаятная сохранность прежних proofs, новые confident outliers, calls/speed.
- [x] Причины потерь/успехов; factorial не внедрён.
- [x] Bounded next-DSP probe:33 paired cases, final77×3; production sparse-8.
- [ ] Подготовленные regression tests запустить только в общем финале цели.

Docs updated: этот каталог; STATUS дополнится результатом. Runtime пока не меняется.

## Следующая причинная проверка

Factorial отвергнут:short_initial84→96,15lost/27added,+672ms unflagged;short_expansion84→84;short_both84→85,15lost/16added.10weak-word cases из-за marks не превращены в ложные proofs:многие слова и без marks неверны.

На77:16 tempo region содержит0пауз/0model calls; GT16 внутри range, принятой паузы нет; следующая DSP-пауза относится к18, а не16. Проверить однократный next-DSP probe только при пустом region; не менять пороги/время/proofs. Изначально только isolated candidate, production без изменения.
