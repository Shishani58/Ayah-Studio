# FAST: дискриминирующий fixture sample-rounded onset

Изauthoritative соседнегоonset-confidence отчёта: общийsuite не чистый из-за
нашегоtest_fast_sampled_onset:actual==requested24084,fixtureнепересекаетroundhalf.
Testsэтойработойне запускались; исправитьсамfixtureпередфинальнымpytest.
Read test, FastReads._window rounding, fake weights/anchors/forecast code.

Текущийfake17/62chars, anchors1100/6100 даютrequestedcrop21835.2941176ms,
actualsample-rounded21835.3125ms. Relative2248.575 приводитк24084вобоихслучаях.
Relative2248.2 приводитrequested24083.4941→24083, actual24083.5125→24084.
Это делаетassert(actual!=requested) смысловым иловитвозвратстаройошибки.

Изменяетсятолькотест; productionformulaне меняется, no pytestuntiloverallfinal.
Docs updated: task.md/STATUS.md.

Финальный общий запуск: 683 passed, 8 skipped; fixture прошёл. Журнал: ../2026-10-15-fast-final-verification/pytest.log.
