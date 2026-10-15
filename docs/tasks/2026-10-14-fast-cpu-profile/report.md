# FAST: устранение повторного whole-audio RMS

Прошлый turn —progress, не blocker. Измерен крупныйCPU/memory hotspot,
после анализа исправлен узко, без изменения thresholds/pauses/timestamps.
Применён skill complexity-optimization (measured hotspot, minimal fix, parity).

## Причина

_dsp_pause_metrics для каждого candidate заново вычислялsqrt(mean(audio*audio))
для всей записи. Сура2:1169calls,36,76s cumulative вcProfile (self13,59s).
Количество отсчётов всей записи не связано с конкретной паузой, знаменатель
одинаков для всех shelf metrics. БылоO(P*N) whole RMS work; теперьO(N),
остальные локальные FFT/RMS/ZCR и детекторы не изменены.

## Изменение

pause_pipeline_decisions вычисляет прежнийfloat32 RMS один раз послеnonempty
energy candidates и передаёт его_dsp_pause_metrics какoptional keyword.
Standalone3argument helper сохраняет прежний fallback. Не менялисьdtype,
reduction order, thresholds, canonical text, sampling, budgets, modelinput.
Публичный candidates.py перенесён только после полного parity; bytes guard
подтвердил, что main с момента snapshot не изменился. No incidental edits.
Existing mocked helper вtest_candidates принимаетkeyword; новаяregression
проверяет одинwhole reduction наpipeline call иexact standalone metrics.
Оба тестовых изменения только подготовлены, pytest не запускался.

## Проверка до переноса

Frozen before/after snapshots текущегоmain, symlink read-only assetsdata/models,
manifest/codeguards. Fresh DSP полные12сур:1,2,32,55,59,65,73,77,79,80,98,112.
Все1697PauseDecision objects совпали точно:accepted/reason/VAD/bounds и
flatness/ZCR/RMS ratio. DSP37,955→1,253s; сура2 36,407→0,568s.
Это DSP phase, не обещание30× ускорения всегоalignment.

Fresh full2 sparse10 experimental-adapter profile before/after:
286rows,5proofs,23actual dispatches одинаковы; exact boundary/proof/geometry/
transitions иhashes audio/reference/initial pause pool совпали.
DSP37,753→1,049s, search57,720→21,062s. Modelalignment собственно19,967→20,013s,
то есть ускорение относится кDSP. After _dsp_pause_metrics cumulative0,084s.
Mua adapter остаётсяprocess-local diagnostic; public model support не расширен.
Нейронные метрики не подменяют ранее сохранённыйsparse15 corpus294proofs.

## Ошибки исследования

Первый запуск -m cProfile далcwd source приоритет передPYTHONPATH, поэтому
загрузилpublic sparse10 вместонамеренного frozen sparse15. Сохранён как
реальный sparse10 профиль; попытка equality сsparse15 закономерно failed и
не использована какparity. Runner сexplicit source root исправил precedence.
Первый DSP source-path guard failed наmacOS /var→/private/var alias доdecode;
оба paths resolved, исходныйerrorlog сохранён. Повторный запуск начался только
послеterminal exit1, никаких скрытых повторов live jobs.

## Артефакты и ограничения

profile-2/after-2 JSON/log/pstats,project-hotspots, alignment-comparison;
bench_dsp.py,dsp-before/after JSON/log, dsp-comparison;
source-pair.zip, snapshot paths/manifests, production.diff/sourceguard.
Scoped Ruff clean, AGENTS/CLAUDE byte-identical. Pytest только вобщемфинале.
No commit/push, DBschema/user DBs неизменны. Чтениеfullaudio толькоDSP;
actual text-model reads23local, неwhole acoustic pass.
Следующий этап —общий frozen corpus с новымDSP и предыдущимиaccuracy prototypes,
затем оставшиеся weak-model/temporal причины. Goal остаётсяactive.
Docs updated: task.md/report.md/docs/STATUS.md/docs/project.md.
