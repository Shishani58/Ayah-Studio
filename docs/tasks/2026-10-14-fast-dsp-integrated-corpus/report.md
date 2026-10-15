# FAST: интеграционный полный корпус послеDSP RMS reuse

Previous turn progress; goal active. Production executor unchanged this turn.
Frozen sparse15accuracy source copied after67manifest guard, replaced ONLY
candidates.py by exactproven RMS reuse. Исходный candidates был byte-equal
cpu-profile before, новый —byte-equal production/after. Runner explicitly guards
resolved loaded-source path, avoiding -m cProfile import precedence.

## Fresh12run

Полные1,2,32,55,59,65,73,77,79,80,98,112;607rows/294proofs/1060actualcalls.
СначалаDSP whole record, acoustic исключительнолокальные Ledger reads.
Всеrows/actualdispatches/boundaries/fullproofs/geometry/transitions/errorarrays и
input audio/reference/initialpool hashes совпадают сretained sparse15 точно.
Source67files guard passed afterrun. Максимальныйcrop5,5s, coverage≤61,614%,
input≤99,386%, sequenceviolations0. Manual/reference DBs unchanged.

Search499,964→457,998s (-41,966s,8,39%). DSP38,232→1,285s (-36,948s).
Inference440,909→436,436s: разница4,47s —run variability, не отдельная
оптимизацияакустическоймодели. На2 search182,720→145,405s,
DSP36,668→0,606s. Ускорение сохранилось вместе со всемиaccuracy proofs.
13reference outliers остались неизменны; native/Mua prototypes не активированы.

## Другие причины, проверенные пока шёлrun

Previous-tail cost190calls/73,489s. Для184read targets retained;85fulltail,
99notfull;6targets не сохранены и не классифицированы. Диагностика grouping
поsurah/ayah/левомукраюpauses показывает44ladder сlaterfulltail послеinitial
suffix<3strongunits. Поэтому предложенноеearly-stop правило «нет сильного
suffix вshortcrop →нечитатьдольше» отвергнуто. Оно моглоубрать useful evidence.
Это counterfactual only, не44доказанных потерь итоговыхboundaries.
6ladder безlaterfulltail не доказывают безопасныйglobalstop длядругихreciters.

2:57stop: initial1,5s иwasl5,5s независимоошибаютсявfirstword(consonantظ/ض,
лишнийل). Retained wasl read301 темнеменее содержитexact fullcanonical pair
«عليكم الغمام»,minquality0,918237,offset18,actualstart1036492,512ms.
Static search allwhole canonical pair starts показывает единственногоowner792,
внутри57. Это text-body evidence дляbounded future-only search, неonset57.
Следующийexperiment: suchgloballyunique bodycue можетразрешить поиск58/59
послеотказаcurrent recovery, оставив57review0; allthresholds/budgets сохраняются.
Никакой новойpositiveboundary сейчас непринято.

## Ошибки наблюдения и проверки

Первыйpartial comparison ожидалreference_sha256 в промежуточномcheckpoint,
ноbench добавляетreference scoring лишь вфинале. Read-onlyKeyError не вызвал
перезапускlivejob. Partial сравнение использовалоROactualreference hash,
финальноеcompare.py проверяетполныеreference/errorarrays. Jobsession2032
полученterminalexit0, active-session.json теперьmarkedterminal.
Вселогимодельныхattempts, snapshotszip/manifest иloaded-source guard сохранены.
Ruffrunner/compare clean; pytest не запускался согласноuserconstraint.
AGENTS/CLAUDE идентичны. No commits/push. Docs updated: task.md/report.md/STATUS.md.
Docs unchanged: project.md —production semantics ужеописаныcpu-profile task.
