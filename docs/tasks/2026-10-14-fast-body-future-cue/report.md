# FAST: bodycue future search —итог полного корпуса

Production executor unchanged, pytest не запускался. Previous turn progress.
Cause2:57 —wrong firstword, ноretained5,5s wasl exactwholewordpair
«عليكم الغمام»/min.918237, unique canonical pair owner792. Это толькоsearchcue,
не доказательствоначала57 и не основание исправлятьconsonants/quality.

## Попытки

1.Single forecast remaining57 rawweights:8fresh1,5s reads, nofutureproof.
Selected8nearest58forecast excludes59pause152.58pause146 rightcrop уженачинается
с«قلنا ادخلوا», missing«وإذ»; расширение безpartialfuture не разрешено.
2.Separate58/59forecasts, reserve4nearest each/fill upto8 insamefinite region:
7fresh1,5s reads, полный59prefix9units/min.5142328143. AcceptedDSPpause152
midpoint1082230 vsROreference1082161 (+69ms). No57/58onsetaccepted.
Raw/Quran model targets unchanged, no knowntext suppliedtoaudio decoder.
3.Frozen sparse17 onlyafter alloldcurrent recoveryfails, firstwordcue absent,
strongunique bodypair independentlyheard. Bodycue onlyguidesremainingrawweight
forecast, oldfuture proof/alias/minquality/partialexpansion gates reused.
Mua/native prototypes stillprivate, main remains sparse10+RMSreuse.

## Freshfull2

50→57proofs,7added,0lost; allold50confirmedrows exactunchanged.
Bodyfuture57→59,62→64.57/58/62/63remainreview/confidence0.
Progressuntil67;68unresolved. Calls302→385, search145,405→190,533s.
Coverage7,312%, input12,206%, maxcrop5,5s. No speedup claim fornewproofs.
One newreferenceoutlier2:60+986ms ispause_native_onset atdecayingtailinterval
1100880–1103220 (lastquarterRMS0). Existingexperimental temporal policy remains
unresolved; body59anchor itself is69msfromreference. No activation.

## Проверки и remainingwork

Source.zip/manifest67files; diagnosticJSON/logs, full2JSON/log/summary,
comparison-2,preservation-2,new60proof/read/geometry retained.
ScopedRuffall4scripts+frozenexecutor clean. Read-onlySQLite query hadmissing
whitespaceBETWEEN57, fixed observationonly; no modelrestarts fromthaterror.
Freshother11session95663confirmedlive, first1already7proofs/12calls unchanged.
Awaitterminal+all12parity/outlier audit before decidingscope. Lastayahhasnocanonical
future: needexplicit guard ifbodybranch wastesreads there; do notmutatelivesnapshot.
Potentialnextgeneric branch can use bounded readinterval asASRbody searchcue,
buttiming-free ASR mustneverpretendtohavephonemeonset. Notimplementedhere.

Docs updated: task.md/report.md/STATUS.md. Docs unchanged: project.md —production
behaviourunchanged thisstep. No commits/push/DBwrites/tests.


## Полныйfresh корпус завершён

Other11session95663terminalexit0.607rows/301proofs против294baseline,
7added/0lost/0changedpreviousconfirmations. Все11кроме2 сохранилиrows иactual
neuraldispatch sequence точно; ни однойBODYветки наlastayah, лишнихlastreadsнет.
Всего1143calls против1060,search506,591s против457,998s;bodyfuture помогает
продвижению, неускоряеткорпус. Source67guards passedafterrun.
Единственныйnewoutlier2:60+986; прежние13outliers неизменны.
corpus-event-audit, comparison-corpus иbody-corpusJSON сохранены.
Первыйevent-audit snippet имелSyntaxError из-заскобки, исправленROanalysis;
не вызвал дополнительныхнейронныхcalls.

Static supportedASRreview:8bodycues; eligibleexistingtempoonlyTiny98:5,
Tiny79:20 иBase59:10.65:1нетpreviousanchors,Base98:8lastayah, исключены.
УASRpositionsNone —cueтолькоactualreadinterval, неwordtimestamp.
FreshBase59 diagnostic4calls не подтвердилfuture; read59:11raw«أَلَمْ تَرَيْنَ»
превращёнadapterв«الف لام ميم ترين»: арабская нормализация теряетHamza до
spokenletters expansion. Canonicaltargetraw«ألم» helperнерасширяет —конкретное
расхождениеobserve/represent. Nextstage —сохранитьrawwordownership иневыдавать
interrogativeألمзаopeningALM. Этоadapterbug, отдельный отNNошибкиترين/تر.
StageASRdiagnostic/log находятсявfast-asr-body-future; mainпоканеправлен.
