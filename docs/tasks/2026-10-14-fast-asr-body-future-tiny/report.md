# FAST: ASR body → bounded future — диагностический и полный контроль

Предыдущий этап —progress: production rawownership fix с16новымиBaseметками.
В этом этапе production executor не изменён, pytest не запускался.

## Причины и локальные попытки

Tiny79:20: текст первого слова верен, но essential BPE quality0,542<штатного
ASR0,6. Body«الآية الكبرى» quality≥0,997 указывает content, не время начала.
Tiny98:5: начало распознано неверно/неполно («وماء» вместо«وما أمروا»);
body«وذلك دين» quality≥0,786 глобально различается. ASRpositionsNone.

Cue используется как actualreadinterval, его середина только дляranking.
Finite temporal region включает remainingcurrent и nextayahchar weights,
чтобыforecast дляsecondlookahead не оказалсявне окна; до8DSPкандидатов,
4nearest длякаждогоnextforecast, chronological reading. Ниcurrenttimestamp,
ниwordposition не выдумываются. Future требуетfullgloballydistinctopening,
selectedtokenquality≥0,6, oldalias/truncation gates. Expansion толькопоpartialfuture.

Первые2diagnostics случайно унаследовали CTCпорог0,25 изстарогодрайвера.
Это ошибкаdiagnosticconfiguration; v1source/log/JSON сохранены. Повторены обе
попытки сPolicy().asr_quality=0,6:79→22 за2calls,min0,99957,+14ms;
98→6 за3calls,min0,999847,+12ms. Всего10freshdiagnosticcalls включаяv1,
неподменятьих финальнойпроверкой. No currentonsetaccepted.

## Freshpairedsupportedcorpus

Frozen production before/after (onlyexecutor body branch/versionprobe),
4models×12×2:96cases/4856rows/1568actualdispatches. Source67+67guards passed.
ASRonly,≥2confirmedanchors,currentnotlastayah; originalproofvalidation reused.
CTC INT8/FP32 иBase: oldrows/dispatches exactsame. Tiny84→90 (+6):
98:6/7 и79:22/23/24/26. Lost0, changedpriorconfirmed0.
Skippedcurrent20/21,5 остаютсяreview0; no falsecurrentonsetlabel.
Tinycalls217→237, search48,233→52,546s (+4,314s), неускорениекорпуса.

Newtimeoutlier79:26+508ms, ранее тотжетimestamp уBase (known508case).
Retainedtextquality0,999868 не являетсяtimeprobability. Geometry/read/proof
сохранены79-26-temporal-evidence. Не активироватьветку доразборавремени.
Остальные5новыхметок невыходятза500ms. Дополнительныйtextprogress не скрывает
новуюсомнительнуюboundary; старыеmanual/referenceDBs неизменны.

## Следующий причинный эксперимент

79:20 readstart91840ms, DSPpause91600–92040, reference91788 использовантолько
дляоценки. Возможнокороткий200mspreroll обрезаетначальнуюатаку, поэтомуNN
декодируетправильноеfirstword снизкимquality. Этоhypothesis, неслуховаяпроверка.
Strongunique body можетразрешитьодин bounded earlier-preroll recheck; onlyfull
firstword≥0,6 подтвердитboundary.98:5 может требовать2,5s дляразделения«وما/أمروا».
Сначалаfreshлокальныеконтроли, неhandcorrectletters инеlowerthreshold.
Отдельно79:26: DSPmidpoint можетнепредставлятьdesiredstart; проверитьphysical
onset/lefttail преждеуниверсальныхвременныхправок.

Всеscripts/logs/JSON/summary/sourceZIP/manifests сохранены. ScopedRuff clean;
AGENTS/CLAUDE identical. No commits/push/DBwrites, no whole-acousticpass.
Docs updated: task.md/report.md/docs/STATUS.md. Docs unchanged: project.md —
production behaviour thisstepunchanged, экспериментнеактивирован.
