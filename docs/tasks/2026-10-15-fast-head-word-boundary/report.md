# FAST: перепроверка границы ASR-слов после независимого body cue

Рабочий executor теперь sparse11-head-boundary. Перед изменением выполнены
диагностика и свежий парный корпус; user tests пока не запускались.

## Причина и алгоритм

98:5 shortread «وماء» не совпадает сcanonical «وما أمروا». Строгое старое
expansion правило не видело точного первого слова и прекращалоэто чтение.
При этом другое localread независимо читалоunique canonicalpair «وذلك دين»
сselectedtokenquality≥0,786. Этотcue доказываетcontent, но не начало.
Factorcontrol доказал: достаточноувеличитьspeechcontext,preroll менятьненужно.

Новаяfallbackветка выполняется послеотказапрежнихcurrentrecoveries, когда:
есть≥2старыхanchors, глобальноразличающийнеcontextualplan, firstcanonicalword
один ASRunit≥3letters иplan требуетследующегослова; independentlyheard full
unique bodypair проходитобычныйASR0,6/attribution/alias/truncation gate.
Старыйheadread обязан иметьvalidtokenownership иrecognizedprefix полного
первогослова слишнимокончанием. Егонизкоеquality неслужитproof, толькоcue
неразрешённойграницыслов. Начало нерисуетсяизbody/поGT.

Та жеизмереннаяDSPpause перечитывается2500ms, затем4000ms только еслипосле
первойперепроверки естьстарое штатноеexpansionevidence. Confirmation вновь
требуетполныйglobalplan ивсеessentialselectedtokens≥0,6, безaliases/conflicts
/truncation; общийvalidator проверяетproof/read/time/budget. No lettersrepaired,
no threshold lowered. Bodyquality не подменяетquality новогоhead.

## Fresh96cases

Frozen4supportedmodels×12сур×before/after:96cases/4856rows. Всеsource67+67guards
послепрогонапрошли. QPCINT8/FP32 иBase: exactrows/dispatchsequence unchanged.
Tiny84→87:98:5/6/7, errors78/12/44ms. Потерь 0; прежние подтверждения не изменены; новых отклонений >500 мс — 0.
RootHEAD_BOUNDARY_RECHECK только98:5,2calls наcandidate6; послеcorrecthead
обычныйexecutorсампродолжил6/7. Это не bodyfutureprototype,79:26 не разблокирован.
Затронутыйстарыйpartialwordq0,5196 не являетсяподтверждением; finalheadmin0,99765.

Mainbyteguard совпалsfrozenbefore до переноса. Послеопытногокорпуса только
VERSIONубранprobe suffix; control/modelinputлогиканепоменялась. Runkey сменён,
чтобы прежниеcachedauto не обозначалиновое поведение; manual/verified не тронуты.
Четыре regression checks подготовлены (strong/weak/partialbody,unrelatedhead),
но не запускались. ScopedRuffengine/scripts/tests clean, AGENTS/CLAUDE identical.

Source-pair.zip/manifests, analyze.py/eligibility, всеJSON/log/summary,
comparison/added-and-events/run-totals, production.diff/guard сохраняютattempts.
Первыйstaticeligibility erroneously требовалwrongheadwholequality≥0,6;
реальныйweakhead долженбытьтолькоvalidstemcueприstrongbody. Финальныйnewhead
всегда≥0,6. Ошибкаwindowaverage vsessentialwordquality отдельноисправлена
ванализе. InitialRuff E701 вanalyze.py исправленformatter; никакихpytest.

Docs updated: task.md/report.md/docs/project.md/docs/STATUS.md.
No commits/push/DBwrites, no fullacousticpass, all ledgerlimits respected.
Remainingresearch:79:20Tinyuncertainty иknown79:26temporal508; bodyfuturebranch
остаётсяprivate, другиеmodel flags/defaults не менялись.
