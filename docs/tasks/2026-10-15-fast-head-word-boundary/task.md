# FAST: локальная перепроверка неверной границы ASR-слов

Предыдущий turn progress: diagnostic6calls, purepost context исправляет98:5;
79:20 остаётсяbelow.6, Baseнаидентичномcrop.981. Mainпоканеизменён.
Read head-crop diagnosticJSON/source иsupported v4corpus.
Не подменятьbodyfuture: та ветка остаётсяexperimental из-за79:26+508.
Здесьпричина —слияние/достройка первого короткого слова ASR («وماء»), когда
полное начало («وما أمروا») требует context, а дальшеестьunique reliablebody.

- [x] Static eligibility по всемстарымTiny/Basecases, не толькослучаю98.
- [x] Frozenbefore/after, реальныйsame-pause recheck2500/4000ms черезledger.
- [x] Fullhead proof ≥ASR.6, alias/truncation/globalplan/geometry gates,
      никакихпереписыванийбукв/временипоэталону.
- [x] Freshsupported4×12×2, reviewlosses/newtimingoutliers beforepromotion.

No wholeacousticpass;6s/65%/200%, canonical/DBreadonly. Pytestoverallfinalonly.
Docs updated: task.md.

Staticfirsteligibility requiringwrongwholeheadq≥.6 yieldednone: actual98وماء
firstunitquality islow, averagewindowquality isnotwordquality. Refineauthorization:
bodypair must≥.6, oldheadonlyvalidtokenownership + exactcanonicalstem with
extraending (≥3letters), notconfirmation. Finalnewheadthresholdremains.6.
Thismatchesexistingpartial-expansion semantics, no probabilitysubstitution.

Completed96cases, Tiny84→87/3added/errors78,12,44ms/0loss/0new>500.
Mainpromotedbybyteguard;4regressionspreparednotrun.
Docs updated: task.md/report.md/docs/project.md/docs/STATUS.md.
