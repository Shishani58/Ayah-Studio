# FAST: причина отказа head после strongbody

Previous turn progress:96paired bodyfuture cases/Tiny+6/nooldloss,
new79:26+508timeoutlier, prototype notpromoted. Testsnotrun.
Goal: distinguish clipped/noisyfirstword from true ASRfailure.
Read supportedTinyafter-v4 traces and bodyfuture diagnostic report.
79:20fullfirstwordq.542<.6, bodypair>.997;98:5short«وماء» vs«وما أمروا».

- [x] Fresh79:20:400mspreroll withinoriginalDSP interval, minimal1300msspeech.
- [x] Fresh98:5: originalheadpause6,2500msspeech forword-boundary ambiguity.
- [x] Ifnofullheadproof, oneconcrete boundedextension only; retainallfailures.
- [x] Finalheadproof usespublicASR.6/alias/truncation/globalwordplan gates.

NoGTinput, no currenttimestampaccepted bydiagnostic. Allcallsledger≤6s,
no modelwholepass;canonicaltext/DBunchanged. Docs updated: task.md.

Initial4callscompleted:79(.4s+1.3/.6s+2.5) headq.390/.594 both<.6;
98(.2s+2.5) incorrectامرون, (.4s+4) fullوماامروا/min.99768.
Sameaudio/sameinitialcrop Base79headq.981 vsTiny.488: model-specificweakness.
Twofactor controls98: (.2s+4)postonly, (.4s+2.5)prerollonly, both run to
attributecause; no timestampaccepted andnoqualitylowered.

Completed6calls; cause98postcontext,79model-specificuncertainty.
Docs updated: task.md/report.md/STATUS.md; mainunchangedbydiagnostic.
