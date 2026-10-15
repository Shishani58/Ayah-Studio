# Head crop diagnostic: модель против длины контекста

Послепрерыванияsession8888подтверждёнterminalexit0; ничего не перезапускалось
из-заотсутствиянаблюдения. Всего6fresh localcalls:2для79,4для98.

79:20:400mspreroll/1300speech даётfirstwordquality0,390;
600mspreroll(ограниченDSPstart)/2500speech даёт0,594. Оба<ASR0,6.
Sameoriginal1500msPCM crop91840–93340: Tinyq0,489,Baseq0,981.
Правильноераспознаваниестрокитекста не заменяетconfidencegate. Универсальное
«добавитьpreroll» не доказано; модельнаянеуверенностьTiny здесьсущественна.

98:5:200preroll/2500speech →«وما امرون»,неполное/неверное начало;
400preroll/4000speech →«وما امروا»,fullplan/min0,99768.
Факторныеконтроли:200preroll/4000speech тожеуспешен/min0,99765;
400preroll/2500speech по-прежнему«امرون». Значит, сработалpostcontext,
а не сдвигначаласреза. Первыйshortcrop«وماء» смешивает/достраиваетграницу
слов; послеcontext полнаяразличающаяфразавосстановлена безправокбукв.
Referenceиспользованпослечтений: originalDSPmidpoint35290,GT35212,error+78ms.
DiagnosticничегонеписалвDB/непринималboundary. Causalproductionrecovery и
96парныхконтролей находятсявследующемtask2026-10-15-fast-head-word-boundary.

ASRwholeheadqualityисходногоوماء0,5196, аwindowaverage0,9037; этичисла
нельзяподменятьдругдругом. Подтверждениетолькоfullphrasequality≥0,6.
Всеsource/log/JSON иsame-crop-model-comparison сохранены, Ruffclean,
pytestне запускался. Docs updated:task.md/report.md/STATUS.md.
