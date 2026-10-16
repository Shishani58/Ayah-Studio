# Исполнение системного исследования FAST

Поправка после обратной связи пользователя: закрытие общей цели по этому этапу
было преждевременным. Приведённые ниже измерения сохраняются как завершённый
исследовательский этап, но не как выполненная задача улучшения рабочего FAST.
Исполнение продолжено в [новой задаче](../2026-10-15-fast-production-improvement/task.md).

Цель: выполнить M0–M7 [поручения](../2026-10-15-fast-systematic-research/plan.md),
проверить причины bootstrap/каскадов и переносить только подтверждённые решения.

Прочитаны: AGENTS/CLAUDE, RTK, plan.md целиком, research.md, актуальные
FAST/executor/pipeline/corpus/model разделы project.md и последние FAST записи
STATUS, отчёт sparse12. Они задают ограничения и исторический baseline.

Решения: не трогать FULL/UI/defaults/БД; работать с отдельными source roots;
reference только после predictions; pytest и contract suites только в M7.
Исторические результаты не объявлять новыми. Проверять реальные imports.

- [x] M0: manifests, baseline, corpus/measurement contract, причины/каскады.
- [x] M1: представления 114 сур, адаптеры, причинные локальные опыты H1–H4.
- [x] M2: identity/cue/time, независимый validator и attribution.
- [x] M3: выбранные bootstrap/recovery гипотезы и парные результаты.
- [x] M4: временной протокол,18unlabelled WAV/SVG, внешняя слуховая зависимость.
- [x] M5: reference-risk/coverage, calls/PCM и дерево памяти,3тёплых повтора.
- [x] M6:86пар/172modelcases и actualnegative controls; положительный
  межчтецовый gate не пройден. F не достигнут; прототип не перенесён.
- [x] M7:772passed/8skipped текущейsuite,10isolatedcontractpassed,
  Ruff/diff/guards/sync; итоговый отчёт иSTATUS.

Итог: исследовательский цикл завершён, production не менялся. Шатри112/114
prototype даёт по2прямых proofs INT8/FP32, каскад0; у других чтецов нет
положительного применения. Небезопасно называть это общим исправлением FAST.
Причины и измеренный предел — [report.md](report.md), всеH1–H8 имеют решения.

Во время работы общую копию меняли другие задачи (progress/FULL/UI/release).
Их изменения сохранены; initial/current roots, diff иAST guards разделены.
Freshcurrent smoke8cases:6сопоставимых rows/dispatches совпали точно;
основной measured corpus относится к frozeninitial. Ничего не сбрасывалось.
Независимая слуховая разметка остаётся внешней зависимостью; пакет подготовлен,
прослушивание не заявлено. Низкое покрытие и Base79:26+508ms остаются открытыми.

Docs updated: task.md/report.md/baseline.md/attempts.md, измерительные и
evidence/annotation/adapter документы в этой папке, docs/STATUS.md.
Docs unchanged by this task: docs/project.md — production поведение не менялось.
Канонический текст/индексы,224data guards(98DB+126MP3), AGENTS/CLAUDE сохранены.
Эта задача не выполняла commit/push и не меняла SQLiteформат. Параллельное
изменение HEAD общей копии до3dc2864 сохранено в final-audit.json.
