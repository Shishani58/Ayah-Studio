# Исполнение системного исследования FAST

Цель: выполнить M0–M7 [поручения](../2026-10-15-fast-systematic-research/plan.md),
проверить причины bootstrap/каскадов и переносить только подтверждённые решения.

Прочитаны: AGENTS/CLAUDE, RTK, plan.md целиком, research.md, актуальные
FAST/executor/pipeline/corpus/model разделы project.md и последние FAST записи
STATUS, отчёт sparse12. Они задают ограничения и исторический baseline.

Решения: не трогать FULL/UI/defaults/БД; работать с отдельными source roots;
reference только после predictions; pytest и contract suites только в M7.
Исторические результаты не объявлять новыми. Проверять реальные imports.

- [ ] M0: manifests, baseline, corpus/measurement contract, причины/каскады.
- [ ] M1: представления 114 сур, адаптеры, причинные локальные опыты H1–H4.
- [ ] M2: identity/cue/time, независимый validator и attribution.
- [ ] M3: выбранные bootstrap/recovery гипотезы и парные результаты.
- [ ] M4: временной протокол, пакет независимой разметки, ограничения.
- [ ] M5: risk/coverage, calls/PCM и дерево памяти, холодные/тёплые повторы.
- [ ] M6: подтверждающий корпус применимого scope и отрицательные контроли.
- [ ] M7: общая suite, Ruff/guards/sync, итоговый отчёт и STATUS.

Текущая стадия: M0. HEAD 37f3309f76fa077651239fe4892575de3b5d1fa6;
рабочая копия грязная, AGENTS/CLAUDE идентичны. Подробности — checkpoint.json.
Блокеры: независимая слуховая разметка отсутствует; не блокирует другие ветки.
Docs updated: этот task.md (работа продолжается).
