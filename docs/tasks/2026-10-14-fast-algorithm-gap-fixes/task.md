# Исправление локальных recovery FAST

Цель: исполнить шаги алгоритма при повреждённом начале и похожих началах,
отдельно проверить геометрию паузы и возможность Tiny/Base local timing.
Канонический текст, пользовательские БД, настройки, FULL и режимы не менять.

Прочитано: AGENTS/CLAUDE (идентичны), RTK, project.md/STATUS, исходный proposal,
fast-algorithm-cause-review report/task и real/fake evidence; текущие executor,
adapters, reads, worker. Serena manual прочитан, проект активен.

Решения: все дополнительные акустические входы только через общий ledger,
6s/65%/200%/32MiB; конечное расширение по конкретному свидетельству.
Слабый onset не становится точным временем из-за сильного later-word identity.
ASR timing включать только после измеренных temporal controls.

- [ ] Bounded CTC opening identity recovery и независимые конкуренты.
- [ ] Contextual pause proof после incomplete/noisy opening.
- [ ] Честная temporal uncertainty и измерение geometry без short bypass.
- [ ] Проверить/исправить Tiny/Base time-producing local stage.
- [ ] Одна заключительная фаза тестов и адресных real probes, отчёт.

Attribution: исходный HEAD 37f3309; параллельные FULL изменения в align.py,
align_tiled.py, full_windows.py, alignment_result.py, docs/project.md,
docs/STATUS.md и docs/full-performance.md не принадлежат этой задаче.
Коммитов/push нет. Проверка и блокеры будут записаны в report.md.

Продолжение после прерывания (retry): исходная попытка создала только task.md;
FAST source pristine относительно37f3309. FULL shared diff сохранён. Решения:
noisy first-word recovery требует независимый decoded remainder и дополнительное
целое слово; не принуждает недостающую фонему. Seam recovery разрешён только
после конкретного incomplete/noisy observation на той же паузе. Все proof
повторно проверяются validator. Геометрия оценивается для любых длин interval,
без зависимости от текстовой уверенности. Тесты — после связанных правок.

## Завершение достижимого скоупа

- [x] Реализована узкая CTC identity recovery и finite small extensions,
  independent rivals/negative gates/validator; missing physical onset не подделан.
- [x] Contextual pause recovery после incomplete distinct opening;
  fake positives/negatives и сохранение existing real109 proofs.
- [x] Измерение short geometry без bypass, исходный4×gate, честная диагностика;
  новый calibrated gate отклонён как необоснованно снижающий coverage.
- [x] Fresh Tiny/Base timing experiments: axes/query convention/exact BPE/
  adjacent crops/silence shifts/wrong targets. Capability не включена.
- [x] Заключительная suite625/8 + final FAST96, Ruff/diff-check и real pair.
- [ ] Реальная1:2 надёжная identity proof: extra canonical label не подтверждён.
- [ ] Независимая слуховая geometry/onset calibration и устранение79:26.
- [ ] Надёжный Tiny/Base local time-producing runtime stage.

Docs updated: этот task.md, report.md, scripts/JSON в этом каталоге.
Docs unchanged: project.md/STATUS.md принадлежат параллельной FULL задаче;
обновление сводного STATUS передано lead, чужие файлы не перезаписаны.
Изменены толькоfast_executor.py иtests/test_fast_algorithm_gaps.py.
Блокеры/точные пределы — вreport.md; задача не объявлена полностью исправленной.

### Поправка независимого ревью

Alternative seam теперь не уничтожает original distinguishing opening после
weak/wrong/truncated tail. Original plan/phrase/observation/scalar proofs
восстанавливаются до обычных future/finite expansion ветвей; sparse future
сохраняет opening_id. Новые4 meaningful regressions, final FAST100 passed;
финальная полная suite631 passed/8 skipped, Ruff/diff-check passed. Sparse7.

### Противоречащее продолжение

Сильное independently heard continuation другого/неизвестного слова нельзя
отбросить коротким seam даже с правильным previous-tail. Проверены strong
future/wrong, weak continuation и validator corruption. Родительский baseline
regression probe: четыре случая regression=false. Финальный FAST: 104 passed;
общая suite: 635 passed, 8 skipped; Ruff/diff-check passed. Model calls не
повторялись; real CTC 1:2, Base 79:26, Tiny/Base timing остаются открытыми.
