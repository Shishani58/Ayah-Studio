# Адресное дочитывание и восстановление FAST

## Цель / scope

Измерить и перенести минимальное улучшение рабочего FAST с установленной
Tarteel Base. Сохранить quality/temporal gates, локальные бюджеты и ручные метки.
Не завершать задачу отчётом о пробах без production результата.

## Основания

Прочитаны AGENTS.md/CLAUDE.md, RTK.md, docs/project.md, docs/STATUS.md,
model-reading-audit/plan.md и findings-deep.md, task предыдущей
fast-production-improvement. Они задают контракты, контроли и причинный 2×2.
Первоначальный report уточнён findings-deep. На старте предыдущая задача ещё
работала (PID36242, изолированный F replay); затем перенесённый F13 принят
за baseline. Текущий dirty checkout сохраняется.

## Решения

- Сначала frozen source; prospective F baseline отдельно от фактического current.
- Reference только после предсказаний, read-only; окна определяются DSP и decode.
- A/B/C/D: baseline / адресный cue / rejected geometry identity / оба.
- Identity не подтверждает midpoint. N-best отложен до измерений greedy recovery.
- Все прежние исследовательские записи development. Контроль назначить целыми
  записями до подбора правил, фиксировать hashes.

## Чек-лист / проверка

- [x] Текущий baseline после завершения предыдущего переноса, source/runtime/PCM/DSP.
- [x] Локальный 2×2 executor на суре2, отрицательные контроли.
- [x] Пофайловые независимые gains других чтецов; blocks/ошибки/бюджеты/стоимость.
- [x] Широкий corpus /114сур /legacy37; не присваивать прежние gains.
- [x] Узкий production перенос, fingerprint/resume/diagnostics.
- [x] Целевые и общий финальный checks, рабочий smoke, protected-data/sync.

## Блокеры / ограничения

Нет независимых human onset/cut labels: physical temporal accuracy не заявлять.
Рабочий checkout содержит изменения других чатов; сохранять их при переносе.

Docs updated: [docs/project.md](../../project.md), [docs/STATUS.md](../../STATUS.md), [report.md](report.md), этот task.md.

## Checkpoint: изолированные прототипы

- Предыдущий F перенесён в production v13. Frozen A совпадает с ним AST
  кроме VERSION; отличия align/model_store просмотрены: FULL/MFA ветки,
  Base loader/runtime неизменны. Свежий snapshot фактического current на2/80
  дал те же owners, времена и actual PCM dispatches (current-baseline-equivalence).
  Это отделяет прежний прирост65/73 от данной задачи.
- A/B/C/D на2/1:2→3 proofs на2 только от дочитывания; C identity-read2:4
  не добавляет временных proofs, D имеет тот же прирост и больше calls.
- B первоначально не покрывает single-word планы; B2 учитывает clipped/glued
  high-quality ASR word + измеренную продолжающуюся речь у края. Все starts
  и внутренние фрагменты перечисляются симметрично, cue не выбирает owner.
- B2 development80:10→38 proofs, но новый80:37 имеет benchmark error672ms,
  nonstationary DSP radius530ms. B2 отвергнут для production. B3 сохраняет
  identity/order внутри, новые широкие нестационарные midpoints публикует review
  по прежнему radius cap500ms. Правило зафиксировано ДО чтения control результатов.
- B3 development exact-PCM replay: все окна совпали, только80:37→review;
  это replay реальных B2 observations, не новые inference.19 поведенческих
  тестов прошли в изолированном snapshot; scoped Ruff чистый.
- Назначены и скачаны whole-record controls55/79 уГамиди/Хусари/Миншауи;
  первые предложенные2/3/36/67 отсутствовали. Никаких control crop/threshold tuning.
- Negative audio:2полных FAST executor (нули/чужая сура) и2локальных
  необходимых opening-контроля (1:4/внутреннее2:5) отвергнуты.
- Запущены fresh114 A/B2, fresh межчтецовые пары и fresh legacy37 с
  нынешней Base и историческим search/proof. Финальные gates ещё впереди.

## Продолжение 16 октября: объединение с production F13

- Пользователь поручил закончить проверку и перенести совместимый прирост в рабочий код.
- После32 готовых записей BS:93 новых proofs, прежние times/anchors сохранены;25:21 новый reference delta878ms. Диагностика: quiet pause2260ms, radius1130ms. Identity правильная, но midpoint физически не определён тишиной.
- BS2 меняет только публикацию новых recovery chains: прежний radius cap500ms применяется также к stationary quiet; identity/time остаются в диагностике, публикация review. Без reference-specific порогов, новых model reads или исправлений timestamps.
- Свежий114 BS продолжает работать на неизменном snapshot; BS2 проверяется через exact-PCM replay тех же независимых наблюдений и свежий рабочий smoke. Контроли67/77 ещё не просмотрены.

- BS2 промежуточный gate37 не проходит: новая37:50 delta501ms при radius430ms; прежняя37:22 (уже A delta1011ms) переходит в review из-за wide pause. Порог под501ms не подгоняем; перенос всей ASR-head ветки отложен.
- SF — минимальная причинная абляция: только один дополнительный следующий DSP-кандидат из S, без partial-ASR-head кода; тот же существующий cap500ms для recovery publication. Источник зафиксирован до просмотра67/77. Проверяется, достаточно ли этой добавки для безопасного production прироста.

- Свежий SF: Шатри25 2→7,37 19→25,80 10→33; Миншауи79 2→5. В этих парах нет новых >500ms и потерь прежних anchors/times. Только DSP fallback остаётся кандидатом переноса.
- Из-за отказа BS его широкая серия остановлена после42 записей; partial не объявляется завершённым114. Baseline A114 завершён. Для SF114 — counterfactual exact-PCM reuse только реально запрошенных независимых observations из A/BS; отсутствующий crop требует fresh whole-record FAST run. Reference только после результата. Fresh SF совпадение owners/times отдельно проверяется. Это не свежий inference114 и не wall-time benchmark.
- Финальный PF (v14) сохраняет только S+review guard. Публикационный guard выполняется до прежнего weak-prefix guard, чтобы тот не потерял trigger proof. Полный PF replay и fresh app smoke проверят итоговую композицию с F13. Model/DSP search от SF не меняется.

- Перед финальным replay PF усилены только проверки provenance: origin должен быть >=1; прежний body→head retry у добавленной паузы тоже сохраняет bounded origin. Добавлен красный/зелёный regression для порядка publication/weak guards: SF ошибочно оставляет downstream wide proof, PF отдаёт review.11 новых проверок прошли. Acoustic reads не меняются.
- Изолированный targeted run с нынешними repository tests:246 pass/6 fail из-за чужих обновлений pipeline/ProjectDB после snapshot (unknown-mode/edition protection), не FAST. Проверить повторно на реальном current после переноса; не копировать старые pipeline/DB файлы.
- Чтобы не повторять full PCM decode/DSP под нагрузкой хоста, недостающие наблюдения снимаются реальной Base на независимо hashed frozen PCM и сохранённом DSP pool A. Отдельный frozen_bench показывает decode/DSP=0 как supplied inputs; это не wall-time benchmark. Для стоимости отдельные serial warm/profile runs без этой оптимизации.

- Финальный способ добора уточнён до запуска: hybrid executor PF использует сохранённое независимое observation только при точном совпадении PCM SHA/длины. При отсутствии ключа вызывает реальную установленную Base, отдельный dispatch-spy фиксирует новый PCM. Для каждого logical read сохраняется происхождение; полный logical ledger/budgets учитывает и replay reads, без обхода32MiB runtime cache. Это НЕ fresh whole-record29 и не time benchmark. Frozen-bench подготовлен, но не запущен; повторно измерять известные ASR окна нет необходимости. Контрольные27 пары и warm/API остаются полностью fresh.

- Итоговый PF quality gate141пар прошёл: Шатри1186→1422 (+236), Гамиди107→108, Хусари47→54, Миншауи15→24. Новых >500ms, потерь/сдвигов прежних proofs нет;27 прежних positive deltas >500ms остаются.6236 canonical rows, всеhard budgets/zero full-emissions/fallback соблюдены.
- Guarded перенос F13→F14 выполнен в существующую main без коммита. Все982 production pytest прошли/10skip; scoped Ruff/diff-check чистые. Core205 и11 новых regression cases дополняют фактический current checkout run.

## Завершение

- Рабочий FAST — sparse-14-next-dsp-recovery, production SHA65301e9121f890565db34bffde05002896ced2636d58af8aedf02ccf4d5d4896. Код/11regression cases вmain; коммитов/пушей нет.
- Финальный141-pair gate,982pytest/10skip, Ruff/diff-check и real API/save smoke пройдены. Actual PCM/auto times совпали с измереннымPF;42rows иполные edited/verified review/time сохранены; новыйresume key. Wrapper diagnostic JSON текущегоGUI учтён после исправления test harness; повторныйреальный прогонпрошёл.
- Final audit:47dataDB,141audio,4references,10Base assets/Silero hashes неизменны; AGENTS/CLAUDE идентичны. Пользовательские проекты не трогали, только временные smoke fixtures.
- Whole114:236 новыхproofs;27 дополнительных fresh pairs:ещё17; нет новых >500ms или потерь/сдвигов прежних proofs.27 прежних подтверждённых расхождений не исправлены. Fresh legacy37:410legacy accepted/9bad,390F13/4bad,439F14/4bad; всех>500ms66/147/95, semanticsразные.
- Цена:logical reads3167→3742(+18.2%);warm paired median+13.4%,p954.279x. Soft ориентирp95≤2x НЕпройден;80median10.13→27.73s(23→69calls),первая79имеет106.45sвыброспри одинаковыхокнах(прочие17.47/16.34), причина неустановлена. Ускорение/эквивалентнаяlatency незаявлены. TreeRSS1570.9→1585.4MiB;profilesencoder/decoder/other вPF-costs.json.
- Блокеров переноса нет. Ограничения: дороже успешное продолжение, физическая onset accuracy без human labels не проверена, старые расхождения остаются. Для уже открытого GUI нужен перезапуск; существующие пользовательские auto DB сами не пересчитаны.

Docs updated: docs/project.md, docs/STATUS.md, report.md, task.md.

- Завершены все собственные worker/controller процессы. Удалены только два собственных регенерируемых PCM cache по строгому совпадению birth time сB2/BS launch и114 ожидаемыхnpy; освобождено12026768064bytes. Source/archive/JSON, исходноеаудио, модели и smoke fixtures сохранены. PF-source.zip FAST bytes совпадают сproduction SHA.
