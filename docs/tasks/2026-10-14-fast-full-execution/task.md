# Полное исполнение FAST v2

## Scope
Исполнение M0–M6 proposal.md v2; production switch и удаление legacy только после M5.
Без commit/push, новых deps, изменений SQLite-схемы, canonical/user data.

## Прочитано
AGENTS.md/CLAUDE.md, RTK.md, proposal.md целиком, task исходного плана,
fast-contract-review/task.md, docs/project.md, актуальный STATUS — контракт и baseline.
Serena initial_instructions: проект активен.

## Решения M0 до кода
Recovery: максимум 8 кандидатов на аят, 3 ступени речи 1.3/2.5/4.0 s,
с паузой максимум 6 s. Расширение только после услышанного общего префикса.
Первый onset: максимум 8 DSP гипотез; нет фиктивного 6000ms.
Wasl: один прогноз от подтверждённой опоры, до двух аятов и 6 s.
Cache: 32 MiB; eviction не возвращает budget. Hard budget 6s/65% union/200% sum.
Strict stop: неразрешённый аят блокирует акустические подтверждения хвоста.
Snapshot: /var/folders/xr/sk3g_x_53rdg9wkyjdkq44tc0000gn/T/surah-fast-full-baseline-68_upa0h.
Сравнивать с ним, не HEAD; сохранять чужие изменения.

## Checklist evidence — итог
- [x] M0: corpus40, snapshot, frozen policy, fresh paired legacy37, isolation.
- [x] M1: общий guard/cache/evidence/validator; M1a сохранён.
- [x] M2: независимые CTC/Tiny/Base acoustic adapters и реальные прогоны.
- [x] M3: finite sequential executor, 46 сценариев, corpus/replay/edges.
- [x] M4: experimental integration/run-key/bundle/save/cancel/manual merge;
  штатная активация и exact-key preflight остаются за M5/M6.
- [x] M5: все 111 пар измерены, ворота пересчитаны независимо.
- [ ] M5 acceptance: coverage FAIL у всех трёх; listening и независимые
  acoustic wasl/похожие начала subgroup ground-truth не выполнены.
- [ ] M6: production switch/removal не выполнены из-за M5; isolation сохранена.

## Проверка / блокеры

Обычный pytest текущего workspace независимо у родителя: **738 passed,
7 skipped, 16.88s**, без ACTIVE_ID plugin/reset; число включает параллельные
FULL-тесты. Наши 46 executor scenarios passed, 2.03s. Scoped Ruff/diff-check чистые.
Все 49 protected SHA сохранены; AGENTS/CLAUDE идентичны.

Реальный frozen v3: 3×40 записей, по986аятов; fresh legacy: 3×37, по564аята.
Парное подтверждённое покрытие: CTC276→9, Tiny395→132, Base411→208 — FAIL.
No-review >500ms: 6→0, 8→0, 9→1; остаточный Base81:15,+607ms уже был в legacy.
Новый путь: ноль contract violations. V4 observation replay:120/120 совпали
по результатам/окнам/PCM SHA; это не повторный acoustic benchmark.15 реальных
v4 edge probes прошли;500ms у всех моделей:0 вызовов и все confidence0.

[Полный отчёт, команды и ограничения](report.md),
[независимое ревью](independent-review.md), [attribution](changed-files.json).
Качество недостаточно для production migration; условия proposal v2 не изменены.

## Docs
Docs updated: этот task.md, report.md, docs/project.md, docs/STATUS.md,
исходный fast-execution-contract/task.md (только актуальный статус).
Proposal v2 не переписан. Independent-review.md принадлежит родителю.

## Выполнение и промежуточные проверки

- M1–M3: `fast_executor.py` — один последовательный experimental executor,
  `fast_adapters.py` — независимые акустические операции. Legacy loops не вызываются.
  Модельные версии и config/PCM/text/DSP snapshot отделены от provenance сохранения.
- `LocalEvidenceReads`: один ledger, ошибки/отмена оплачиваются, cache ограничен
  32 MiB resident containers; удаление не возвращает sum/union. Legacy cache eviction
  не включён: до M5 production inference behavior сохраняется.
- Fake-сценарии используют настоящий канонический набор аятов; покрывают порядок,
  whole-word phrases, 2:11/13/68–70, future block, wasl, strict stop, short audio,
  cache/cancel/failure, временную геометрию и end-to-end temporary DB.
- M4 делегирован отдельному исполнителю с ownership writer/ProjectDB/callers.
  Реализованы truthful commit outcomes, duration/manual merge под транзакцией,
  immutable history v2 с полным trace, failures sidecar, exact resume utility.
  Web/Qt без известного current key не пропускают auto FAST; полностью ручные
  суры защищены. Production activation/preflight сохраняются gated до M5.
- P18: импорт `server` больше не читает пользовательские настройки/не меняет
  ACTIVE_ID; convenience ASGI app загружает settings в lifespan. Явный create_app
  сохраняет прежнее поведение. Тест отдельного процесса проверяет отсутствие доступа
  к settings при импорте.
- Изолированный промежуточный pytest с plugin setup ACTIVE_ID=quran-phonetics:
  701 passed/7 skipped. После P18 обычный `rtk proxy .venv/bin/python -m pytest -q`
  без plugin/reset: 722 passed/7 skipped, 10.10s. После этого добавлен e2e тест;
  окончательный общий прогон приведён выше и в report.md.
- SHA49 canonical/reference files совпадают со snapshot; AGENTS/CLAUDE идентичны;
  git diff --check clean. `protected-check.json` хранит результат.

## Корпус и неизменяемые ворота

Фиксированный реальный корпус: Шатри 78–114 (564 аята), дополнительные1/2/9
(всего40 сур986 аятов). Веса трёх установленных моделей; без скачиваний.
Исторический baseline `tarteel-tiny-fast/comparison.json` имеет одинаковые audio/DSP,
но три source SHA отличаются от начала этой задачи; поэтому выполнен свежий legacy37.
Не понижаем M5 thresholds по результатам. Обязательны нулевые contract violations,
нерастущие >500ms confirmations и неубывающее coverage отдельно для каждой модели.

Первый diagnostic corpus v1: 986 аятов; confirmations CTC9/Tiny71/Base114,
крупные confirmed disagreements0/0/1 (Base81:15,607ms), contract violations0.
Это НЕ критерии готовности. По независимому ревью исправлены ошибки реализации:
CTC cross-word collapse, whole-word wasl-tail, prior duration после confirmed onset.
Corrected/final результаты и ссылки приведены в report.md; исходный corpus.json сохранён.

## Ограничения проверки

Слуховое подтверждение новых disagreements не имитируется. Другой чтец локально
не найден в доступном аудиокаталоге; no-intro/repeat/truncation/silence дополнительно
проверены как явно производные stress inputs, не как новые эталонные записи.
Qt runtime не установлен; worker проверен сигнал-заглушками, native GUI не запускался.
