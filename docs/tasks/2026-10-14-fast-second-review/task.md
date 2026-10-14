# Независимое второе ревью FAST

Цель: проверить исходный план и выполненный experimental FAST, найти причины
потери полезного покрытия и подготовить проверяемый план исправлений.
Scope: код исполнителя/адаптеров/guard, identity/writer/маршруты, тесты и
сохранённые корпусные доказательства. Production, БД и история прежних запусков
не изменяются. Коммит и push не выполняются.

Прочитано: AGENTS.md/CLAUDE.md (идентичны), RTK.md, proposal/task исходного
плана, task/report/independent-review/changed-files выполненной работы,
docs/project.md и STATUS.md. Первые задают цель и историю решений; код,
адресные пробы и исходные JSON проверяют утверждения независимо.
Serena активна; codebase-memory MCP недоступен. Используются Serena и rg.

Решения: проверять дефекты на fake-acoustics и сохранённых реальных
observations; не повторять дорогостоящий полный модельный корпус ради уже
доказанного coverage FAIL. Новый акустический материал требуется только
следующей реализации нового matcher/recovery, а не подтверждению этого FAIL.

Чек-лист:
- [x] Проверить причины отказов, phrase/quality/first/wasl/recovery.
- [x] Пересчитать парные метрики и ограничения replay/edges.
- [x] Проверить identity, отмену, commit, edited/verified и маршруты.
- [x] Выполнить адресные пробы и тесты, сохранить review.md.
- [x] Зафиксировать проверки, ограничения и статус документации.

Блокеры: слуховая adjudication и другой чтец не предоставлены;
native Qt и packaged runtime требуют отдельного окружения.

Проверка: 186 passed / 2.66s адресного suite; probes.py выполнен успешно;
Ruff для probes.py и git diff --check чистые. Независимо подтверждены111пар,
120уникальных результатов и совпадение текущего executor SHA с replay.json.
Все49 protected SHA совпали с protected-data.json исходного snapshot;
AGENTS.md/CLAUDE.md идентичны. Дорогие model reruns не выполнялись.

Результат: семь находок в review.md, версия следующего плана с prospective
критериями. M5 остаётся FAIL; implementation/production switch не начаты.
Отдельно уточнено, что replay не пересчитывает initial DSP decisions.

Docs updated: новый task.md/review.md и отдельная запись docs/STATUS.md.
История предыдущих задач, production-код и данные unchanged.
