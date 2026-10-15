# FAST: полноценно прочитанный предыдущий body не является opening target

Предыдущий turn — progress. Причина2:42 была установлена по реальному decode:
`ولا تشتروا` принадлежит внутреннему2:41, target42 — `ولا تلبسوا`.
Полные два слова уже распознаны в2,7s/min0,479, но общее первое `ولا`
держало `had_prefix`, приводя к лишнему4,2s read и остановке всей последовательности.

## Узкая negative rule

Predicate не принимает ни одной новой границы. Он только отклоняет candidate:

- предыдущий ayah подтверждён и непосредственно предшествует текущему;
- target требует больше первого полного canonical word;
- decoded first word совпадает с полным первым target word;
- **два полных слова** совпадают с внутренним body предыдущего, не его opening;
- first canonical text тоже совпадает, second отличается по буквам;
- все unit scores конечны и не ниже прежнего threshold, продолжение сильно
  противоречит текущему target.

Тогда сбрасывается ложный candidate blocking, но не подтверждается current.
Следующие candidates должны проходить прежние complete/quality/validator gates.
Partial/weak rival, same-letter variants и distinct current first word не
подпадают под это правило. Текст, формат БД, budgets и manual protection не меняются.

## Prospective изолированный sparse14

Полная2:36→42proofs,6added/0lost/0newconfident>500. Отказ ровно один —42,
read228/809500–812200. Opening42 затем подтверждён818020ms по actual
`фаттакун + ولا` context у следующей подходящей паузы. Процесс доходит до48.
Calls231→260 и поиск150,74→165,05s: вся работа выросла за новые аяты;
отвергнутый candidate больше не требует последнего4,2s окна.

Свежие остальные11cases дали ещё12proofs в59 (9→21), отказы на12 и23
за full previous-body rivals/min0,918 и0,871. Пары canonical second words:
`أخرجتم` versus `أخرجوا`, `الرحمن` versus `الله`; в2 —`تشتروا` versus `تلبسوا`.
На остальных сурах число подтверждений сохранилось.

Merged12cases/607rows:255→273proofs,18added/0lost, новых confident>500нет.
Прежние11reference outliers остаются нерешёнными и не удалены из метрики.
Calls891→983, поиск427,25→468,00s, inference ledger371,23→409,68s.
Coverage≤57,279%, input≤97,073%, каждый read≤5,5s, порядок и counts сохранены.
Нельзя объявлять ускорение всего корпуса: код прекратил бесполезные чтения
неподходящих candidates, но позволил выполнить больше полезной работы.

`rejected-2/other11.json/.summary.json/.log`, singleton/corpus comparison,
frozen source/manifest/ZIP сохраняют все данные. Merged файл не считается
повторным inference. Source guard и dispatch assertions проходили внутри bench.

## Все поддерживаемые варианты, новая парная проверка

Из текущего production sparse9 созданы before/after frozen sources,
отличающиеся только этим rejection и version marker. Никакие native/context
prototypes не попали в after. Все12surahs (agreed10 + полная2 +79),
4supported variants ×before/after:96cases/4856rows/1442actual calls.

| Модель | Proofs before/after | Calls before/after | Поиск с DSP, с |
|---|---:|---:|---:|
| Quran Phonetics INT8 | 2 / 2 | 120 / 120 | 52,72 / 52,63 |
| Quran Phonetics FP32 | 2 / 2 | 111 / 111 | 55,52 / 55,27 |
| Tarteel Tiny | 84 / 84 | 217 / 217 | 86,05 / 86,26 |
| Tarteel Base | 114 / 114 | 273 / 273 | 142,58 / 142,05 |

Все строки совпали, потерь/новых errors>500/decline events0. Отдельная пара
таймеров не доказывает ускорение. Данный failure воспроизвёлся у local Muaalem,
а supported variants пока не дошли до этих случаев или не дали такой rival.
Низкое число QPC proofs — проблема local evidence, не регрессия этого изменения.
Все actual dispatches, audio/reference/initial DSP hashes, source/canonical
guards и6s/65%/200% проверены. `.summary.json` присутствует у каждого terminal run.

## Production и дополнительная защита

После этих controls перенесён только negative predicate в main,
версия `sequential-fast-v3-sparse-10`; main до записи байтово сверён с before.
Нет Muaalem support, normalization/native onset/context rollout или смены default.

Затем добавлена более строгая защита: existing `arabic_units` сравнивает
canonical second-word letters без огласовок/орнаментов; marks-only difference
не может отменить blocking. Это только сужение rejection. Predicate-only replay
всех3реальных declines подтвердил разные letter forms; на96supported cases
declines и раньше не было. Новых neural calls для этой проверки нет, она
отмечена именно как replay (`orthographic-guard-replay.json`).

Подготовлены4observable regression checks на неизменном canonical112:
`ولم يولد` внутри3 versus `ولم يكن` в4, partial/weak rival иdistinct head.
`tests/test_fast_previous_body.py` **не запускался**: пользователь оставил
pytest до общего финала. Scoped Ruff clean.

За этап:108pipeline cases/5463rows/2425actual calls. Setup introspection
дважды обращалась к неверным именам API пути модели, без decode/inference;
после чтения source использован реальный `runtime_path`, установленный FP32
подтверждён. Эти read-only ошибки не включены в neural counts, скачиваний нет.

Docs updated: этот каталог, docs/project.md, docs/STATUS.md. Canonical/user DBs,
edited/verified, веса/default иcommit/push не затронуты. Общая цель активна:
temporal outliers, модель для local evidence, repeated prefix48 и дальнейшая
скорость ещё требуют работы; общий финальный pytest впереди.
