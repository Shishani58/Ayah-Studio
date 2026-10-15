# FAST: целое название буквы и native onset у отвергнутой паузы

Предыдущий этап — progress: парный supported-model контроль не нашёл регрессий.
Этот этап проверяет полную2, затем остальные суры. Production не менялся;
Muaalem FAST и новые правила остаются изолированными. Pytest не запускался.

## Полная2 с declared alif

Точное полное название «алиф» (native source-character phones0:5) открыло
0→3 подтверждения: аяты1/2/3, без confident errors>500ms. Полная запись
7086576,3125ms, 286строк, 13фактических local calls, max5,5s/call,
union22,4s/0,316%, input29,6s/0,418%. Это не полный acoustic pass.
`named-alif-2.json/.summary.json/.log` и comparison содержат полный результат.

Остановка на4 произошла до нужного чтения: DSP-пауза33760–34660 была
отвергнута по `decaying_tail_midpoint_unproved`. Затем алгоритм читал начало5
и его продолжение, а wasl forecast40600–46100 попал в середину текста4.
Поэтому эту остановку нельзя приписать плохому распознаванию открытия4.
Полные переходы и reads — `2-4-stop-evidence.json`.

Независимый right-edge контроль прочитал34460–36160/37360/38660:
prefix8/18/20 из20 units, min качества0,572/0,441/0,526; native onset
34703/34702/34700ms. Все calls подтверждены dispatch spy, модель не получает
целевую фразу; `declined-pause-2-4.json/.log` содержит сырые наблюдения.
Это диагностический контроль, не принятие boundary.

## Изолированный pause_native_onset

Frozen sparse-12 prototype допускает native timing только когда:

- DSP interval измерен, но отвергнут именно из-за decaying tail midpoint;
- adapter имеет native positions, план глобально различим и не contextual;
- независимо декодирован полный prefix, прежние unit-quality/beam gates пройдены;
- фактический read начинается у right edge с прежним200ms pre-roll;
- отдельный validator проверяет неизменный pause ID/interval, причину отказа,
  offset0 и `actual_sample_start + decoded_position[0]`.

Середина такой паузы не принимается. Forced posterior prefix, missing-opening
recovery, contextual shortening и будущий midpoint anchor для нового kind
отключены. Стандартные доказательства и все budgets сохранены. Исходник,
manifest/ZIP, constructor/driver SHA и post-run guard сохранены.

На полной2:3→14proofs, потеря0, 81reads, coverage1,624%, input2,716%,
max5,5s/read. Три новых reference outliers:4+503ms,9+1012ms,10+2214ms.
Их нельзя исключать из метрики. Independent DSP показывает низкую энергию у
GT и рост энергии возле native positions, но RMS сам по себе не доказывает
правильность phonetic onset. Для всех трёх сохранены exact GT из read-only
reference, canonical openings, полный proof/read и аудио с контекстом:
`native-onset-2-outlier-evidence.json`, `2-{4,9,10}-context.wav`.
Человеческое прослушивание не заявляется. Это требует timing policy/calibration,
а не объявления GT ошибочным или снижения порога.

## Сохранность полного набора

Затем свежий отдельный прогон остальных11сур, без изменения источника в процессе.
`native-onset-corpus.json` — упорядоченное объединение2 и other11, не новый
повтор этих вызовов. Comparison против прежнего sampled-onset12:

| Сура | Прежние proofs | Native + declared alif |
|---|---:|---:|
| 1 | 7 | 7 |
| 32 | 0 | 23 |
| 55 | 29 | 30 |
| 59 | 9 | 9 |
| 65 | 1 | 1 |
| 73 | 20 | 20 |
| 77 | 37 | 38 |
| 80 | 33 | 33 |
| 98 | 8 | 8 |
| 112 | 4 | 4 |
| 2 | 0 | 14 |
| 79 | 46 | 46 |

12cases/607rows:194→233proofs,39added,0lost. Из прежних подтверждений
изменились три:1:5 время21693→21681ms (error157→145),55:27 только ничтожная
разница float качества,77:31 качество0,571→0,416 при неизменном времени/error69.
Это зафиксировано отдельно в `native-onset-existing-changes.json`.

Calls579→739, поиск292,34→359,38s, inference в ledger243,52→309,27s.
Ускорение не получено; полнота увеличилась за дополнительную работу.
Максимальное покрытие57,152%, input ratio97,251%, read5,5s. Порядок,
канонические counts, audio/reference hashes, dispatch samples/ledger и budgets
проверены. Шесть новых reference outliers>500ms:32:11+1890,32:16+566,
32:20+1063,2:4+503,2:9+1012,2:10+2214. Прежний79:26+508 остаётся.
32:11 уже возникал у предыдущего named-alif32 prototype (+1867); его GT
попадает в полный нулевой сигнал, отдельный анализ сохранён на прошлом этапе.
Другие новые расхождения ещё требуют индивидуальной temporal проверки.
Новых confirmed→unconfirmed потерь нет; это не утверждение, что все233времени
правильны. Production перенос **закрыт** до разрешения temporal неопределённости.

За этап:13завершённых pipeline model/surah cases,893строки/752calls,
плюс3diagnostic right-edge calls =755actual calls. Merged корпус не посчитан
дважды. Static mapping/DSP не являются neural forwards.

## Следующие причины, не угадывание

Остановка2:18: глобальный canonical owner floor требует длинную фразу,
поскольку тот же prefix встречается внутри2:171. На нужной паузе слышимый
decoder prefix расходится с модельным wasl template возле nasal/consonant run;
рядом существуют внутраятные DSP-паузы. Wasl forecast ещё и обрезает начало
предыдущего различающего tail. `2-18-stop-evidence.json` сохраняет полный
материал. Это два различных ограничения: длина различающей фразы и
representation/decoder mismatch. По этим данным нельзя утверждать ошибку
чтеца, удалять согласные или расслаблять quality. Нужно отдельно проверить
waqf variants и decoder, затем минимальный полный context proof.

Static generality audit штатного disconnected-letter словаря обнаружил
13alif openings:2/3/7/10/11/12/13/14/15/29/30/31/32. Везде первая source
character mapping — полное `ءَلِف`, без arbitrary subword. Файлы
`all-alif-declarations.json/.log`; это только static evidence, аудио остальных
11новых openings не проверено. Driver пока использует прежние2/32 declarations.

Docs updated: этот каталог, docs/STATUS.md. Scoped Ruff clean. Source guard:
67frozen files и5driver/declaration inputs неизменны; pyproject marker добавлен
в archive для воспроизводимости resource resolution. AGENTS/CLAUDE не правились.
Commit/push, запись пользовательских БД/весов/канонического текста отсутствуют.
Общая цель активна: temporal policy, variants, скорость и финальный pytest
ещё требуют работы.
