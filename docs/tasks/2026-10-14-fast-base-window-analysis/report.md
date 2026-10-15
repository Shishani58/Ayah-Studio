# FAST: размеры окон и восстановление после пустого DSP region

14 октября2026. Продолжение общей задачи; цель ещё активна.
100 свежих model/surah запусков,4391 строк/1819 реальных local model calls.
Runtime правка внедрена только после анализа и33 парных corpus случаев.
Тесты pytest в этом продолжении не запускались: пользователь разрешил их
только в общем финале. Четыре регрессионные проверки подготовлены, Ruff чистый.

## Первый вопрос: что именно меняет короткое окно Base?

Frozen sparse-7 с текущим contradiction repair. ЧтецАш-Шатри, согласованные
1,32,55,59,65,73,77,80,98,112:275 аятов/серия. Начальное чтение1.0/1.3s и
следующее расширение2.0/2.5s проверены раздельно; третье4.0s. `speech_ms`
используется и для previous-tail, поэтому это factorial **общей локальной
политики**, а не изоляция только decoder opening. Никаких изменений thresholds.

| Speech ms | Proofs | Потери прежних | Добавлено | Confident>500ms | Время |
|---|---:|---:|---:|---:|---:|
| 1300/2500/4000 | 84 | — | — | 0 | 79.32s |
| 1000/2500/4000 | 96 | 15 | 27 | 1 | 86.93s |
| 1300/2000/4000 | 84 | 0 | 0 | 0 | 80.70s |
| 1000/2000/4000 | 85 | 15 | 16 | 0 | 85.25s |

Пример73:6:1.3s decode=`إن ناشر`, после2.5s=`إن ناشئة الليل`, правильная
различающая фраза слышна.1.0s=`إننا` (quality.584), поэтому first-word anchor
теряется и дальнейшее увеличение на этой паузе не разрешено. На73:13 shorter
выдаёт `وطعامة` вместо `وطعاما`; следующая остановка оставляет13–20 estimates.
Это разные ошибки локального ASR и последующий cascade алгоритма, а не восемь
независимых ошибок модели на аятах, которые она вообще не читала.

Short-initial снова подтверждает80:37=156610 противGT155938, +672ms,
confidence.9903. Тот же midpoint уже дал ошибку уTiny в предыдущей серии.
Word identity и DSP temporal geometry нельзя смешивать. Ни один global window
variant не принят. Подробные lost/added/read/events — `analysis.json`.

## Второй вопрос: не отбрасываются ли только огласовки?

По точным локальным vocab/BPE bytes найдены10 word observations, где min quality
ниже.6 только из-за Unicode marks, удаляемых `arabic_units`. Например33608 —
tanwin kasr,11703 —shadda. Но многие такие observations уже имеют неверные
буквы/не принадлежат planned proof (`الميز`, `ملميزان`, `تبروا`, `فتنظر` и др.).
Это не10 новых допустимых доказательств. В роли букв/marks проверены точные UTF8
байты, включая split tokens; incomplete byte crops помечены unsupported.
`token-roles.json` содержит scalar diagnosis, не replay новых boundaries.
Quality policy не ослаблялась и marks автоматически не исключались.

## Третий вопрос: почему Base77 обрывается на16?

После подтверждённого15=start43760 tempo forecast16=47075,
region44423–52048ms. В этой области **нет принятого DSP кандидата**, модель
на16 вообще не вызывается; next-stage word timing уTarteel отсутствует.
GT16=48141 находится внутри region. Значит, это не доказанный неверный tempo
cap. Глобальный DSP даже не сформировал local decision около45–53s.
Физическая wasl/короткая тишина без прослушивания не установлена.

Первая допустимая следующая пауза54060–54300, midpoint54180, соответствует
18 (GT54335). Предварительное предположение о границе16/17 уточнено по данным;
оно не использовано как label. `empty-region-cause.json` сохраняет точные числа.

## Решение и ограничение чтения

Sparse-8: только когда есть≥2 подтверждённых anchors, обычный tempo region
пуст и adapter не даёт word positions, разрешается **один** DSP-кандидат справа.
Его начало не дальше `right + predicted_span * expansion_lookahead`;
по умолчанию horizon равен двум прогнозируемым длительностям. Geometry должна
быть допустимой. Исходные1.3/2.5/4s,8 candidates,6s/65%/200% и общие proofs
сохраняются. Новое поле SELECT_CANDIDATE объясняет дополнительные bounds.

Tempo только задаёт гипотезу; не подтверждает владельца или время. Если услышан
уникальный следующий аят, прежний validator проверяет его независимо, а
пропущенные остаются estimates/conf0/review. CTC с word positions сохраняет
прежний anchored_wasl recovery: новый ранний probe для него отключён.
Нет scan всей записи, forced full pass, переключения режима или новой модели.

## Свежая парная проверка

До/после:33 model/surah случаев,561 строк/модель — основной набор и **полная2**.
Все recording/DSP SHA равны, reference открыт после inference. Independent
dispatch spy сверяет каждый PCM slice с ledger. Source архивы фиксированы.

| Модель | Proofs до→после | Потери/изменения прежних | Confident>500 | Все ошибки>500 |
|---|---:|---:|---:|---:|
| Phonetics | 2→2 | 0/0 | 0 | 536→536 |
| Tiny | 68→68 | 0/0 | 0 | 479→479 |
| Base | 86→92 | 0/0 | 0 | 460→447 |

Выигрыш толькоBase77:15→21 proofs. Новые18/19/21/22/23/24 имеют ошибки
155/38/14/49/19/39ms.16/17/20 остаются review, не объявлены распознанными.
Большинство остальных ошибок таблицы — estimates, а не confident timestamps.
Прежние времена и confidence подтверждений точно сохранены.

Дополнительные два probe запускают последующую полезную работу: Base11-сурный
inference+matcher85.90→92.11s в одиночной серии, около+7.2%. Это улучшение
полноты/ошибки с измеренной ценой времени, не ускорение Base. Предыдущие
оптимизации вычислений sparse-7 сохраняются. Phonetics/Tiny timings практически
не изменились; общее обещание ускорения всем моделям не делается.

Prototype первоначально допускал probe для любых timing capabilities; на этих
33 случаях CTC probe count=0. Final production guard сохраняет CTC path.
Свежий final77 на всех3 моделях точно повторил rows/boundaries/PCM dispatches
prototype (`final-77-comparison.json`). Полный33-case replay final code не
переобозначен как новый замер: отличие CTC guard не срабатывает в измеренном
corpus, а поддержанный CTC flow сохраняется структурно и отдельным unit test.

## Логи, проверки и открытое

- `run.py`, `variant.py`: точные argv factorial; `*.log`, JSON и summaries.
- `next_pause_run.py`: последовательные fresh controls/candidate; source archives,
  manifests, сравнения и actual reads. Initial unbounded pilot77 — отдельный
  артефакт; его source восстановлен с точным SHA из manifest. Он не заменяет
  bounded corpus33. Final source архив соответствует production sparse-8.
-100 cases: max actual input5.5s, union43.96%, summed input57.11%,0 нарушений
  последовательности. Canonical DB114/6236 и SHA неизменны в benchmark.
- Новые tests/test_fast_empty_region.py: pending до общего финала. Проверяются
  retained missing current, wrong owner, bounded horizon и supported wasl.
  Ruff own paths/diff-check прошли; AGENTS/CLAUDE идентичны. Не commit/push.

Задача в целом не закрывается этим локальным улучшением: Phonetics имеет2 proofs
на основном275, initial-a1:2 всё ещё отсутствует в acoustic output; надёжные
ASR word positions и DSP temporal calibration не реализованы. Следующий
приоритет — независимые локальные чтения установленной альтернативной CTC
моделью, чтобы отделить ограничения acoustic модели от дальнейших правок planner.

Docs updated: docs/project.md, docs/STATUS.md, файлы этой задачи.
Пользовательские БД, настройки, основной model selection и FULL не изменялись.
