# Следующий шаг: исправить интеграцию, затем доказать пользу FAST

Это план, не выполненная переделка. Штатный единый FAST остаётся выбранным
пользователем; откат/смена режима не предлагаются автоматически.

## 1. Узкая интеграционная правка R1–R4

- Общий preflight explicit FAST и auto→FAST. Пока current snapshot неизвестен,
  пересчитывать auto, пропускать только fully edited/verified. Затем optional
  exact-key preflight должен готовить тот же bundle/PCM/DSP snapshot без acoustic
  reads и передавать его исполнителю, не делать второй decode/model selection.
- SSE summary и debug-log читать из review_required_ayahs/acoustic_reads,
  legacy fields использовать только для старых logs. Отдельно показать модель,
  accepted proofs, estimates и failed/cancelled/denied reads.
- Неизменяемый run snapshot создавать по actual algorithm при requested auto;
  requested/actual mode сохранить раздельно. Не переписывать старые runs/keys.
- Фактический CTC model ID снять в bundle; weights/runtime SHA не заменить
  именем выбора из ACTIVE_ID после decode.

Критерии: temporary old auto FAST без/с чужим ключом не skipped; same prepared
key skipped; mixed manual/auto merge сохраняет manual; fully manual skipped;
auto выбор reference/FULL/tiled не переобозначается как FAST. Qt worker проверить
через существующий signal-stub seam, отдельно отметить отсутствие native smoke.
Trace с2 unresolved и3 reads отображает2/3 и реальные окна; две auto→FAST записи
дают два history JSON и один latest. FP32/INT8 имеют верный ID и разные weights
identities. Failure-after-commit остаётся saved_with_diagnostics_error.
Запустить один итоговый suite после всех правок; не повторять без изменения кода.

## 2. Устранить причину отказов на одном локальном случае

Первый диагностический пример — CTC1:2: три pre-roll дали одинаковый missing-a.
Не начинать со смены порогов или массовой нормализации транслитерации.
Разобрать независимые greedy/beam и canonical/null/residual likelihood на
одном локальном PCM; проверить поддержку именно первой различающей единицы,
вариант наличия слабого начала, модельную ошибку и valid rendering. Если нужен
другой crop, расширение должно иметь конкретное локальное свидетельство.

Критерии: внешняя слуховая/word-onset разметка нескольких positives/negatives;
противоречащее первое слово и внутренний похожий фрагмент не подтверждаются;
минимальная целая distinguishing phrase сохраняется; confidence остаётся
минимумом essential units; timestamp не выходит из доказанной geometry.
Простой pre-roll fix не считать успешным — свежая проба его не подтвердила.

Раздельно разбить saved unresolved причины: нет кандидата, left-prefix mismatch,
noisy phoneme/word, terminal competitor, repeated owner, unsupported timing,
budget denial, strict blocked tail. Количество blocked priors — эффект остановки,
не доказательство тысяч независимых ошибок моделей. Сначала чинить первичный
отказ, затем измерять его влияние на хвост; не добавлять непрерывное model scanning.

## 3. Проверить полезность на согласованном наборе

После конкретного исправления — frozen prospective версия и один парный прогон
согласованных сур1/32/55/59/65/73/77/80/98/112 для каждой модели. Legacy не
возвращать в runtime: сравнивать с сохранёнными совпадающими artifacts либо
отдельным воспроизводимым git snapshot, никогда как fallback FAST.
Для mismatch audio/DSP/source честно обозначить непарность. Отдельные109 и79:26
проверяют повторные proofs и короткий decay; длинный2 проверяет stopping/budgets
после адресного улучшения, без повторения всего корпуса по умолчанию.

Критерии зафиксировать ДО прогона:0 locality/canonical/manual violations;
никаких новых крупных без-review расхождений на adjudicated примерах;
не падающее покрытие отдельно по CTC/Tiny/Base и сценарным группам;
меньшие ошибки estimates и меньшая доля blocked tail; время decode/DSP/inference/
decision отдельно. Reference midpoint disagreement не объявлять ошибкой без
прослушивания. Обязательные6с/65%/200% и finite recovery не ослаблять.
Если достигнута лишь механическая безопасность, так и написать; не объявлять
coverage PASS из-за нуля false positives или потому что tests зелёные.

## 4. Отдельные незакрытые исследования

ASR timing: независимые word onsets, shift-invariance и negative targets для
Tiny/Base. Пока критерии не пройдены, timing=False. Pause midpoint policy:
прослушать Base79:26 и matched controls коротких/длинных затуханий; определить
общий сигналовый критерий, затем prospective validation, без подгонки одного
порога к одному508мс примеру. Другой чтец и native packaged app — отдельные
smoke/quality checks. Это не повод подключить FULL/tiled под видом FAST.

Synthetic cross-ayah seam не обнаружен в114 канонических сур: protective test
можно добавить при изменении represent/planner, но он ниже подтверждённых R1–R3.
