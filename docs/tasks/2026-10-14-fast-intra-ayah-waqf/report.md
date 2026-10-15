# FAST: услышанный контекст разрешает расширение, но не заменяет полное слово

Предыдущий turn — progress. Production не менялся; Muaalem и sparse-13
остаются изолированными. Pytest не запускался по требованию пользователя.

## Причина остановки2:18

Canonical opening `صم بكم عمي فهم لا يرجعون` требует6полных слов: похожая
фраза встречается внутри2:171. Canonical seam гораздо короче: последнее
целое `يبصرون` из17 + первое целое `صم` из18, и эта пара глобально различима
в выбранной суре. Static full phonetizer exactly reproduced native mapping;
всё — из неизменного текста и quran-transcript0.6.4.

Сохранённое opening2,7s сначала давало первые5phones `صُممم`, min0,740.
Они совпали с whole-word waqf representation, но в наблюдении дальше были
дополнительные фонемы. Это не доказывает, что чтец остановился или что слово
полностью прочитано в нужном режиме. Поэтому waqf shortcut для acceptance
не введён, согласные/пороги не изменены.

Два новых независимых previous-context reads:
250860–252160 (1,3s) содержит только конец `رُۥن`, insufficient;
249660–252160 (2,5s) содержит весь expected `يُبڇصِرُۥن`, min0,653.
Existing wasl forecast начинался250727,7 и обрезал начало этого слова.
`previous-tail-2-18.json/.log` различает fresh context от retained opening replay.

Реально услышанный full previous tail + exact current fragment стали причиной
одного prospective расширения right opening253700–257900 (4,2s). Оно
восстановило весь **connected** first word10phones, min0,634. Дополнительный
last Nun в конце окна имеет q0,192 <0,25, поэтому прежний strong-continuation
guard его не считает сильным противоречием. Guard не ослаблен.
`extended-opening-2-18.json/.log` сохраняет новый raw decode и actual dispatch.

Причина включает decoder context effect и алгоритмический отказ расширить
неполный повторяющийся prefix при наличии действительно различающего контекста.
По этим данным нельзя приписывать ошибку чтецу или принимать waqf variant
как замену недостающим phonemes.

## Изолированная правка

Sparse-13:

- Разрешает finite expansion для native-pause candidate, если сильный exact
  fragment ещё не является полной seam phrase, partial tail+fragment не имеет
  textual rivals и full tail фактически прочитан у левого края паузы.
- Fragment остаётся **только expansion cue**. Полный connected opening,
  прежние unit-quality и strong continuation обязательны для acceptance.
- Разрешает существующее contextual shortening также для native onset;
  validator отдельно сверяет previous context read ID/end, whole tail,
  все quality values, actual PCM origin и исходную DSP-причину.
- Нет forced posterior, missing-unit recovery, удаления consonants или
  автоматического полного прохода. Каждое окно/попытка остаётся в ledger.

`prepare_context_expansion.py`, frozen source ZIP/manifest, snapshot path
воспроизводят изменение отдельно от грязного общего worktree.

## Prospective результаты

Полная2:14→36proofs,22added/0lost,231reads против81,
max5,5s, coverage4,461%, input7,314%. Поиск81,81→150,74s;
повышение полноты потребовало дополнительной работы, ускорение не заявляется.
2:18 принят только complete connected10 + full previous10, confidence0,634.
Следующий19 принят существующим минимальным `يرجعون + أو` context proof,
без чтения испорченного длинного standalone opening. Затем процесс дошёл до42.

Свежие остальные11cases и позднее объединение в `context-corpus.json`:
12cases/607rows,233→255proofs,22added/0lost. Все новые proofs — в полной2;
на остальных11нет потерь или новых positive→over500 переходов.
Сохранены raw per-case результаты, reference/audio hashes, порядок и actual
dispatch проверки. Calls739→891, поиск359,38→427,25s,
inference ledger309,27→371,23s. Coverage≤57,279%, input≤97,073%, read≤5,5s.

Четыре новых reference outliers>500:2:18+1476ms,21+639,23+1335,25+980.
Прежние семь corpus outliers сохраняются (включая32:11/16/20 и79:26).
Времена некоторых прежних меток сдвинулись на2–3ms из-за другого локального
crop, поэтому сохранность proof не выдаётся за побайтовую сохранность времени.
Модельные фонемы не являются автоматическим доказательством правильного
времени по эталону. Activation остаётся закрытой до temporal проверки.
`context-corpus-comparison.json` и singleton comparison содержат всё без
исключения неудобных rows. Merged corpus не считается новым inference run.

За этап:12завершённых pipeline cases/607rows/891calls,2fresh previous-context
calls и1fresh extended-opening call:894actual neural calls. Static operations
и retained opening replay не включены в это число.

## Следующая конкретная ошибка состояния —2:42

Декодер правильно распознал предыдущий текст: `ولا تشتروا` внутри2:41,
а нужный target42 — `ولا تلبسوا`. В окне809500–812200 весь предыдущий
двухсловный prefix совпадает с native mapping, minq0,479. Первое1,5s окно
ещё не содержит полного rival; после2,7s он уже прочитан целиком.

Алгоритм продолжает расширять до4,2s, поскольку первое общее `ولا` считается
`exact_first_word`; затем `had_prefix` блокирует дальнейшие candidates.
Общее первое слово не установило owner42. Это ошибка candidate/state handling,
а не ошибка распознавания самого `ولا تشتروا`.
`2-42-stop-evidence.json` и `2-42-previous-body-analysis.json` фиксируют
переходы, canonical indices611/612, все scores и отсутствие нужного opening.

Следующий prospective шаг — узко отклонять candidate, когда **полные два
слова** сильного decoded rival принадлежат внутреннему тексту предыдущего
подтверждённого аята, первое слово общее, а второе противоречит target.
Неполный или слабый rival не должен снимать blocking guard; подлинный
distinct current prefix также защищён. Это пока диагноз/план, код не менялся
на основании одного предположения. Нужны prospective2 и paired corpus перед
переносом такого отказа в production.

Docs updated: этот каталог, docs/STATUS.md. Scoped Ruff clean, pytest не
запускался. Canonical text/пользовательские БД/веса/default модели не записаны;
commit/push отсутствуют. Общая цель активна: temporal outliers, candidate
deflection, практическое production улучшение и скорость ещё требуют работы.
