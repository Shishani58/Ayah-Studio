# Независимая перепроверка исправлений FAST

14 октября 2026. Исполнитель — новый агент `fast_fixes`; родитель проверяет
исходники и результат самостоятельно. Production FAST остаётся legacy.
Родитель не правил реализацию параллельно исполнителю. До работ сохранён
snapshot 15 файлов; изменения FULL в общей рабочей копии не относятся к задаче.

## Результат проверки кода

1. Recovery больше не требует успешного greedy первого целого слова для любого
   расширения. Независимый beam и exact prefix posterior сравниваются с
   конкурентами, blank и остаточной вероятностью. Частичный/шумный decode
   разрешает только конечное локальное расширение. Посторонняя речь не
   подтверждается. Sparse anchors сохраняют пропуски с confidence 0; изменение
   политики/version описано prospectively, старые FAIL сохранены.
2. Повторы получают составное доказательство previous-tail + opening. Для
   паузы это два разных чтения у её краёв. Порядок/темп сами по себе не
   доказывают владельца; обрезанный, слабый, устаревший или неправильный
   предыдущий контекст отклонён независимыми пробами.
3. Spoken names и компактные ASR aliases начальных букв сохраняют канонический
   word ownership. Различающая часть не превращается в правку текста Корана.
4. Проверяется качество selected BPE tokens именно доказательной фразы.
   Проверены UTF-8 character split, token spanning two words, weak unrelated
   tail, non-reversible decode и NaN. Неоднозначное отображение имеет quality 0.
5. Реализован локальный tail+opening recovery после подтверждённого якоря.
   ASR first-onset/wasl timing остаётся неподдержанным: это частичное закрытие
   исходной находки. Реальные bounded attention-пробы показывают зависимость
   результата от query convention и не дают калиброванный onset. Нет
   независимой разметки начала слова; DSP activity и reference pause center
   нельзя выдавать за неё.
6. Полный DSP pause interval отделён от короткого speech crop. Длинный
   затухающий хвост не получает уверенную midpoint boundary только потому,
   что текст справа распознан; стабильная пауза отдельно проверена.
7. Run identity описывает фактически загруженные веса/config/vocab/providers,
   worker/runtime и выполняемый Python code с defaults/kwdefaults. Проверена
   неизменность identity после изменения файла уже загруженной модели.
   Fingerprint консервативен: смена checkout/Python/instrumentation может
   инвалидировать reuse. Это не гарантия одинаковых floating-point результатов
   на произвольном оборудовании.

Во время независимого ревью обнаружены и исправлены дополнительные дефекты:
невозможные beam paths, неполный cache accounting, слабая проверка геометрии
контекстного чтения, NaN/подмена prefix proof, одинаковый run key при разной
реализации, отсутствие defaults в fingerprint, ошибочная блокировка current
после provisional future hint и завышенный confidence при более слабом
обязательном предыдущем контексте. Их отрицательные воспроизведения сохранены в
`independent-*.py/.json`.

## Выполненная независимая проверка

- Финальный `pytest -q`: 784 passed, 7 skipped, 11.49 s. Пропущены opt-in реальные audio tests;
  FULL здесь отдельно не запускался.
- Scoped Ruff и `git diff --check` чистые; AGENTS.md/CLAUDE.md идентичны.
- 104 независимых адресных случая: 32 beam + 35 marginal prefix + 5 attribution
  + 6 context + 5 prefix-proof + 9 loaded identity + 5 sparse + 7 speech-edge.
- Восстановлены PCM и SHA всех 49 вызовов temporal pre-roll diagnostic; проверены
  длительности, input/coverage budgets и обе query-row conventions.
- 49 защищённых файлов повторно совпали по SHA после всех прогонов.
- Pipeline, project writer, catalog и alignment_identity совпадают с baseline;
  production switch и формат БД не менялись.

## Аудио и оставшиеся ограничения

Frozen **sparse-4**: 138 реальных model×surah случаев, 46 сур, 1 475 аятов
на модель. До последней scalar confidence правки родитель независимо сверил
все source SHA, audio/reference/DSP hashes, фактические PCM inputs и ledger,
6s/65%/200% budgets, cache cap, уникальность ключей, число/порядок/диапазон аятов
и пересчитал ошибки. Данные: `independent-artifact-check.json`.

| Модель | Подтверждено / 1 475 | Подтверждённых ошибок >500ms | Общие37: legacy → sparse-4 |
|---|---:|---:|---:|
| CTC | 33 | 0 | 276 → 28 |
| Tiny | 305 | 0 | 395 → 230 |
| Base | 404 | 1: 79:26, +508ms | 411 → 302 |

M5 **FAIL**: coverage ниже legacy; Base79:26 дополнительно не проходит error
gate. Его короткий DSP interval 115320–116180ms имеет затухающую энергию
(RMS первых/последних четвертей ≈.016/.0043); имеющийся gate длинных пауз
этот короткий interval не отклоняет. Пороги не подгонялись под reference,
расхождение не исключено из метрик. Median/p95 всех результатов большие,
потому что большая часть аятов честно остаётся conf0/review.

**Текущая sparse-5** отличается агрегацией confidence essential context и
проверкой её верхней границы. Исторический138 sparse-4 не rekeyed и не выдан за
новый acoustic прогон. Родитель самостоятельно выполнил четыре свежих случая
20/109 × Tiny/Base, затем независимо восстановил все **126 actual PCM passes**,
проверил текущие source SHA и пересчитал reference errors. Tiny:22/141,
Base:36/141 подтверждений; среди них ошибок >500ms нет. В109 обе модели
подтверждают6/6, включая3/5 с реальным previous-tail proof. Confidence каждой
контекстной границы равен min(opening, essential tail). Границы и решения о
подтверждении во всех четырёх случаях совпали со sparse-4. Дополнительно родитель
независимо проверил все 138 scalar replay случаев: только 19 confidence scores
уменьшены; все прочие значения совпадают с raw corpus.
Обратное применение source-4-to-5.diff точно восстановило frozen SHA исполнителя.
Это подтверждает ограниченный scope перехода, не является новой inference. Данные:
`independent-fresh*.json`, `independent-verification.json`.

Дополнительно восстановлены 15 derived inputs (crop, leading silence, repeat,
truncation, very-short), их PCM SHA и фактические reads. Для ведущей 12s тишины
и вставки повторного сегмента известен перенос исходных reference boundaries;
все положительные границы после переноса имеют ошибку ≤50ms. Crop/truncation
не выдаются за независимо размеченный word-onset корпус.

Нет независимого wasl/repeat word-onset ground truth, второй чтец и native Qt
не проверялись. Измеренное качество CTC и доля unresolved результатов всё ещё
ограничивают полезность FAST. M5 требует отдельного доказанного прохождения;
механические исправления и green tests не разрешают production activation.

Docs updated: этот отчёт, independent-checklist.md, independent probes/results.
Документацию реализации/STATUS обновляет исполнитель.
