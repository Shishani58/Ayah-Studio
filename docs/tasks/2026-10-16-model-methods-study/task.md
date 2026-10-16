# Сравнение способов применения одинаковых моделей

## Цель
По прямому поручению пользователя выполнить исследование на реальном аудио,
прежде всего Quran Phonetics. FULL, локальный ASR и timing оценивать отдельно.
Нужен измеренный выбор подхода, а не только список возможностей или исправление FAST.

## Документы и источники
AGENTS/CLAUDE/RTK; project.md (FULL/CTC/runtime/инварианты), STATUS;
model-capability-audit/report, adapter-diagnosis, asr-model-selection/report;
bench_full_windows/bench_models и действующие align/emissions/full_windows/
full_boundary/align_tiled/model_worker/model_runtime/reference_prior.
Первичные источники QPC, Transformers, Muaalem, NeMo из предыдущего аудита.

## Зафиксированные решения до измерений
- Frozen исходники, архив/SHA; общую грязную копию не изменять.
- QPC FULL: один checkpoint INT8, четыре target/topology способа:
  current; no-word-delimiter; phones-only (пробел/дефис/разделитель исключены
  из представления, canonical words/spans сохранены); unknown intro/outro.
  Дополнительная заранее заданная абляция blank-logit−1 (остальное current)
  назначена после4 development, до просмотра control результатов. Это проверка
  inference objective, не заявленная замена обучения CTC label priors.
- Одинаковые вычисленные acoustic outputs разрешено переиспользовать по точному
  SHA нормализованного входа/shape/weights. Это контролируемая абляция alignment,
  не свежий inference каждого варианта и не latency benchmark.
- FULL context: strong-pause20s текущий, fixed30s/context2, strong-pause6s.
  Дополнительные методы контекста на заранее заданном поднаборе разных чтецов.
- PyTorch/FP32/INT8 parity — одинаковые реальные локальные PCM, без target.
- Muaalem: один phoneme checkpoint, madd_aared2/4/6 при одинаковых emissions.
- Tarteel Base: greedy/beam5; FastConformer: CTC/RNN-T на одинаковых полных
  коротких записях. Recognition и timestamps не приравнивать.
- Во всех prediction процессах запретить открытие reference БД. Найденный
  bundled reference_prior для tiled отключить одинаково во всех вариантах;
  использовать текстовый prior. Это отличие от штатной конфигурации явно учитывать.
- Дополнительную Muaalem-проверку из QPC FULL отключить одинаково: сравниваем
  одну модель; прочие QPC/DSP этапы оставить. Production не менять.
- Reference читать отдельным scorer только после predictions; это benchmark
  cut agreement, не независимая физическая разметка слов.
- Корпус44 записей: Шатри1/2/32/55/59/65/67/73/77/79/80/98/112/114,
  все9 доступных записей Гамиди/Хусари/Миншави и3 Басита mujawwad.
  Development: Шатри1/80/112/98; остальные — фиксированный study control,
  но часть использована исторически, поэтому не объявлять глобально unseen.
- Не менять production/defaults/weights/пользовательские БД/канонический текст.
  Полные acoustic passes — только явно обозначенные FULL experiments; FAST
  не получает результаты полных проходов и не выполняется с fallback.

## Чек-лист
- [x] Snapshot, корпус, protected hashes, исполнимый протокол.
- [x] QPC FULL targets/topology и независимый scoring всех исходов.
- [x] QPC context и экспорт/квантование.
- [x] Дополнительные модели на том же аудио, контроли.
- [x] Причины, gains/losses, стоимость, воспроизводимый отчёт и итоговый выбор.
- [x] Проверки артефактов/сохранности, документация.

## Ограничения
Слуховых word/phoneme GT нет. Не имитировать прослушивание. Timing выводы
ограничены packaged reference; отсутствие human labels не мешает сравнению
покрытия/распознавания и выявлению регрессий, но ограничивает физическую приёмку.

Docs updated: task.md. Итоговый список заполнить после завершения.

## Изменение требований в ходе работы

Пользователь потребовал исключить любое влияние временных эталонов из
приложения. Отдельная задача remove-reference-influence удалила prior,
CTC−200ms, калибровку, reader/eval из package и GUI/catalog-derived данные.
Frozen prediction код не переписывается: reference_prior в нём всегда возвращает
пустое значение, lag=0; SQL guard разрешает только canonical text DB. Старые
lag200 outputs перемещены в historical-lag200-strong20 и не оцениваются как
новый результат. Исследовательские методы не переносятся в production.

QPC и вторичные модели запускаются независимыми процессами параллельно ради
измерения качества. Их wall times не являются честным latency сравнением.
FP32 и отрицательные контроли поставлены в очередь после основного QPC.
Muaalem первый harness использовал сырые Quran glyphs вместо model renderings;
полученный KeyError исправлен через inference_words(kind=muaalem). Ошибки
сохранены отдельно, повтор использует те же cached acoustic matrices. Это
ошибка harness, а не отказ модели.

## Итог

Завершены352 QPC и180 secondary положительных результатов на реальном аудио,
контроли нулей/чужой суры/сдвига, исходный PyTorch/FP32/INT8 parity и отдельный
последовательный encoder benchmark. Обе серии12 записей различаются и указаны
в protocol.md. Все варианты current/no_delimiter/phones/free_edges/blank−1
проверены на44 полных записях, включая2. Версии источников/хэши/raw outputs
сохранены. 95 защищённых файлов и86 snapshot source прежние;1447 cached
matrices проверены. Основные выводы, ограничения и команды — report.md.

Отрицательные4с CTC tests не вмещали target; дополнены нулями длины настоящей
112 для всех моделей. В Whisper дополнительно проверены overlap контекста,
SDPA и настоящий nbest по уже существовавшему правильному low-level probe.
Текущие положительные серии без отказов. Первоначальные ошибки harness и
исключённый lag200 архив сохранены отдельно. Остановленный для ресурсов Muaalem
возобновлён и завершён; фоновых model jobs после исследования не осталось.

Исследовательские способы не переносились в runtime. Отдельное прямое поручение
удалить влияние эталонов выполнено в задаче remove-reference-influence.
Текущий общийpytest983/10skip; публикации исходной/native сборки не было; локальные
sdist/wheel проверены. Нет незавершённых обязательных прогонов.
Ограничение: независимой human onset/safe-cut GT нет; point agreement и фонетический
word_tr PER не объявляются физической приёмкой.

Docs updated: report.md, protocol.md, task.md, docs/STATUS.md; отдельная задача
обновила AGENTS/CLAUDE и docs/project.md по удалению reference influence.
