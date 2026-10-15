# Модель audio → text для Корана

Дата исследования: 2026-10-14.

**Вывод:** первый доступный кандидат — `mohammed/fastconformer-quran-ar`
с независимым RNN-T decode. Он действительно распознаёт аудио без передачи
эталона и возвращает времена распознанных слов. В нашем небольшом probe он
лучше Tarteel Base, но НЕ прошёл проверку стабильности всех трёх режимов.
Готовую универсальную замену default по имеющимся данным выбрать нельзя.

## Что сейчас делает приложение

- Default: `TBOGamer22/wav2vec2-quran-phonetics`, ONNX INT8.
- FULL CTC раскладывает известные токены по аудио. FULL Whisper использует
  teacher forcing канонического текста и attention-DTW. ASR там помогает
  выбрать диапазон слов; итоговые времена получаются иначе.
- Штатный FAST у Tarteel Tiny/Base уже распознаёт локальную фразу независимо,
  но `timing=False`: локальная точность времён не доказана.
- Runtime FastConformer принудительно выбирает CTC-голову для known-text
  alignment. Наличие модели в каталоге не означает независимый ASR-пайплайн.

Код: `core/align.py`, `core/model_worker.py` (`Acoustic.__init__`,
`whisper_local`), `core/model_runtime.py`, `core/fast_adapters.py`,
`core/model_store.py`. Здесь и далее пути относительно `surah_cutter`.

Любое ASR декодирует акустические признаки в токены. Требуемое отличие:
**сначала независимая гипотеза из аудио, затем сопоставление распознанных слов
с Кораном для определения surah/ayah/word**. Передавать ожидаемые буквы модели
для принудительного получения нужного текста при таком распознавании не нужно.
Канонический текст остаётся владельцем индексов и не заменяется гипотезой.

## Найденные модели

| Вариант | Основание | Решение |
|---|---|---|
| mohammed/fastconformer-quran-ar | Quran ASR, hybrid RNN-T/CTC, 114.6M параметров, CC-BY-4.0; веса уже локально | Первый доступный кандидат, проверен независимо ниже |
| Muno459/fastconformer-quran | Более новый Quran ASR, ONNX, обучение также на телефонных записях; таблица авторского held-out benchmark: WER 5.41% | Перспективный отдельный кандидат; gated access и NPL-1.1, локально не испытан |
| tarteel-ai/whisper-base-ar-quran | Независимый ASR, опубликованный WER 5.7544 на недостаточно описанном eval set | Уже доступен; не решает доказанную проблему времён слов |
| naazimsnh02/whisper-large-v3-turbo-ar-quran | Quran fine-tune, авторский WER 1.18% на EveryAyah test | Второй открытый вариант для последующего парного испытания; короткие crop и timing не доказаны |
| wasimlhr/whisper-quran-v1 | Quran Large-v3, авторские данные по 19 чтецам; eval всего 115 samples, checkpoint WER 5.74% | Ни гарантий длинных записей, ни проверки word timing; CC-BY-NC-4.0 |
| Qwen3-ASR-1.7B / 0.6B | Независимый ASR с Arabic | Не основной выбор: у Qwen3-ForcedAligner Arabic не входит в заявленные 11 языков |
| Muaalem v3.2 | Вывод phonemes и tajweed attributes | Не соответствует запросу на обычный текстовый ASR |

Источники: [mohammed model card](https://huggingface.co/mohammed/fastconformer-quran-ar),
[Muno459 model card](https://huggingface.co/Muno459/fastconformer-quran),
[Tarteel card](https://huggingface.co/tarteel-ai/whisper-base-ar-quran),
[Quran Turbo card](https://huggingface.co/naazimsnh02/whisper-large-v3-turbo-ar-quran),
[Whisper Quran v1](https://huggingface.co/wasimlhr/whisper-quran-v1),
[Qwen official card, language table](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
[Muaalem](https://github.com/obadx/quran-muaalem).

Опубликованный WER mohammed 0.14% — валидация автора, а не обещание для любых
чтецов, коротких FAST-фраз и тайм-кодов. Проценты разных eval set не сравнивать.

В [Quran-Lab held-out board](https://quran-lab-quranic-asr-leaderboard.hf.space/)
на 600 клипах в таблице: Muno offline 5.41%, mohammed 6.95%, Tarteel Base
7.97%, обычный Large-v3 10.10%, Qwen3-ASR-1.7B 33.71% WER.
Это данные авторов benchmark, не наш прогон и не оценка времён.
Верхний headline страницы для Muno говорит 4.13%, а таблица и model card —
5.41%; используем согласованную таблицу, не выбираем более красивое число.
Zipformer лидирует через phoneme retrieval с закрытым набором аятов, поэтому
не считается победителем свободного текстового ASR для этого запроса.
Набор gated; его файлы не скачивали. Независимость оценки от авторов моделей
не доказана. FAST приложения и streaming в таблице — разные вещи:
FAST может запускать offline ASR на коротком локальном crop.

## Новые локальные проверки

`probe_asr.py` выполнил по 14 случаев для каждого из трёх вариантов:
mohammed RNN-T, тот же checkpoint с независимым CTC greedy и Tarteel Base.
Эталон никогда не передавался в recognition call.

- Шатри: полная Фатиха, 59.01 с, и полная 112.
- Полная 112: Гамиди, Миншави, Хусари, Абдул-Басит mujawwad.
- Для каждой из шести записей отдельный local crop: Фатиха 6300–10300 мс;
  112 — первые 6000 мс. Это **LOCAL-ASR probes**, не запуск FAST executor.
- Повтор crop Фатихи с 600 мс добавленной тишины; отдельно 4 с чистой тишины.
- Полные эксперименты отдельно от local. Для Tarteel FULL-ASR — независимые
  30-секундные блоки без known-text alignment и без контекстной склейки.
  Ошибки стыка входят в результат этого простого probe; это не оптимизированный
  production long-form decoder и не доказательство потолка модели.

После decode `evaluate_asr.py` считает edit distance слов: огласовки сняты,
формы alef и alef-maqsura нормализованы. Для 112 отдельно исключается ровно
распознанный начальный префикс басмалы, который не является аятом этой суры;
raw errors тоже сохранены. Буквы/слова эталона не исправляются. 104 эталонных
слова на шести FULL-записях; overlap с обучением не проверен.

| Независимый decoder | Ошибки слов FULL | WER этого probe | Чистая тишина |
|---|---:|---:|---|
| FastConformer RNN-T | 5 / 104 | 4.81% | Пустой текст |
| FastConformer CTC greedy | 11 / 104 | 10.58% | Ложная `ت` |
| Tarteel Base greedy | 13 / 104 | 12.50% | Ложное `وَالْمُؤْمِنِينَ` |

RNN-T полностью распознал Фатиху, но дважды пропустил `الله` в 112 Гамиди
и Миншави; у Шатри добавил хвост `الرس`. На local Фатихе текст правильный,
на первых 6 с Хусари RNN-T вернул пусто. CTC на Фатихе FULL текст правильный,
но local даёт `الحمُْ ... الْعَِينَ`; добавление тишины меняет его на правильный
текст. Поэтому даже одна и та же модель не доказала устойчивость FULL/local.

Штатное NeMo окружение воспроизводимо падало при импорте ONNX:
`ml_dtypes.float4_e2m1fn` отсутствует. Только для probe установлен
`ml_dtypes==0.6.0` в `/tmp/surah-cutter-asr-ml-dtypes-20261014` и передан
через PYTHONPATH. Установленное окружение приложения не изменено.

## Тайм-коды: отдельный результат

В NeMo включены native greedy timestamps с `preserve_alignments=True`,
`compute_timestamps=True`; times взяты из token offsets с шагом encoder 80 мс.
Это времена независимого decode, без forced alignment заданных слов.
[Документация NeMo](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/asr/inference.rst).

Фатиха RNN-T: 7 начал, median absolute error 144 мс, max 519 мс, 1 ошибка
>500 мс. Аль-хамду: 6800 против GT 6508 мс, +292 мс. В прежнем QPC FULL
из сохранённого аудита: 5339, −1169 мс. Это сравнение **одной записи**;
предыдущее QPC измерение заново здесь не запускалось.

У CTC text FULL Фатихи правильный, но native word timestamps дают median
1840 мс, max 2976 мс и 6/7 ошибок >500 мс. Это отказ конкретного стандартного
timestamp estimator; не доказательство невозможности получить хорошие времена
из CTC frames отдельным алгоритмом.

RNN-T shift check +600 мс на правильных четырёх словах:
start residual −40/+120/+40/−40 мс; end residual −280/−360/−360/−680 мс.
Полная калибровка не пройдена. Начало слова по decode может запаздывать,
а последняя эмиссия токена не гарантирует физический конец произнесения.
Нельзя считать `timestamp['word']` точной разметкой автоматически.
Ручных word-level GT и слухового подтверждения нет.

## Решение по режимам

1. **FAST:** first-choice experimental adapter — независимый RNN-T на
   существующих минимальных локальных crop. Передать только PCM; затем
   сравнить распознанные слова с каноническими кандидатами. Сохранить текущие
   ограничения 6 с / 65% / 200%, actual-read ledger, contextual evidence для
   повторов, cancellation и review при отказе. Сейчас это предложение,
   adapter в production не добавлен; 1.3/2.5-секундные окна не проверены.
2. **FULL:** полноценный ASR на разрешённых полных участках, затем привязка
   гипотезы к canonical word owners. Пропуски, повторы, wasl и басмала должны
   отражаться в результате/ревью, а не незаметно подставляться из эталона.
   Muno ONNX имеет смысл испытывать отдельно после получения доступа.
3. **Words:** те же независимо распознанные слова, но отдельная проверка
   точности start/end. RNN-T native times пока experimental; не переносить
   известный lag QPC и не включать как проверенный production timing.

Для допуска нового default нужен парный корпус по режимам: все 114 сур
нескольких чтецов, отдельно mujawwad, 2/42 и начальные буквы, 55/77 и повторы,
wasl, длинные паузы/эхо, тихое начало, обрезанная фраза и тишина. Оценивать
WER/пропуски/галлюцинации, recall подтверждённых границ, ошибки >500 мс без
review, времена слов по ручному GT, скорость/RAM на целевом CPU, и ledger FAST.
Сравнение текстов и сравнение времён — отдельные критерии. Не требовать совпадения
с непроверенным соглашением GT в длинной паузе.

**Блокеры:** широкая проверка и word GT отсутствуют; у нового Muno нужны
доступ и проверка условий распространения; NeMo production runtime требует
исправления зависимостей. Qwen word alignment Arabic не заявлен.
Ни один из проверенных вариантов не доказал «стабильно во всех ситуациях».

## Воспроизводимость и файлы

В `provenance.json`: исходные аудио SHA, crop intervals, PCM SHA, pinned model
revision и SHA локальных checkpoint/config/tokenizer. В `*-asr.json`:
independent text, word times (NeMo), runtime versions, load/inference times.
В `evaluation.json`: post-decode errors, basmala policy, ayah onsets и shift.
Промежуточный Base прогон без attention mask заменён финальным с явной mask;
его гипотезы совпали. Для выводов использован только финальный JSON.

Команды из корня репозитория:

```sh
rtk proxy .venv/bin/python docs/tasks/2026-10-14-asr-model-selection/probe_asr.py prepare
rtk proxy uv pip install --python /Users/ayub/.surah-cutter/nemo-runtime/bin/python --target /tmp/surah-cutter-asr-ml-dtypes-20261014 'ml_dtypes==0.6.0' --no-deps
rtk proxy env PYTHONPATH=/tmp/surah-cutter-asr-ml-dtypes-20261014 HF_HUB_OFFLINE=1 /Users/ayub/.surah-cutter/nemo-runtime/bin/python docs/tasks/2026-10-14-asr-model-selection/probe_asr.py fastconformer
rtk proxy env PYTHONPATH=/tmp/surah-cutter-asr-ml-dtypes-20261014 HF_HUB_OFFLINE=1 /Users/ayub/.surah-cutter/nemo-runtime/bin/python docs/tasks/2026-10-14-asr-model-selection/probe_asr.py fastconformer-ctc
rtk proxy env HF_HUB_OFFLINE=1 /Users/ayub/.surah-cutter/model-runtime/bin/python docs/tasks/2026-10-14-asr-model-selection/probe_asr.py whisper-tarteel-base
rtk proxy .venv/bin/python docs/tasks/2026-10-14-asr-model-selection/evaluate_asr.py
rtk proxy .venv/bin/ruff check docs/tasks/2026-10-14-asr-model-selection
```

42 финальных cases; дополнительные предварительные/повторные запуски не
считаются независимыми samples. Ruff пройден. Production suite не запускался:
production-код не менялся. Default, пользовательские БД, веса и настройки
не изменены. Незавершённые чужие FAST/FULL изменения не тронуты.

Docs updated: task/report и запись в `docs/STATUS.md`.
