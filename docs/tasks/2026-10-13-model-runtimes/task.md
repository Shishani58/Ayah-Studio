# Реальные runtime дополнительных моделей

Цель: подключить собственные веса MMS_FA, Muaalem ONNX, Quran FastConformer,
Whisper Turbo Quran и шести обычных моделей; реально скачать, запустить,
проверить выбор, аятный и пословный расчёт. Канонический текст, схема данных
и пользовательские проекты не изменяются. Коммиты и push не выполнялись.

Прочитано: docs/project.md (пайплайн и инварианты), STATUS, задачи каталога,
статусов и иконок; align/emissions/model_store/word_align/server; карточки
моделей и исходники torchaudio, Transformers, NeMo и quran-transcript
(словарь, признаки, порядок головы и языковые адаптеры).

## Решения

- Тяжёлые зависимости работают в отдельных управляемых Python 3.11 окружениях
  model-runtime/nemo-runtime. Приложение не импортирует Torch/NeMo.
  NeMo требует NumPy 1.26.4, pyarrow 19.0.1 и scipy 1.13.1 на проверенном Mac;
  quran-transcript использует отдельное окружение с NumPy 2.
- Выбираются ровно запрошенные веса. Перед сохранением ID выполняется настоящий
  encoder/head probe. Ошибка оставляет прежнюю настройку; другой моделью
  результат не подменяется. API всех десяти моделей проверен с временным settings.json.
- MMS_FA: штатная CTC-голова, uroman, без star/word separator. MMS Arabic:
  strict-загрузка базовых весов с фактическим размером lm_head, затем локальный
  ara-адаптер. HF from_pretrained(target_lang=ara) подключал адаптер раньше
  базовой головы и давал mismatch 121/154; порядок исправлен.
- Muaalem: ONNX phonemes, SeamlessM4T features, Hafs/madd=4/waqf по аятам.
  Оба NeMo: явный encoder → ctc_decoder (hybrid forward возвращал features).
  Эквивалентные BPE-разбиения принимаются только при точном совпадении написания
  с фиксированной целью. Базовая модель читает Imlaey без огласовок, Quran — с ними.
- Whisper: teacher forcing известного текста, decoder cross-attention и DTW.
  Prefix поглощает начальную тишину; standalone-space не задаёт начало слова.
  Длинные записи используют окна 30 с, подтверждение текстовых опор и перекрытие
  одним словом. ASR только определяет диапазон канонических слов. Недостаточные
  опоры/отсутствие прогресса отклоняются; тайм-коды из ASR не принимаются.
- Дополнительные runtime доступны в FULL/tiled/words. FAST отклоняется до
  inference для всех десяти: никаких скрытых полных проходов или fallback.
- Скачивание: pinned revision, HTTPS, подтверждённый Content-Range, диапазоны
  8 МБ, полный размер и SHA перед атомарной публикацией. Сетевой сбой сохраняет
  префикс для повтора; отмена очищает partial. Проверенные веса сохраняются при
  ошибке установки окружения. Настоящие premature EOF/timeout CDN воспроизведены.
- В source сохраняются ID/revision, версия адаптера known-text-v2, алгоритм,
  параметры, SHA вспомогательного текста и фактические версии runtime.

## Проверка

Вспомогательный Uthmani/Imlaey cache покрывает 77 432 слова, 6236 аятов,
114 сур. Проверяются SHA Quran DB, исходный текст и индекс каждого слова;
37:130 сохраняет один QPC Word с двумя орфографическими единицами.
Native tokenizer MMS Arabic также проверен на всех 77 432 словах.

Реальные process_surah → write_timings на временных .scproj проверены для
MMS_FA, Muaalem, Quran FastConformer, Whisper Small и MMS Arabic. Строки
edited/verified в timings/review_meta сохранены без изменений; результат
содержит все шесть аятов 114-й суры. Пользовательские БД не записывались.

Шатри 114/112, эталон data/references/abu_bakr_ash_shatri.db. Медианная
абсолютная ошибка аятного начала (мс); максимумы, confidence, времена работы,
SHA аудио/эталона и полный source находятся в JSON рядом. Это малый набор,
не оценка всего Корана и не доказательство превосходства одной модели.

| Модель | 114 | 112 |
|---|---:|---:|
| MMS_FA | 182.5 | 188 |
| Muaalem | 124 | 139.5 |
| Quran FastConformer | 195.5 | 104.5 |
| FastConformer Arabic | 209 | 132.5 |
| Whisper Turbo Quran | 313.5 | 298 |
| Whisper Small | 344 | 291.5 |
| Whisper Medium | 499.5 | 423.5 |
| Whisper Large-v3 | 139.5 | 111.5 |
| Whisper Large-v3 Turbo | отказ: недостаточные опоры | 293.5 |
| MMS Arabic | 202.5 | 198 |

Актуальные короткие замеры: ctc-validation.json, nemo-validation.json,
whisper-validation.json и четыре *-validation.json обычных Large-v3,
Large-v3 Turbo, Medium, MMS Arabic. Первые отчёты v1 исторические: Whisper
до исправления prefix давал искусственный нулевой локальный старт; использовать
актуальные v2. Пословный confidence исправлен: CTC Segment.end включительный,
однофреймовый токен не пустой.

Длинная сура 78: MMS_FA/Muaalem/Quran FastConformer завершили 40 аятов,
медианы 242/227.5/285.5 мс, максимумы около 1.3 с. Whisper Turbo Quran v2
завершил 40 аятов, медиана 389 мс, максимум 1417 мс (410.5 с работы).
Whisper Small отклонил 78 из-за недостаточных опор. Обычный Turbo на 114
после сдвига окна выдал «اشتركوا في القناة» (подпишитесь на канал): проверка
остановила расчёт. Эти отказы сохранены в JSON, без выдуманных результатов.
Пословная уверенность MMS Arabic и Quran NeMo низкая для части слов;
нулевые оценки сохранены. Автоматические границы требуют слухового ревью.

Последняя проверка: pytest — 390 passed, 7 skipped; ruff чист;
git diff --check и cmp AGENTS.md CLAUDE.md успешны. Проверены macOS CPU
и API приложения из исходников. Собранный desktop binary и Windows не проверялись.

Повтор короткого benchmark (только чтение):

```sh
.venv/bin/python -m tools.bench_models --models mms-fa muaalem fastconformer fastconformer-arabic whisper-quran whisper-small whisper-medium whisper-large-v3 whisper-large-v3-turbo mms-arabic --surahs 114 112 --audio-dir /Users/ayub/Desktop/Abu_Bakr_Ash-Shatri --reference data/references/abu_bakr_ash_shatri.db --output /tmp/model-validation.json
```

- [x] Скачать и запустить десять моделей
- [x] Подключить аятный/пословный расчёт и выбор
- [x] Проверить канонические цели и защиту ручной разметки
- [x] Реальное аудио, качество и явные отказы
- [x] Обновить docs и STATUS

Блокеров реализации нет. Ограничения качества и непроверенные платформы — выше.
Docs updated: docs/project.md, docs/STATUS.md, этот task.md, JSON-отчёты рядом.
