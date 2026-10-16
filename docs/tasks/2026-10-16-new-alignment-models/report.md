# Новые модели: подключение и реальные проверки

2026-10-16. Задача выполнена частично: MFA подключена и реально испытана;
Muno FastConformer и Zipformer зарегистрированы, но их веса закрыты условиями
Hugging Face. Принятие условий/передачу контактов за пользователя не выполняли.

## Что доступно

- `mfa-quran-hafs`: FULL, words. Веса+словарь
  скачаны в управляемый каталог с проверкой upstream SHA/Git blob. Подготовлено
  отдельное native окружение MFA 3.4.0; Kaldi не импортируется приложением.
- `fastconformer-muno`: зафиксирован NeMo checkpoint 459 МБ и лицензия.
  Это другая модель, чем существующий mohammed FastConformer. Выбран NeMo для
  повторного использования существующего адаптера; компактный ONNX пока не
  подключён. Готовность/выбор выключены до доступа и настоящего inference.
- `zipformer-quran`: зафиксированы INT8 ONNX, tokens, phonetic references и
  лицензия. Адаптер не реализован без изучения фактических файлов/выхода;
  готовность/выбор выключены. Наличие строки не означает готовую интеграцию.
- Downloader читает локальный `HF_TOKEN`/`HF_TOKEN_PATH`/стандартный HF token
  после входа. Токен не попадает в ответ, receipt или настройки приложения;
  при redirect на другой host Authorization удаляется существующим кодом.
- FAST для всех трёх отклоняется до decode/inference, без FULL fallback.
  MFA tiled также отклоняется: bounded tiles для native MFA не реализованы.
  Default/settings, текст и пользовательские БД не менялись.

## MFA: важные детали

Используется native `mfa align_one`, beam=40, retry_beam=160, boost_silence=1.
Silence probability берётся из зафиксированного acoustic archive (0.19),
а не из описания карточки. Каждый запрос имеет собственные temporary files и
MFA_ROOT_DIR; subprocess ограничен временем, отмена убивает и дожидается его.
Probe проверяет версию и реально вызывает acoustic decoder с загруженными
весами на синтетическом входе — это проверка совместимости, не качества.

Исходный словарь полностью покрывает 77 432 канонических слова через наш
проверенный Uthmani cache (у 37:130 один canonical Word имеет две единицы).
MFA FileData автоматически делает NFKC, а Kalpy dictionary loader — нет:
первая попытка породила `<unk>` и была отклонена. Исправлена только локальная
lookup-копия ключей словаря и входа в том же NFKC; фонемы/канонический текст
не меняются. Диагностика отказа — mfa-nfkc-diagnostic.json.
Выход принимается только при точном совпадении всех labels/числа/порядка;
`<unk>`, пропуски, невалидные/пересекающиеся интервалы отклоняются.
`<eps>` — штатный silence label, не слово. Несколько единиц сохраняют один
canonical owner и время первой единицы.

MFA likelihood не является вероятностью корректной границы: confidence=0,
все метки остаются для ревью. Никакой фиктивной уверенности не добавлено.

## Парное сравнение

Аудио и SQLite reference только читаются. SHA, полные строки, confidence,
времена стадий и trace: shatri-comparison.json. Ничего не записывается в
пользовательские проекты. Ошибки относительно начала аята из существующей БД,
это не ручной физический эталон начала каждого слова.

| Сура Шатри | Основная: median/max, мс | MFA: median/max, мс |
|---|---:|---:|
| 1 | 37 / 110 | 150 / 209 |
| 112 | 29 / 41 | 1414 / 2810 |
| 114 | 40.5 / 102 | 195.5 / 879 |

В 17 началах MFA имеет 4 ошибки >500 мс; все 17 помечены для ревью.
Пословный расчёт первого аята прошёл для обеих моделей на всех трёх сурах;
число/порядок/диапазон проверены, точность каждого слова не оценена: GT нет.
Время скорости не обобщать: на компьютере параллельно шли другие задачи.

Дополнительно проверен рекомендованный корпусный `mfa align` на той же 112:
он также сильно смещает границы (mfa-corpus-output.json/probe.log), преимущество
не обнаружено. Лог содержит только first-pass alignment; отдельная fMLLR
адаптация в этом запуске не подтверждена. Не утверждаем, что она была выполнена.
Этот опыт не заменяет native align_one и не меняет production параметры.

Benchmark был начат до добавления в source строки `lookup=nfkc` и усиления
probe; сам alignment уже использовал NFKC. Финальный отдельный smoke 112
проверяет текущие source/probe и совпадение аятных результатов.

Вывод: MFA пригодна только как эксперимент для локальной проверки; оснований
заменять основную модель нет. Данных о новом Zipformer/Muno пока нет.

## Воспроизведение

Окружение проверено на macOS ARM64. Explicit conda artifact lock с контрольными
суммами — surah_cutter/core/runtime_locks/mfa-osx-arm64.txt, версии/хэши пакетов
также сохранены в mfa-runtime-manifest.json. Windows/Intel и собранная .app не
проверялись. Для установки нужен micromamba (или совместимый conda):

```sh
rtk proxy micromamba create -y -p ~/.surah-cutter/mfa-runtime --file surah_cutter/core/runtime_locks/mfa-osx-arm64.txt
rtk proxy .venv/bin/python -m tools.bench_models --models quran-phonetics mfa-quran-hafs --surahs 1 112 114 --audio-dir /Users/ayub/Desktop/Abu_Bakr_Ash-Shatri --reference data/references/abu_bakr_ash_shatri.db --output /tmp/new-model-comparison.json
```

Альтернативный путь executable задаётся `SURAH_CUTTER_MFA`; требуется MFA3.4.0.
Установка native окружения вручную необходима на другом компьютере: кнопка
скачивания получает веса, но не устанавливает micromamba/системные пакеты.

## Оставшиеся блокеры

1. Принять условия по двум страницам моделей и выполнить локальный HF login.
2. После доступа — изучить и подключить Zipformer, проверить Muno runtime и
   устранить известную несовместимость ml-dtypes/ONNX в NeMo перед допуском.
3. Выполнить одинаковое реальное сравнение всех кандидатов; word-level GT
   по-прежнему отсутствует. До этого у закрытых кандидатов нет активного выбора.

Источники: [MFA](https://huggingface.co/Quran-Lab/mfa-quran-hafs),
[Zipformer](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3),
[Muno](https://huggingface.co/Muno459/fastconformer-quran),
[native MFA command](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/v3.4.0/montreal_forced_aligner/command_line/align_one.py).


## Финальная проверка

152 целевых pytest passed; Ruff и git diff --check чисты, AGENTS/CLAUDE
идентичны. Native version+acoustic probe завершился успешно. Final smoke112
на окончательном source с NFKC дал те же аятные и пословные строки в точности.
FAST и MFA tiled отклоняются до decode; cancellation subprocess проверена.
Docs updated: docs/project.md, docs/STATUS.md, task/report и артефакты рядом.
