# Свежий baseline A

48/48 rows и actual dispatches точно совпали с историческим sparse12. Это новый прогон baseline-a2, не переименованный старый результат.

|Модель|Подтверждения/607|Calls|Load/probe,s|Decode,s|DSP,s|Executor,s|
|---|---:|---:|---:|---:|---:|---:|
|quran-phonetics|2|120|0.405|18.783|2.010|32.585|
|quran-phonetics-fp32|2|111|0.869|18.783|2.010|33.890|
|whisper-tarteel-tiny|105|260|5.540|18.783|2.010|61.373|
|whisper-tarteel-base|130|326|4.733|18.783|2.010|129.029|

Источники: baseline-a2-command.json/imports.json/process.json/log/json/summary.json.
Основной процесс и живые workers: одновременный tree RSS200ms; peak 2171.4MiB. Это не сумма исторических ru_maxrss.
Первичный неудачный baseline-a exit1 сохранён: модель не запускалась из-за отсутствия snapshot models.
Полный per-case CSV/JSON: baseline.csv/baseline.json; причинные traces: baseline-causes.json.
Первичные CTC отказы блокируют605оставшихся строк на модель; это потенциальный каскад, не605независимых акустических диагнозов.
Сравнение к reference не заменяет physical labels; Base79:26+508ms без review сохранён.
