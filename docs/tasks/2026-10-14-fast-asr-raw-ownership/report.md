# FAST: исправление исходного написания ASR перед расширением названий букв

Предыдущий этап дал новый конкретный дефект адаптера; сначала проверены
сохранённые данные, затем отдельные исходники и свежий парный корпус.
Рабочий fast_adapters.py теперь исправлен. Тесты не запускались.

## Причина и изменение

attributed_asr_units нормализовал хамзу до алифа. Tarteel.observe применял
whisper_alignment_text уже к «الم», превращая обычное «أَلَمْ» в «الف لام ميم».
Canonical represent использовал исходное «ألم» и оставлял обычное слово.
Это расхождение адаптера, а не ошибка модели в распознавании первого слова.
Модельное «ترين» вместо «تر» в диагностике остаётся отдельной ошибкой NN.

В attribution теперь сохраняется raw_word: исходные буквы и огласовки
выбранных UTF-8 BPE-токенов, с прежним сопоставлением каждого символа токенам.
Решение о названиях букв принимается по raw_word до потери хамзы.
Bare «الم» и каноническое «الٓمٓ» продолжают расширяться; «أَلَمْ» остаётся словом.
Для непроверяемой token-word attribution сохранён fallback с нулевым качеством,
без KeyError по отсутствующему raw_word. Порог и вероятности не менялись.
Версия Tarteel adapter —v4-raw-alias-ownership; прежние cache/run keys не
обозначают новое поведение. CTC, worker, FULL и канонический текст не правились.

## Анализ и свежая проверка

Scan всех retained Tiny/Base чтений нашёл два затронутых случая: Base59/read43
и Base77/read34. Ни один retained named-opening case не требует такого изменения.
Затем fresh четыре модели ×12сур ×before/after:96cases,4856rows.
QPC INT8/FP32 и Tiny: строки и actual dispatch sequences совпали точно;
число подтверждений2/2/84 осталось прежним. Base114→130 (+16), нет потерь,
изменений ранее подтверждённых строк или новых отклонений>500ms.

Новые подтверждения:59:11–17,19–24 (13);77:25,27,28 (3).
Их абсолютные ошибки:10–447ms. Existing Base79:26+508ms остался, не исправлен
этим изменением. Все фото/слуховые claims отсутствуют; эталон только оценивал
результаты после получения predictions, не участвовал в поиске.

Base actual calls273→326, search103,391→122,386s: больше полезной работы,
это не ускорение корпуса. Ранее production DSP RMS reuse сохраняется.
Все фактические интервалы проходят6s/65%/200%, sequence violations0.
Snapshots/code manifests проверены после terminal exit0 всех8 процессов.
Main bytes равнялись frozen before до переноса; production.diff/hash guard сохранены.

## Регрессии и артефакты

tests/test_asr_raw_ownership.py: пять подготовленных проверок. Разделённый
UTF-8 символ в двух BPE-токенах, общий токен двух слов, сохранение Hamza/
огласовок, compact/named ALM и неподтверждаемая attribution с качеством0.
Тесты НЕ запускались: пользователь разрешил их только при общем завершении.
Scoped Ruff adapter/scripts/test clean; AGENTS/CLAUDE побайтово совпадают.

scan.py/retained-alias-cases, frozen source-pair.zip/manifests, все96прогоны с
JSON/log/summary, compare.py/comparison, added-errors, production.diff/guard,
regression draft сохраняют весь путь анализа. Никаких commits/push/DBwrites.
Docs updated: task.md/report.md/docs/project.md/docs/STATUS.md.
