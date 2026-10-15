# Актуальный baseline

Свежий frozen source sparse11. 4 модели ×12 полных сур =48 случаев, 607 аятов на модель. Все ошибки ниже относятся к соответствующему эталону; confirmed и estimates считаются отдельно. RSS — high-water процесса, children не включает живой worker; не сумма пиков. Total исключает загрузку; load отдельно в summary. DSP — внешняя подготовка + внутренняя фаза.

| Модель | Подтверждено | Ревью | Вызовы | Inference, с | Executor, с | Confirmed >500 мс |
|---|---:|---:|---:|---:|---:|---|
|quran-phonetics|2|605|120|8.711|14.673|[]|
|quran-phonetics-fp32|2|605|111|11.303|17.143|[]|
|whisper-tarteel-tiny|87|520|231|46.076|51.687|[]|
|whisper-tarteel-base|130|477|326|130.733|136.652|[(79, {'ayah': 26, 'error_ms': 508, 'confirmed': True, 'review': False})]|

## Первые блоки

| Модель | Сура | Первый block | Причина | Последнее local decode |
|---|---:|---:|---|---|
|quran-phonetics|1|2|incomplete_distinguishing_phrase|['m', 'i', 'l', 'h', 'a', 'd', 'u', 'l', 'i', 'l', 'ā', 'r', 'a', 'm', 'i', 'l', 'ā', 'l', 'a', 'm', 'ī']|
|quran-phonetics|32|1|incomplete_distinguishing_phrase|['a', 'l', 'i', 'h', "'", 'l', 'a']|
|quran-phonetics|55|1|incomplete_distinguishing_phrase|['w', 'a', 'l', 'n', 'a', 'j', 'm', 'w', 'a', 's', 'h', 'a', 'j', 'a', 'r', 'u']|
|quran-phonetics|59|1|incomplete_distinguishing_phrase|['w', 'a', 'q', 'a', 'd', 'a', 't', 'h', 'a', 'h', 'ī']|
|quran-phonetics|65|1|incomplete_distinguishing_phrase|['w', 'a', 't', 'i', 'l', "'", 'k', 'a', 'h', 'u', 'd', 'ū']|
|quran-phonetics|73|1|incomplete_distinguishing_phrase|['y', 'ā', 'y', 'u', 'h', 'a', 'l', 'm', 'u', 'z', 'a', 'n']|
|quran-phonetics|77|1|incomplete_distinguishing_phrase|['w', 'a', 'l', 'm', 'u', 'r', 'u', 's', 'a', 'l', 'ā', 'k', 'y', 'a', 'r', 'f', 'ā', 'f', 'a', 'l', 'ā', 'k', 'h', 'i', 'f', 'ā', 't']|
|quran-phonetics|80|1|incomplete_distinguishing_phrase|['f', 'a', 'a', 'n', 't', 'a', 'l', 'a']|
|quran-phonetics|98|2|incomplete_distinguishing_phrase|['m', 'i', 'n', 'a', 'l', 'ā', 'y', 'a', 't', 'l', 's', 'k', 'u', 'f', 'a', 'n', 'ṭ', 'a', 'h', 'ā', 'f', 'i', 'n', 'ā']|
|quran-phonetics|112|1|incomplete_distinguishing_phrase|['w', 'a', 'l', 'a', 'y', 'a', 'h', 'u', 'l', 'l', 'a']|
|quran-phonetics|2|1|incomplete_distinguishing_phrase|['a', 'l', 'i', 'h', "'", 'l', 'a']|
|quran-phonetics|79|1|incomplete_distinguishing_phrase|['y', 'a', 'w', 'm', 'a', 't', 'a', 'r', 'j', 'u']|
|quran-phonetics-fp32|1|2|incomplete_distinguishing_phrase|['m', 'i', 'l', 'h', 'a', 'd', 'u', 'l', 'i', 'l', 'ā', 'r', 'a', 'b', 'i', 'l', 'ā', 'l', 'a', 'm', 'ī']|
|quran-phonetics-fp32|32|1|incomplete_distinguishing_phrase|['a', 'l', 'i', 'h', "'", 'l', 'a']|
|quran-phonetics-fp32|55|1|incomplete_distinguishing_phrase|['w', 'a', 'l', 'n', 'a', 'j', 'n', 'm', 'w', 'a', 's', 'h', 'a', 'j', 'a', 'r', 'u']|
|quran-phonetics-fp32|59|1|incomplete_distinguishing_phrase|['w', 'a', 'q', 'a', 'd', 'a', 't', 'h', 'a', 'h', 'ī']|
|quran-phonetics-fp32|65|1|incomplete_distinguishing_phrase|['w', 'a', 't', 'i', 'l', "'", 'k', 'a', 'h', 'u', 'd', 'ū']|
|quran-phonetics-fp32|73|1|incomplete_distinguishing_phrase|['y', 'ā', 'y', 'u', 'h', 'a', 'l', 'm', 'u', 'z', 'a', 'n']|
|quran-phonetics-fp32|77|1|incomplete_distinguishing_phrase|['w', 'a', 'l', 'm', 'u', 'r', 'u', 's', 'a', 'l', 'ā', 'k', 'y', 'a', 'r', 'f', 'ā', 'f', 'a', 'l', 'ā', 'k', 'h', 'i', 'f', 'ā', 't']|
|quran-phonetics-fp32|80|1|incomplete_distinguishing_phrase|['a', 'b', 'a', 's', 'a', 'w', 'a', 't', 'a', 'w', 'l', 'l', 'ā']|
|quran-phonetics-fp32|98|2|incomplete_distinguishing_phrase|['m', 'i', 'n', 'a', 'l', 'ā', 'y', 'a', 't', 'l', 'ū', 's', 'k', 'u', 'f', 'a', 'n', 'ṭ', 'a', 'h', 'ā', 'f', 'i', 'n', 'ā']|
|quran-phonetics-fp32|112|1|incomplete_distinguishing_phrase|['w', 'a', 'l', 'a', 'y', 'a', 'h', 'u', 'l', 'l', 'a']|
|quran-phonetics-fp32|2|1|incomplete_distinguishing_phrase|['a', 'l', 'i', 'h', "'", 'l', 'a']|
|quran-phonetics-fp32|79|1|incomplete_distinguishing_phrase|['y', 'a', 'w', 'm', 'a', 't', 'a', 'r', 'j', 'u']|
|whisper-tarteel-tiny|1|1|future_ayah_without_current|['الحمد', 'لله']|
|whisper-tarteel-tiny|32|1|incomplete_distinguishing_phrase|['الف', 'لام', 'ميم']|
|whisper-tarteel-tiny|55|1|future_ayah_without_current|['علم', 'القرر']|
|whisper-tarteel-tiny|59|4|incomplete_distinguishing_phrase|['ذلك', 'بانهم', 'شاء']|
|whisper-tarteel-tiny|65|1|no_local_evidence|['وتلك', 'حدود']|
|whisper-tarteel-tiny|73|1|future_ayah_without_current|['قم', 'الليل']|
|whisper-tarteel-tiny|77|7|incomplete_distinguishing_phrase|['ان', 'ما', 'تريد']|
|whisper-tarteel-tiny|80|2|future_ayah_without_current|['وما', 'يدريك', 'لعله']|
|whisper-tarteel-tiny|98|5|incomplete_distinguishing_phrase|['وما', 'امروا', 'الا', 'ليعبد']|
|whisper-tarteel-tiny|112|None|None|None|
|whisper-tarteel-tiny|2|4|future_outside_bounded_block|['ختم', 'الله']|
|whisper-tarteel-tiny|79|1|future_ayah_without_current|['والناشطات', 'نشر']|
|whisper-tarteel-base|1|1|future_ayah_without_current|['الحمد', 'لله']|
|whisper-tarteel-base|32|1|incomplete_distinguishing_phrase|['برحمن', 'الرحيم']|
|whisper-tarteel-base|55|4|future_ayah_without_current|['الشمس', 'والعظيم']|
|whisper-tarteel-base|59|8|future_ayah_without_current|['والذين', 'تبوءوا', 'الدار']|
|whisper-tarteel-base|65|1|no_local_evidence|['وتلك', 'حدور']|
|whisper-tarteel-base|73|1|future_ayah_without_current|['قم', 'الليل']|
|whisper-tarteel-base|77|16|future_outside_bounded_block|['كذلك', 'نفعل', 'بالمجرمين']|
|whisper-tarteel-base|80|2|future_ayah_without_current|['وما', 'يدريك', 'لعله']|
|whisper-tarteel-base|98|8|incomplete_distinguishing_phrase|['رضي', 'الله']|
|whisper-tarteel-base|112|None|None|None|
|whisper-tarteel-base|2|1|future_ayah_without_current|['ذلك', 'الكتاب']|
|whisper-tarteel-base|79|1|future_outside_bounded_block|['فالسابقات']|

Приоритеты: Tiny77:7 — наблюдаемый split (отдельно от многих прочих отказов); Base79:26 — текст подтверждён, но timing +508 мс; CTC/FP32 имеют иные слабые/неполные независимые чтения и не исправляются ASR split. Review после block — каскад, а не число независимых модельных ошибок. Причины first_block сохранены без переименования в уверенные акустические диагнозы. Для всех случаев ledger проверен runner spy; интервалы и runtime identity в raw JSON.
