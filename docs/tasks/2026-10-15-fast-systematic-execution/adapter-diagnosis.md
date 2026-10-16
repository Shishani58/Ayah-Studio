# Локальный CTC: установленные причины и пределы

Данные отделены от интерпретации. Свежие observations, PCM/hash/ledger и
runtime identities: local-diagnostics.json, local-ledger-*.json, local-ctc.log.
16реальных вызовов, две модели×четыре случая×два вложенных crop. Reference
не участвовал; word boundaries не размечены слухом.700ms — минимальный контекст,
а не доказанно изолированное слово. Поэтому training-domain mismatch остаётся
объясняющей гипотезой, а не доказанной физической причиной всех ошибок.

| Случай | INT8/FP32:700ms | INT8/FP32:1500ms | Вывод |
|---|---|---|---|
|1:1 контроль|bis'm / bis'm|bis'milā / bis'milā|Слишком короткий crop обрезает полное bis'mi; нельзя сокращать все чтения без cue |
|80:1|adh / adha|absawataw / abasawataw|Targetʿabasa отсутствует и после сокращения; missingʿ + внутренние ошибки остаются |
|98:2|fasū / fasū|fasūnū / fasūnū|Targetrasūlun не восстанавливается; это не только greedy-alternative loss |
|112:1|qul'hū / qul'hū|ul'mualā / ul'mualā|Context причинно меняет начальноеq при том же PCM происхождении и target; узкий положительный H1 |

Для112 короткий full qul имеет minimum essential quality .9003(INT8)/.9583(FP32).
Обычный crop target first-entry p≈.0121/.0129; короткий .8966/.9542. Greedy и
posterior — разные способы доказательства: порог posterior .9 не применяется
к greedy score. Ни один из этих scores не назван вероятностью temporal accuracy.

H3: сканирование только сохранённых frames по независимым greedy starts
не возвращает target обычного crop80/98/112. Максимумы — диагностические,
не нормированный внутренний event; перенос max-offset как posteriorproof отвергнут.
Forced path и residual не удаляются. Ошибка112 устраняется новым реальным
коротким inference, а не переименованием первого prefix event.

H2: representation-audit.json содержит исходный/CTC/ASR target для77432 слов,
114сур/6236аятов:0OOV,0non-NFC. Pad37 совпадает с pinnedconfig, vocabsize40;
добавленные38/39(<s>,</s>) документированы tokenizer_config. Гласные, apostrophes,
repeated phones и named-letter source labels не переписаны. Одно совпадение
word counts/алфавита не доказывает правильную фонетику. Документированного
нового несовместимого преобразования, разрешающего массовую правку, не найдено.

Pinned training metadata сохранены сSHA в primary-source-manifest.json.
[Карточка автора](https://huggingface.co/TBOGamer22/wav2vec2-quran-phonetics)
описывает word-level обучение, а [датасет](https://huggingface.co/datasets/Buraaq/quran-md-words)
содержит word_tr и ownership. Это согласуется с локальной чувствительностью
к контексту, но не является независимой слуховой верификацией произношения.

Вердикт INT8/FP32: строгий локальный FAST воспроизводимо имеет малое покрытие;
потеря материала возникает уже в независимом acoustic decode. Для отдельных
коротких слов исправима процедура выбора контекста, но пригодность общего
исправления требует межчтецового положительного gate. Квантование само по себе
не объясняет80/98/112: оба variants показывают те же классы ошибок.
Не исследовалась полная физическая причина/переобучение; defaults не меняются.

Tiny/Base: freshA сохраняет105/130 из607,2суры без опоры, timing=False.
Аудит всех сильных неиспользованных unique openings даёт только1:1:
полный текст есть, но отсутствует поддержанное первое время. Остальные
неиспользованные минимальные совпадения32/55 отвергаются alias/refrain
конкурентами. Новый cue/recovery bypass на этом основании не обоснован.
Смешение уверенности текста с первым onset не исправляет эту причину.
