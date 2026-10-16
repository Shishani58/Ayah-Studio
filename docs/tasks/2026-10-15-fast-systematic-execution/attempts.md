# Реестр опытов

Все команды и exit — в *-command.json/checkpoint.json, stdout+stderr — *.log.

| ID | Причина / ожидаемое наблюдение | Фактический исход / решение |
|---|---|---|
| baseline-a | Свежий frozen baseline, без изменения алгоритма | exit1 до модели: snapshot не содержал checkout models. Сохранено, не считается акустическим отказом |
| baseline-a2 | После подключения имеющихся weights с SHA baseline должен повторить sparse12 | exit0;2/2/105/130, сравнение каждой строки/dispatch pending |
| local-ctc | H1:700ms против1500ms на1:1/80:1/98:2/112:1; H3: first-entry против поздних offsets на тех же frames |16calls;80/98 отрицательны;112700ms возвращает qul (INT8/FP32),1500ms теряетq. Это новый crop case, не повтор прежнего1:2 |
| short-a-dev | H1/H4: сильный уникальный remainder двух phones трёхфонемного целого opening разрешает одно500ms speech перечитывание той же паузы. Новый полный exact proof обязателен; при неуспехе исходное observation сохраняется |112 — development;109/113/114 — заранее выбранные аналоги. Не снижать пороги; не восстанавливать буквы. Результат pending |

Прототип изолирован в отдельном source root. Production не меняется до
причинной, межчтецовой и корпусной проверки. Новый короткий crop остаётся
в уже прочитанном интервале, не расширяет acoustic union и учитывается в sum.
Вся короткая гипотеза ограничена complete minimal opening из3phones, чтобы
не повторять отвергнутый длинный missing-a1:2 контекст и не запускать широкий
перебор длины окна.500ms speech выбран до новых аналогов, не подгоняется по ним.
Отрицательные/null/wrong-owner проверки готовятся; pytest не запускался.

B/C/D:8завершённых запусков=48случаев CTC,24before/after пары. Все rows и
dispatches совпадают, ни одного short-context cue. Четыре других чтеца не
дали положительного эффекта ветки. Это отсутствие применимого случая,
не доказательство пользы на другом чтеце. Межчтецовый gate не пройден.
Производственный перенос не разрешён доказательствами. AfterA/E и цена
завершаются для точного ограничения локального результата; F не запускается
после проваленного gate A–E. ASR corpus B–D исключён: новая branch требует
ctc_greedy_probability, ASR metric иной; A fresh ASR pairs всё равно проверяются.

Подготовка annotation-package: первая попытка завершилась AttributeError
PauseDecision.center_ms до завершения пакета, без model calls. Исправлено на
PauseDecision.pause.center_ms; annotations-command.json/annotations.log —
успешная повторная подготовка18WAV/SVG с незаполненными labels. Прослушивания нет.

Первичный Ruff нашёл3unused imports в task scripts; executed-scripts-prelint.zip
иSHA сохраняют использованный вариант. До/после акустический source root не меняется.
