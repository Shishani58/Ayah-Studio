# Устойчивость локальной паузы FULL

## Цель и скоуп
Исправить поздний перенос границы 1:3 в последнем пользовательском FULL Шатри:
11598 → 11675 мс. DSP по текущему окну выбрал 11600–11750 мс, по расширенному
окну — другую площадку. Сохранить исходную границу и ревью при расхождении;
не переносить вправо начало слова с недостаточными доказательствами.
FAST, общий DSP detector, phrase repairs, ручные записи и пользовательскую БД
не менять. В workspace уже есть параллельные изменения; сохранять их.

## Evidence
Прочитаны AGENTS/CLAUDE, docs/project.md (FULL/onset policy), docs/STATUS.md,
full_pause_center.py, candidates.detect_energy_pauses/_refine_pause_edges,
align._check_ayah_onsets, tests/test_full_pause_center.py/test_alignment_audio.py.
Лог: Desktop/Abu_Bakr_Ash-Shatri/Abu_Bakr_Ash-Shatri.logs/sura_001_full.json.
Эталон 1:3=11561 мс используется только при проверке, не при предсказании.

## Решения
Повторный DSP в окне, расширенном на800мс с обеих сторон. Требуется ровно одна
принадлежащая межаятной границе строгая DSP-пауза в каждом окне, оба края должны
совпадать с допуском80мс. При расхождении — исходное время/reliable=False,
trace с обоими окнами. Слабое слово не переносится вправо. Дополнительных
acoustic calls нет: неоднозначность остаётся для слухового/локального ревью.
Новая onset policy заставляет resume пересчитать auto-метки; edited/verified
остаются защищены штатным writer.

## Проверка
- [x] Regression tests: контекст, слабое слово, порядок, intro, bounded FULL.
- [x] Свежий FULL Шатри1/112/114 и диагностика 1:3.
- [x] Полный pytest, scoped Ruff, diff-check.
- [x] Docs/STATUS и результаты/ограничения.

## Результаты
44 целевых теста passed; полный pytest778 passed/8 skipped за11.71с.
Opt-in свежий FULL1/112/114:3 passed за9.53с. Дополнительно сохранены свежие
trace/времена в fresh-full.json. Фатиха1:3 остаётся11598мс вместо позднего
11675мс; ошибка к эталону11561мс сокращается114→37мс. 1:4 сохраняет15937мс
вместо15990 из-за weak-word-right-shift. 1:5/7 также отклоняют нестабильные
площадки. Число аятов, порядок, repaired1:2 сохранены.112/114 проходят
прежние acoustic bounds и проверки размещённых меток в паузе.
Первый диагностический скрипт остановился на отсутствующем baseline112;
повторный скрипт допускает отсутствующий baseline и сохраняет все3 trace.
Scoped Ruff/diff-check и побайтовый AGENTS/CLAUDE sync — OK.

Изменены full_pause_center.py, FULL_ONSET_POLICY в full_windows.py,
test_full_pause_center.py, test_alignment_audio.py. Docs updated:
docs/project.md, docs/STATUS.md, этот task и fresh-full.json.
Блокеров нет. Приложение не пересобиралось, пользовательская БД не менялась,
коммитов нет. Полный корпус/слуховая приёмка не выполнены. Неоднозначные
площадки требуют ревью; новый acoustic phrase verifier не добавлялся.
Изменений CTC/склейки/глобального порядка нет, длинный fresh inference
не повторялся; bounded FULL проверен существующим синтетическим pipeline-тестом.
