# Замедление FULL второй суры: macOS scheduling

Историческое исследование16 октября. LatencyCritical снят17 октября после
проверки нагрева;4:42 ниже — отдельный прежний контроль, не стабильная гарантия.
[Текущая проверка и ограничения](../2026-10-17-silence-overlap-energy/report.md).

## Причина и контроль

Последний пользовательский FULL `f3d8581a7d214b65afee83af76ccfd39`
завершился 16 октября в21:13. Из примерно48мин inference занимал2803.439s
(46:43), поиск пауз32.474s. Все346 акустических окон побайтово совпадают
с прежним `c9420872bf24434eafb6126f70cb1e70` (inference350.539s).
В более ранних снимках inference198.552–241.299s. Весы SHA423777f2…1589a,
аудио, window20/context2/batch1/workers3 прежние. Изменение text-only prior
не объясняет замедление акустического этапа: prior применяется после inference.

У ещё работавшего исходного GUI PID46463 прочитаны публичные kernel rusage
счётчики. CPU user+system2759.31s, из них1394.89s в background,
1353.92s в legacy, только1.13s в userInitiated. Суммарное runnable time всех
потоков25225.31s. Это агрегат потоков, а не длительность работы приложения.
Время переведено из Mach ticks через фактический timebase125/3.
`original-process-rusage.json` содержит исходные значения и пересчёт.
Полной покадровой истории scheduler у старого запуска нет.

В изолированном опыте одни и те же6 окон на тех же весах:

| Режим процесса | Wall, s | CPU, s |
| --- | ---: | ---: |
| Обычный | 4.142 | 25.556 |
| macOS background (`taskpolicy -b`) | 105.200 | 91.313 |

Политика менялась исключительно у одноразового исследовательского процесса
и затем восстанавливалась. Пользовательский GUI/системные настройки не менялись.
Отмена/RunOptions причиной не оказались: 4.259/4.170/3.803s для шести окон
без отмены/с включённой отменой/снова без неё (`probes.json`).

Native pywebview выявил дополнительное ограничение: одна
`NSActivityUserInitiated` предотвращает сон, но после простоя сессии во время
DSP остаётся utility QoS clamp. В контрольном опыте с35s ожидания:
8.684s с обычной activity и3.952s с добавлением `NSActivityLatencyCritical`,
без ручного изменения QoS. Все6 массивов logits побайтово одинаковы
(`native-pool-delayed.json`, `native-pool-critical-only.json`).
Промежуточные незавершённые FULL не выдаются за успешные проверки.

Прямое изменение QoS внешних Python-потоков не устраняло ограничение процесса;
этот код удалён. Вариант intra-op1/workers6 тоже не принят: короткий опыт
показал отличающиеся logits (max1.3779), полный опыт остановлен без результата.
Параметры ONNX, точность чисел и параллелизм production не изменены.

## Исправление

GUI теперь объявляет активный расчёт как ограниченную по времени native
activity `NSActivityUserInitiated | NSActivityLatencyCritical`. Она закрывается
через finally при успехе, ошибке и отмене. Защита действует также при
синхронной загрузке/проверке выбранной модели и перепроверке границ.
Экран может гаснуть; защита не остаётся после расчёта. Другие платформы и
macOS browser-only без PyObjC не требуют новой зависимости.

В FULL trace дополнительно сохраняются CPU time inference и фактические
версии/runtime options. Прежний wall time сохранён. Это диагностика,
не калибровка результата.

Модель, нормализация,346 окон, число вызовов, DSP-границы и текстовый prior
не изменены. Рабочий runtime не получает эталонных времён.

После одной activity полный native прогон занял366.344s — точность уже
сохранена, но4–5мин ещё не достигнуты. Устранены два остаточных расхода:

- FULL держал bounded deque и ждал старейший future. При длинном раннем окне
  более поздние завершались, а освободившиеся слоты простаивали. Теперь ожидание
  FIRST_COMPLETED, максимум по-прежнему3 вызова; результат/trace укладываются
  по исходному индексу. Тест с gate требует запуска третьего окна до окончания
  первого — прежняя очередь такой сценарий выполнить не могла.
- Exact CTC создавал stack трёх предшественников и argmax на каждом кадре.
  Повторно используемый float64 scratch и строгие сравнения сохраняют tie order
  stay→step→skip. Для non-finite оставлена прежняя argmax-семантика.
  Сложность остаётся O(T×S), backpointer по-прежнему1byte/состояние/кадр.
  Контроль1200 случаев: пути/оценки и256 отказов точно совпали. На одной форме
  13000frames/1100tokens пары before0.5967/0.5927s, after0.3459/0.3160s:
  примерно1.8× быстрее; это отдельный CTC benchmark, не время полного приложения.

Другие настройки не включены. На32окнах baseline24.39→31.20s в начале/конце
серии, workers2 —27.08s, dynamic blocks —23.88s, no arena —26.24s, logits
точно совпали. Выигрыш dynamic blocks меньше фоновой вариации. Workers4
в длинном опыте замедлял первую часть и был остановлен. Окончательный workers3.

Источники API: [Apple: app activities](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheAppLevel.html),
[NSActivityOptions](https://developer.apple.com/documentation/Foundation/ProcessInfo/ActivityOptions).
Расширенная activity выбрана по native измерению; один UserInitiated на этом
хосте недостаточен. Ограничение энергосбережения действует только на работу,
которую пользователь явно запустил и ожидает завершить.

## Проверка

| Полный native FULL2 | Activity only | Итог |
| --- | ---: | ---: |
| Полное время, s | 366.344 | **281.896 (4:42)** |
| DSP, s | 27.181 | 28.762 |
| Acoustic inference, s | 273.013 | 218.330 |
| Подготовка/CTC/прочие проверки, s | 66.149 | 34.804 |

Оба прогона в настоящем скрытом pywebview окне, свежий inference всего аудио
7086.58s, без emission/result cache и без записи в пользовательскую БД.
В итоге все286 строк, confidence, полные per-ayah проверки и346 окон точно
совпали с сохранённым text-only FULL. Source SHA до/после финального запуска
совпали. Снимок сравнения читается только после сохранения новых предсказаний,
никогда не передаётся runtime. `validation.json` фиксирует также порядок286
аятов, цель<300s, неизменность SHA/mtime пользовательской SQLite и снятие
native assertion после завершения.

Все997pytest passed/10skipped,77 целевых, Ruff/diff-check/AGENTS-CLAUDE sync
прошли. `verify_ctc.py` сохранил1200 дифференциальных проверок и парный benchmark.
Нативная проверка выполнена на macOS; Windows/пересборка дистрибутива не выполнялись.
Это сохранение прежнего результата, не переоценка его абсолютной точности.
Для применения в уже запущенном GUI нужен перезапуск. Коммита/пуша нет.

Воспроизведение из корня проекта:

```sh
rtk proxy .venv/bin/python docs/tasks/2026-10-16-full-runtime-regression/native_full.py --output /tmp/full2-verification.json
rtk proxy .venv/bin/python docs/tasks/2026-10-16-full-runtime-regression/verify_ctc.py --benchmark
rtk proxy .venv/bin/python docs/tasks/2026-10-16-full-runtime-regression/native_pool_probe.py --six --delay 35 --basic-activity --output /tmp/basic-activity.json
rtk proxy .venv/bin/python docs/tasks/2026-10-16-full-runtime-regression/native_pool_probe.py --six --delay 35 --output /tmp/protected-activity.json
rtk proxy .venv/bin/pytest -q
```

Изменены runtime: `webgui/activity.py`, `webgui/server.py`, `core/full_windows.py`,
`core/_ctc.py`; тесты: `test_generation_activity.py`, `test_full_windows.py`,
`test_ctc_ties.py`. Документация: `docs/project.md`, `docs/STATUS.md`, этот отчёт
и `task.md`; внешние диагностические scripts/JSON сохранены рядом.
