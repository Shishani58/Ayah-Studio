# Независимое ревью действующего FAST

2026-10-14. Проверяется фактический единый FAST после включения и удаления
legacy. Переключение явно разрешено пользователем и само по себе не является
дефектом. Историческое непрохождение M5 остаётся ограничением качества.
Окончательный HEAD: 37f3309f76fa077651239fe4892575de3b5d1fa6; соседний чат завершён.
Diff/SHA и состояние — в `final-state.json`. После suite удалены только два
trailing newline bytes в tests/test_fast_reads.py; восстановление SHA проверено.
Повтор suite из-за этой косметической правки не запускался.

## Доказанные находки

### R1 · P1: auto-resume пропускает старый FAST без проверки идентичности

`surah_cutter/webgui/server.py:1297–1306`: `fast_resume_suras` вызывается только
при запрошенном `mode=fast`. При `mode=auto`, который штатно выбирает FAST для
60–600 секунд, используется лишь каноническая полнота `_processed_suras`.
`surah_cutter/gui/dashboard_view.py:135–142` имеет ту же развилку.

Воспроизведение: временная БД со всеми 30 аятами суры32, status=auto,
source=`legacy:requested_mode=auto:actual_mode=fast`, без run-key.
`fast_resume_suras` возвращает пустое множество; Web explicit FAST выбирает32;
auto + only_missing возвращает400 «all done». Ни новая модель, ни новая версия,
ни новая запись не участвуют в решении. В CLI такого resume вообще нет:
он пересчитывает обнаруженные записи, writer защищает ручные метки.

Это не перезапись ручных данных; проблема — старые auto-результаты могут остаться
вместо нового FAST. Прежнее утверждение о консервативном GUI resume применимо
только к explicit FAST. Исправление: один preflight для auto→FAST и explicit FAST;
без подготовленного ключа пересчитывать auto-метки, защищённую суру пропускать.
Проверка Qt здесь статическая: native Qt не установлен/не запускался.

### R2 · P2: текущая диагностика скрывает unresolved и реальные чтения

`surah_cutter/webgui/server.py:1450–1455` считает unresolved по удалённому
`pause_sequence_events`; новый executor выдаёт `review_required_ayahs` и
`transitions`. `static/app.js:533–534` отображает этот ноль пользователю.
`server.py:1746` берёт только `model_reads`, тогда как шлюз пишет `acoustic_reads`
(`core/fast_reads.py:92–109`); раздел окон и их количество в debug-log пусты.
Добавленные при миграции fallback для transitions/observed.text не исправляют
сам источник списка reads.

Воспроизведение: trace с одним завершённым чтением1000–2500мс и одним
review-required аятом. Summary: unresolved=0. Форматированный лог: нет окна,
«чтений окон:0». Факт REVIEW_REQUIRED в другом разделе виден, противоречие не
скрыто тестом. Raw ledger/JSON остаются правильными; дефект находится в проекции
для интерфейса. Исправление: брать новые канонические поля trace, поддерживать
старые логи отдельным compatibility fallback; выводить outcome и причины отказов.

### R3 · P2: auto→FAST не получает неизменяемую историю доказательств

`surah_cutter/data/alignment_result.py:128`: history создаётся только по
`requested_mode in (fast, full)`. При запрошенном auto фактический FAST корректно
попадает в source/run-key, но каждый новый запуск заменяет один `sura_NNN_auto.json`.
Предыдущий run_snapshot, локальные proofs и raw model rows теряются из истории.

Воспроизведение: два сохранения во временную БД с trace.auto_mode=fast.
Оба возвращают только latest json/log, отсутствует `run`, `runs/*.json` пуст,
latest содержит только marker второго запуска. БД сохранена корректно.
Исправление: историю определять по actual algorithm, отдельно сохраняя requested
mode; два auto→FAST запуска должны иметь два неизменяемых snapshots и один latest.

### R4 · P3: FP32 подписывается как основная CTC-модель

`surah_cutter/core/fast_adapters.py:98`: `CTCAdapter.model_id` всегда
`quran-phonetics`; `model_store.supports_fast` допускает `quran-phonetics-fp32`.
Adapter fake FP32 emitter воспроизводимо подписан `quran-phonetics`.
Weights SHA сохраняется, поэтому столкновение ключей разных весов этим не доказано.
Это ошибка имени модели в trace/source, а не доказанная подмена вычисления.
Исправление: передавать фактический ID вместе с bundle; не читать поздний ACTIVE_ID.
Реальные FP32 веса этой задачей не запускались.

Все воспроизведения: `probes.py` → `probes.json`. Они используют реальные
функции, временные БД и synthetic observations, без inference и без settings.

## Что осталось от прежних проблем качества

Независимый пересчёт138 raw sparse-4 результатов сверяет каждый error с
read-only эталонной БД, парные audio/DSP/reference SHA, actual dispatch lengths,
сумму и объединение окон. Итог в `historical-check.json`:

| Модель | Парный37: legacy → sparse-4 /564 | Все46: подтверждено /1475 | Все46: расхождения >500мс |
|---|---:|---:|---:|
| CTC | 276 → 28 | 33 | 1379 |
| Tiny | 395 → 230 | 305 | 1087 |
| Base | 411 → 302 | 404 | 997 |

Base79:26 +508мс среди подтверждений воспроизводится арифметически; это новое
непрослушанное расхождение, не adjudicated ошибка эталона. У CTC/Tiny таких
расхождений в этом историческом корпусе нет. Это **исторические** model runs,
не свежий benchmark окончательной sparse-5 после удаления legacy.

Большинство estimates имеют причину blocked_by_unresolved_ayah:
CTC1379, Tiny1098, Base1009. Sparse continuation устраняет часть ранних остановок,
но не решает распространение одного неразрешённого префикса на хвост записи.
Ноль уверенных ошибок при столь малом покрытии не доказывает полезность.
В raw корпусе posterior-only confirmations=0 и real wasl confirmations=0;
реализация новых ветвей не является доказательством их акустического выигрыша.

### Свежая проверка гипотезы о потерянном первом звуке

`ctc_preroll.py` → `ctc-preroll.json`: настоящее аудио Шатри001,
Quran Phonetics;3 actual model dispatches,5100мс суммарного входа,
union=3.22% записи. Pause5780–6620мс, цель1:2=`alḥamdu`.
Pre-roll200/400/600мс во всех случаях дал `lḥamdulilā`, без полного greedy prefix.
Spy сверил фактические PCM-срезы с ledger. FULL не вызывался, БД не записывалась.

Тем самым простой совет «добавить pre-roll и восстановить a» **не подтвердился**.
Сохранился mismatch акустического decode и канонического фонетического начала;
его нельзя автоматически объяснить только обрезанным окном. Это не доказательство
неверного произношения или права удалить букву из цели. Следующий шаг — сравнить
independent hypotheses/posterior на тех же локальных frames и проверить локальный
support слабого первого звука. Нужны ground truth/onset и слуховая adjudication,
а не понижение порога ради coverage. Старт окна сейчас фиксирован у right−200мс
(`fast_executor.py:692`), extensions меняют только end (`:699–700`), однако
изменение этого правила должно опираться на отдельное доказательство пользы.

## Статус прежних исправлений

| Область | Текущая оценка |
|---|---|
| Единый runtime путь | `align_fast` — тонкая обёртка; CTC/Tarteel пользуются одним executor. Legacy loop/readers удалены. |
| Локальность и recovery | Shared guard6с/65%/200%,8 candidates,3 ступени; нет вызова FULL/tiled в adapters/executor. Failed/cancelled inputs оплачиваются, eviction не возвращает бюджет. |
| Минимальные фразы/повторы | Whole-word/spoken-letter ownership, separate previous-tail, essential minimum quality и validator реализованы; улучшение на повторном109 доказано прежними свежими запусками. Повторы без различающего контекста остаются review. |
| CTC альтернативы | Beam/null/posterior/residual реализованы; механическое исправление exact-greedy-only подтверждено кодом/tests. Корпусного выигрыша posterior-only пока нет. |
| ASR качество | Attribution selected BPE tokens к proof units и минимум essential tokens реализованы. Это не внешняя калибровка ASR correctness. |
| ASR timing | Честно timing=False. First-onset/wasl не доказаны; inherited attention heads не включены. |
| Sparse и estimates | Future unique anchor допускается только из уже прочитанного окна; gaps conf0/review; большие ошибки priors сохраняются. Внешнее подтверждение wasl отсутствует. |
| Геометрия паузы | Long-decay gate есть; short860мс Base79:26 остаётся открытым случаем. Нельзя механически расширить порог по одному эталону. |
| Identity | Loaded weights/runtime/code/defaults, PCM/text/policy/DSP pool фиксируются; label FP32 и auto resume/history требуют R1/R3/R4. |
| Сохранение/отмена | Merge edited/verified и порядок проверяются под транзакцией; generated_rows отдельно от overlay; after-commit JSON errors имеют saved outcome. Адресные тесты входят в свежий suite. |
| Auto postprocessing | auto→FAST больше не вызывает lag calibration/AUTO_REPAIR; доказанные метки не сдвигаются после executor. |
| FULL после удаления | Runtime FULL/tiled не импортируют удалённые matcher/readers. Phonetics helpers сохранены; общий pytest проходит. Параллельные FULL изменения не приписываются миграции FAST. |

## Не включённые в доказанные дефекты гипотезы

Synthetic seam alpha/beta показывает, что planner/validator не проверяют поток
через границу чужих аятов. Но text-only scan114сур обоих представлений нашёл
**0 таких канонических коллизий**. Это возможный защитный negative-test на будущее,
не текущая причина плохого покрытия или доказанная ошибка на реальном Коране.
Временная ссылка model_store на удалённый align_fast_whisper исправлена соседним
чатом до конечного ревью; её не включаю в оставшиеся находки.

## Проверено и ограничения

Собственный полный pytest текущей рабочей копии: **602 passed /7 skipped,9.18с**.
7 skips — opt-in audio suite; мои3 CTC чтения выполнены отдельно, без заявления
о прохождении всех реальных аудиотестов. Source SHA104 Python файлов и49 data
файлов стабильны между snapshot и suite; AGENTS/CLAUDE побайтово идентичны.
Scoped Ruff runtime и собственных scripts прошёл. Global diff-check первоначально
нашёл лишнюю пустую строку в чужом tests/test_fast_reads.py; final-state фиксирует
конечное состояние: после исправления соседним чатом diff-check чистый.
Я этот файл не правил.

Сохранённые6 normal-pipeline model cases108/109 из unified-fast-default — evidence
соседнего чата; его tests/build не выдаются за мою независимую проверку. Сверка
manifest этих6 случаев с текущим исходным кодом обнаружила только align.py
(параллельная FULL-правка); мои3 fresh CTC reads совпали по всем captured SHA.
Данная задача не доказывает всю .app, native Qt, другого чтеца,114 acoustic runs,
слуховую точность79:26 или независимые wasl word onsets. Replay/fake и fresh
inference в отчёте разделены. Production/данные/настройки/режимы я не изменял,
commit/push не выполнял.

Следующий шаг с критериями — `plan.md`.
