# Независимая проверка родителя

Baseline до делегирования:
`/var/folders/xr/sk3g_x_53rdg9wkyjdkq44tc0000gn/T/surah-fast-fixes-baseline-11s_70li`.
15 исходных файлов + manifest. Родитель не редактирует реализацию параллельно агенту.

Проверяемые результаты:
- Noise/partial разрешает только конечное обоснованное расширение; случайный
  похожий префикс и отсутствующий в crop текст не подтверждаются.
- Repeat ownership различает одинаковый аят при независимом соседнем proof;
  tempo/order сами по себе не позволяют подменить пропуск поздним повтором.
- Spoken letters сохраняют canonical word map; нельзя обрезать различающее слово.
- ASR confidence зависит от proof-word tokens; weak discriminating token не
  скрывается средним, слабый посторонний хвост не заражает сильное начало.
- Long pause: полный DSP interval отдельно от speech crop, confidence/time proof
  не получается из неподтверждённой VAD-полки.
- Recovery не требует физически невозможного целого длинного аята в6s;
  first/wasl timing только с доказанным local temporal proof.
- Captured identity включает фактически загруженные assets/config/runtime;
  файл, изменённый после loading, не описывает старую живую session новым SHA.
- Spy учитывает все фактические PCM-passes, включая дополнительные timing/scoring;
  eviction не возвращает budget; failed/cancelled dispatch учитывается.
- Канонический набор/порядок/диапазон сохранён; edited/verified защищены в writer;
  исход после commit честный. Production остаётся gated при непрохождении M5.

Проверки после готовности: source diff от baseline, адресные независимые
negative probes, обычный pytest/scoped Ruff, независимый пересчёт свежих audio
артефактов и достаточная адресная проверка нового model path. Не запускать
свои модели одновременно с измерениями агента. Replay не заменяет новые
окна/decoder/timing. Старые frozen corpus/reference — материал для сравнения,
не данные, доступные alignment алгоритму.

Независимая проверка завершена: 104 адресных случая, полный pytest 784 passed /
7 opt-in skips, scoped Ruff и diff check чистые. Все 49 защищённых файлов
не изменены. Сверены frozen sparse-4 138 audio records, 15 derived inputs,
49 temporal diagnostic passes. Текущая sparse-5 после scalar context-confidence
правки отдельно проверена свежими четырьмя случаями 20/109 × Tiny/Base
(126 actual acoustic passes), исходники/SHA/бюджеты/ошибки сверены независимо.
ASR temporal proof остаётся неоткалиброванным; recovery закрыт частично, M5 FAIL.
Подробности и ограничения: independent-review.md.
