# FAST: Muaalem independent local acoustic cause probe

Продолжение общей цели; цель остаётся active. Предыдущий goal turn — progress:
проанализированы40 window variants и добавлен проверенный bounded DSP recovery.
ОсновнаяCTC пока не исправлена. Проверить installed Muaalem на тех же local
1:2 crops: слышит ли альтернативный phoneme head отсутствующий onset/whole word.

Прочитано: project.md/STATUS и Base-window report; RuntimeEmitter.build_targets,
model_worker.targets/phoneme metadata, CTCAdapter.observe, scalar prefix kernels.
Целевой словарь Muaalem строится штатным deterministic phonetizer целых аятов;
это static text processing, не acoustic pass. Не менять canonical words/cache,
weights/зависимости/defaults/production adapters. All reads через LocalEvidenceReads.

Решения: только локальные1.5–3.5s входы, raw greedy/beam/qualities/word spans,
source/runtime/PCM hashes, независимый dispatch spy. Нельзя объявлять слово
услышанным по одному forced path. Word ownership/spans проверить до adapter.

- [x] Static phoneme mappings и partition check.
- [x] Свежие local reads, temporal/wrong/silence controls и raw evidence vs QPC.
- [x] Свежие experimental1/32/112, bounded native crop extension.
- [x] Corpus10+2+79 completed; unsafe owner55:14 разложен по canonical words/phones.
- [ ] Canonical word identity floor/variant-safe rivals до model activation.
- [ ] Minimal declared letter-name boundaries32 без длинного чтения всех имён.
- [ ] Native edge extension сравнить на production models до переноса в runtime.

Pytest не запускать до общего финала цели. Docs updated: этот каталог.

Попытка1:0audio model reads — Phonetics constructor требуетweights_sha256,
но loaded Muaalem identity используетassets[muaalem/model.fp32.onnx]. Это
несовместимость metadata schema, не ошибка модели/точности. Для standalone
adapter использовать реально загруженный immutable asset SHA, не обходить
идентификацию weights. Failed stdout сохранён attempt-1-constructor.log.

Попытка2:3local reads выполнились, word spans partition passed, но общий
benchmark spy выбралASR из-заhasattr(recognize_local), присутствующего у любого
RuntimeEmitter. Фактический Muaalem вызов —CTC emissions. Verification failed;
raw arrays этой попытки не экспортированы из-за позднего write. Три входа
1.5/2.1/3.0s учтены как повторные невалидированные измерения; не названы успехом.
Исправить classifier spy и добавить checkpoints до verification.

Артефактный дефект: pipeline_probe.py hardcoded pipeline.json и при edge-prototype
перезаписал первоначальный control raw JSON. Старыйstdout pipeline.log сохраняется,
но raw control не считается доступным. Candidate сохранён edge-pipeline.json и
summary; fresh control будет заново измерен в отдельный файл. Не rekey/relabel
этот replay как первоначальные реальные reads; повторные inputs учесть.
