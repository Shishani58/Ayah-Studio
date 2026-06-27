# Улучшения алгоритма: auto, repair, lag, fingerprint, banded CTC

## Цель

Добавить оставшиеся улучшения алгоритма нарезки таймкодов без смены формата БД:
- `auto`-маршрутизация режимов;
- local repair-pass для подозрительных авто-таймингов;
- консервативная автокалибровка лага;
- opt-in fingerprint/import режим для per-ayah аудио;
- banded CTC внутри tiled с fallback на обычный CTC.

## Скоуп

- `core/`
- `pipeline.py`
- CLI/web API списки режимов
- документация задачи и статус

Тесты по указанию пользователя не запускать.

## Решения

- Старые `full`/`fast`/`tiled` не менять по умолчанию сильнее необходимого:
  новые рискованные шаги доступны через `auto` или opt-in режимы.
- Repair не трогает `edited`/`verified`; после review оставлен opt-in через
  `SC_AUTO_REPAIR=1`, пока не переиспользует emissions выбранного aligner.
- Banded CTC обязан иметь fallback на обычный CTC chunk; после review выключен
  по умолчанию (`SC_TILED_BAND_RADIUS=0`), чтобы `tiled` оставался точным.
- `fingerprint` оставлен для CLI/API с `--ayah-audio-dir`/env; из web-режимов
  убран, пока в UI нет поля выбора папки per-ayah клипов.

## Чек-лист

- [x] Task-файл.
- [x] Local repair-pass.
- [x] Auto-routing.
- [x] Lag calibration helper.
- [x] Fingerprint opt-in mode.
- [x] Banded CTC fallback path.
- [x] Синтаксическая проверка без тестов.

## Проверка

- `python3 -m py_compile surah_cutter/core/_ctc.py surah_cutter/core/align_tiled.py surah_cutter/core/fingerprint.py surah_cutter/core/lag.py surah_cutter/core/repair.py surah_cutter/core/pipeline.py surah_cutter/cli.py surah_cutter/webgui/server.py`
  — OK.
- Тесты/бенчмарки не запускались по указанию пользователя.

Docs updated: `docs/STATUS.md`,
`docs/tasks/2026-06-27-algorithm-suite/task.md`.

## Блокеры

Нет.
