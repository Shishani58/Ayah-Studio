"""Paired corpus gates, then wide114 only if per-row safety gates pass."""
import json
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
models = ['--models', 'whisper-tarteel-tiny', 'whisper-tarteel-base']
stage = sys.argv[1]
jobs = []
if stage == 'corpus':
    for variant in ['baseline', 'prototype-b']:
        jobs.append((f'corpus-a-{variant}', variant, [1, 2, 32, 55, 59, 65, 73, 77, 79, 80, 98, 112], None))
        jobs.append((f'corpus-e-{variant}', variant, [9, 42, 83, 95, 109, 113, 114], None))
    for reciter in ['saad_al_ghamidi', 'mohammed_siddiq_al_minshawi',
                    'mahmoud_khalil_al_hussary', 'abdul_basit_mujawwad']:
        for variant in ['baseline', 'prototype-b']:
            jobs.append((f'corpus-{reciter}-{variant}', variant, [80, 98, 112], reciter))
elif stage == 'wide':
    gate = json.loads((P / 'corpus-gate.json').read_text())
    assert gate['passed'], gate
    for variant in ['baseline', 'prototype-b']:
        jobs.append((f'wide-{variant}', variant, list(range(1, 115)), None))
else:
    raise ValueError(stage)
for label, variant, surahs, reciter in jobs:
    argv = [sys.executable, '-u', str(P / 'run.py'), label, '--variant', variant,
            *models, '--surahs', *map(str, surahs)]
    if reciter:
        argv.extend(['--reciter', reciter])
    print('start', label, flush=True)
    result = subprocess.run(argv)
    print('exit', label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
