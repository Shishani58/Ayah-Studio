"""Wait for the existing CPU experiment, then validate H9c sequentially."""
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
while not (P / 'corpus-abdul_basit_mujawwad-prototype-b.summary.json').exists():
    time.sleep(1)
jobs = [('h9c-development', 'prototype-c', None)]
for reciter in ['saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi']:
    if reciter == 'mohammed_siddiq_al_minshawi':
        jobs.append((f'h9c-baseline-{reciter}', 'baseline', reciter))
    jobs.append((f'h9c-prototype-{reciter}', 'prototype-c', reciter))
for label, variant, reciter in jobs:
    argv = [sys.executable, '-u', str(P / 'run.py'), label, '--variant', variant,
            '--models', 'whisper-tarteel-tiny', 'whisper-tarteel-base', '--surahs', '65', '73']
    if reciter:
        argv.extend(['--reciter', reciter])
    print('start', label, flush=True)
    code = subprocess.run(argv).returncode
    print('exit', label, code, flush=True)
    if code:
        raise SystemExit(code)
