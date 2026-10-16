"""Fresh D confirmation after the wide gate and uncontended warm measurements."""
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
while not (P / 'warm-prototype-c.summary.json').exists():
    time.sleep(1)
for name in ['saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi']:
    command = [sys.executable, '-u', str(P / 'run.py'), 'final-d-' + name,
               '--variant', 'prototype-d', '--models', 'whisper-tarteel-tiny', 'whisper-tarteel-base',
               '--surahs', '65', '73', '--reciter', name]
    print('start final D', name, flush=True)
    code = subprocess.run(command).returncode
    print('exit final D', name, code, flush=True)
    if code:
        raise SystemExit(code)
raise SystemExit(subprocess.run([sys.executable, str(P / 'compare_runs.py'), 'final']).returncode)
