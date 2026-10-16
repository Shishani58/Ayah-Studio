"""Final E: exact full-corpus D replay and fresh cross-reader confirmation."""
import json
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
while not (P / 'warm-prototype-c.summary.json').exists():
    time.sleep(1)
assert json.loads((P / 'warm-prototype-c-command.json').read_text())['variant_effective'] == 'prototype-e'
subprocess.run([sys.executable, '-u', str(P / 'run_aux.py'), 'e-wide-replay', 'replay_e_wide.py', 'e-wide-replay'], check=True)
replay = json.loads((P / 'e-wide-replay.json').read_text())
assert len(replay) == 114 and all(r['rows_equal'] and r['windows_equal'] for r in replay)
for name in ['saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi']:
    command = [sys.executable, '-u', str(P / 'run.py'), 'final-e-' + name,
               '--variant', 'prototype-e', '--models', 'whisper-tarteel-base',
               '--surahs', '65', '73', '--reciter', name]
    print('start final E', name, flush=True)
    subprocess.run(command, check=True)
raise SystemExit(subprocess.run([sys.executable, str(P / 'compare_runs.py'), 'final']).returncode)
