"""Exclusive serial warm repeats, then instrumented encoder/decoder cost."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
while not (P / 'B4-replay-status.json').exists() or len(json.loads((P / 'B4-replay-status.json').read_text())) != 4:
    time.sleep(1)
statuses = json.loads((P / 'B4-replay-status.json').read_text())
assert all(s['exit'] == 0 for s in statuses), statuses
assert json.loads((P / 'legacy37-process.json').read_text())['exit'] == 0
for variant in ('A', 'B4'):
    label = 'warm-' + variant
    print('start', label, flush=True)
    subprocess.run([sys.executable, '-u', str(P / 'run.py'), variant, label,
        '--surahs', '2', '79', '80', '2', '79', '80', '2', '79', '80'], check=True)
for variant in ('AP', 'B4P'):
    label = 'profile-' + variant
    directory = P / (label + '-calls')
    print('start', label, flush=True)
    subprocess.run([sys.executable, '-u', str(P / 'run.py'), variant, label,
        '--surahs', '2', '79', '80'], check=True,
        env={**os.environ, 'FAST_PROFILE_DIR': str(directory)})
(P / 'costs-complete.json').write_text(json.dumps({'passed_execution': True,
    'warm_repeats': 3, 'separate_profile_runs': True}, indent=2) + '\n')
