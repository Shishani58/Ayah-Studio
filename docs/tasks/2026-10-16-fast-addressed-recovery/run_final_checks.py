"""Finish frozen-source evidence; never modify production or use reference in search."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
root = roots['BS2']


def wait_for(name):
    while not (P / name).exists():
        time.sleep(1)
    return json.loads((P / name).read_text())


def replay(label):
    log_path = P / ('BS2-' + label + '-replay.log')
    with log_path.open('w') as log:
        subprocess.run([sys.executable, '-u', str(P / 'selected_replay.py'), label, 'BS2'],
                       cwd=root, env={**os.environ, 'PYTHONPATH': root}, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print('exact PCM replay complete', label, flush=True)


assert wait_for('BS-wide-process.json')['exit'] == 0
replay('BS-wide')
wait_for('selected-transfer-complete.json')
for reciter in ('saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi'):
    replay('selected-BS-' + reciter)
subprocess.run([sys.executable, str(P / 'analyze.py')], check=True)
(P / 'final-corpus-complete.json').write_text(json.dumps({'replays_complete': True}) + '\n')
# All model inference is serial: full corpus, transfer pairs, then warm/cost runs.
for variant in ('A', 'BS2'):
    subprocess.run([sys.executable, '-u', str(P / 'run.py'), variant, 'final-warm-' + variant,
        '--surahs', '2', '79', '80', '2', '79', '80', '2', '79', '80'], check=True)
for variant in ('AP', 'BS2P'):
    label = 'final-profile-' + variant
    subprocess.run([sys.executable, '-u', str(P / 'run.py'), variant, label,
        '--surahs', '2', '79', '80'], check=True,
        env={**os.environ, 'FAST_PROFILE_DIR': str(P / (label + '-calls'))})
(P / 'final-costs-complete.json').write_text(json.dumps({'serial': True, 'warm_repeats': 3}) + '\n')
