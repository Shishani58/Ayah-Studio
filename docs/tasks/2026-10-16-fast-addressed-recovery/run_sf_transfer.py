"""Fresh cross-reciter controls of the minimal SF ablation, one model at a time."""
import json
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
while not (P / 'SF-development-process.json').exists():
    time.sleep(1)
assert json.loads((P / 'SF-development-process.json').read_text())['exit'] == 0
assert json.loads((P / 'SF-minshawi-process.json').read_text())['exit'] == 0
for reciter in ('saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi'):
    for variant in ('A', 'SF'):
        label = 'final-' + variant + '-' + reciter
        subprocess.run([sys.executable, '-u', str(P / 'run.py'), variant, label,
            '--surahs', '65', '73', '80', '98', '112', '55', '79', '67', '77',
            '--reciter', reciter], check=True)
(P / 'SF-transfer-complete.json').write_text(json.dumps({'fresh_pairs': 27,
    'final_holdouts': [67, 77], 'serial': True}, indent=2) + '\n')
