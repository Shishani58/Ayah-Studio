"""Fresh final cross-reciter pairs and new whole-record controls for selected BS."""
import json
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
while not (P / 'BS-wide-process.json').exists():
    time.sleep(1)
assert json.loads((P / 'BS-wide-process.json').read_text())['exit'] == 0
manifest = json.loads((P / 'final-control-manifest.json').read_text())
assert all(c.get('outcome') == 'downloaded' for c in manifest['cases'])
for reciter in ('saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi'):
    for variant in ('A', 'BS'):
        label = 'selected-' + variant + '-' + reciter
        print('start', label, flush=True)
        subprocess.run([sys.executable, '-u', str(P / 'run.py'), variant, label,
            '--surahs', '65', '73', '80', '98', '112', '55', '79', '67', '77', '--reciter', reciter], check=True)
(P / 'selected-transfer-complete.json').write_text(json.dumps({'cases_per_variant': 27,
    'fresh_inference': True, 'new_final_control_surahs': [67, 77]}, indent=2) + '\n')
