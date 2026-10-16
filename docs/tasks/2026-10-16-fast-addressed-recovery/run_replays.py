"""Validate final B4 on every fresh B2 corpus observation, preserving refusals."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
root = Path(json.loads((P / 'snapshot-paths.json').read_text())['B4'])
labels = ['B2-wide', *['B2-' + r for r in (
    'saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi')]]
results = []
for label in labels:
    while not (P / (label + '-process.json')).exists():
        time.sleep(1)
    status = json.loads((P / (label + '-process.json')).read_text())
    if status['exit']:
        results.append({'source': label, 'source_exit': status['exit'], 'exit': None})
        continue
    with (P / (label.replace('B2', 'B4', 1) + '-replay.log')).open('w') as log:
        started = time.monotonic()
        result = subprocess.run([sys.executable, '-u', str(P / 'replay.py'), label, 'B4'],
            cwd=root, env={**os.environ, 'PYTHONPATH': str(root)}, stdout=log, stderr=subprocess.STDOUT)
    results.append({'source': label, 'exit': result.returncode, 'seconds': time.monotonic() - started})
    (P / 'B4-replay-status.json').write_text(json.dumps(results, indent=2) + '\n')
    print(label, 'B4 replay', result.returncode, flush=True)
