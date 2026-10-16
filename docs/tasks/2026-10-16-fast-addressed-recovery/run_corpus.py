"""Serial fresh paired corpus; keep results and failures for every stage."""
import json
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
for label in ('B-development', 'B2-development'):
    while not (P / (label + '-process.json')).exists():
        time.sleep(1)
    assert json.loads((P / (label + '-process.json')).read_text())['exit'] == 0

jobs = []
for variant in ('A', 'B2'):
    jobs.append((variant, variant + '-wide', list(range(1, 115)), None))
for reciter in ('saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi'):
    for variant in ('A', 'B2'):
        jobs.append((variant, variant + '-' + reciter, [65, 73, 80, 98, 112, 55, 79], reciter))
for variant, label, surahs, reciter in jobs:
    command = [sys.executable, '-u', str(P / 'run.py'), variant, label,
               '--surahs', *map(str, surahs)]
    if reciter:
        command.extend(['--reciter', reciter])
    (P / 'checkpoint.json').write_text(json.dumps({'stage': label, 'argv': command,
        'started_unix': time.time(), 'status': 'running'}, indent=2) + '\n')
    print('start', label, flush=True)
    code = subprocess.run(command, check=False).returncode
    if code:
        (P / 'checkpoint.json').write_text(json.dumps({'stage': label, 'status': 'failed',
                                                     'exit': code}, indent=2) + '\n')
        raise SystemExit(code)
(P / 'checkpoint.json').write_text(json.dumps({'stage': 'corpus_complete',
    'status': 'awaiting_rowwise_gate'}, indent=2) + '\n')
