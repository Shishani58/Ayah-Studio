"""Three fixed warm repeats per source/model/case, after wide safety gate."""
import json
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
while not (P / 'f-wide-gate.json').exists():
    time.sleep(1)
gate = json.loads((P / 'f-wide-gate.json').read_text())
assert gate['passed'], 'Do not continue a rejected production candidate'
for variant in ['baseline', 'prototype-c']:
    command = [sys.executable, '-u', str(P / 'run.py'), 'warm-' + variant,
               '--variant', variant, '--models', 'whisper-tarteel-base',
               '--surahs', '65', '65', '65', '73', '73', '73']
    print('start warm', variant, flush=True)
    code = subprocess.run(command).returncode
    print('exit warm', variant, code, flush=True)
    if code:
        raise SystemExit(code)
