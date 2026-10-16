"""Serial causal development and fixed cross-reciter confirmation."""
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
jobs = [('h9b-development', 'prototype-b', None)]
for reciter in ['saad_al_ghamidi', 'mahmoud_khalil_al_hussary']:
    jobs.extend([(f'h9b-{variant}-{reciter}', variant, reciter)
                 for variant in ['baseline', 'prototype-b']])
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
