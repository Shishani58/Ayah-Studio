"""Serial H9c negative/core gates and wide114 pairs, preserving every attempt."""
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
while not (P / 'h9c-prototype-mohammed_siddiq_al_minshawi.summary.json').exists():
    time.sleep(1)


def run(argv):
    print('start', argv, flush=True)
    result = subprocess.run([sys.executable, '-u', *map(str, argv)])
    print('exit', result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run([P / 'run_aux.py', 'negative-h9c', 'negative_audio.py'])
for label, surahs in [('corpus-a', [1, 2, 32, 55, 59, 65, 73, 77, 79, 80, 98, 112]),
                      ('corpus-e', [9, 42, 83, 95, 109, 113, 114])]:
    run([P / 'run.py', label + '-prototype-c', '--variant', 'prototype-c',
         '--models', 'whisper-tarteel-tiny', 'whisper-tarteel-base', '--surahs', *surahs])
run([P / 'compare_runs.py', 'corpus'])
for variant in ['baseline', 'prototype-c']:
    run([P / 'run.py', 'wide-' + variant, '--variant', variant,
         '--models', 'whisper-tarteel-tiny', 'whisper-tarteel-base',
         '--surahs', *range(1, 115)])
run([P / 'compare_runs.py', 'wide'])
