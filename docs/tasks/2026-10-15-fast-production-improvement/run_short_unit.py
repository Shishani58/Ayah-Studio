"""Serial fresh H9 development pairs, no reference-guided crop selection."""
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
for variant in ['baseline', 'prototype']:
    argv = [sys.executable, '-u', str(P / 'run.py'), 'short-unit-' + variant,
            '--variant', variant, '--models', 'whisper-tarteel-tiny', 'whisper-tarteel-base',
            '--surahs', '65', '73']
    result = subprocess.run(argv)
    if result.returncode:
        raise SystemExit(result.returncode)
