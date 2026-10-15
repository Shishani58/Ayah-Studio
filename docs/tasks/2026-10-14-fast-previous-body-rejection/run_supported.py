"""Serial paired supported models, including installed FP32; no pytest."""
import json
import os
from pathlib import Path
import subprocess
import sys

task = Path(__file__).parent.resolve()
root = task.parents[2]
sources = json.loads((task / 'supported-snapshot-paths.json').read_text())
for model in ['quran-phonetics', 'quran-phonetics-fp32', 'whisper-tarteel-tiny', 'whisper-tarteel-base']:
    for variant in ['before', 'after']:
        name = f'{model}-{variant}'
        source = Path(sources[variant])
        args = [sys.executable, '-u', str(source / 'tools/bench_fast_models.py'), '--models', model,
                '--surahs', *map(str, [1, 32, 55, 59, 65, 73, 77, 80, 98, 112, 2, 79]),
                '--audio-dir', '/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri',
                '--reference', str(root / 'data/references/abu_bakr_ash_shatri.db'),
                '--output', str(task / f'{name}.json')]
        print('start', name, flush=True)
        with (task / f'{name}.log').open('w') as output:
            result = subprocess.run(args, cwd=source, env={**os.environ, 'PYTHONPATH': str(source)},
                                    stdout=output, stderr=subprocess.STDOUT)
        print('completed', name, 'exit', result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
