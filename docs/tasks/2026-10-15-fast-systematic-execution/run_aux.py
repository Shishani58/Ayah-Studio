"""Capture a single auxiliary experiment; guard against duplicate IDs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
label, script = sys.argv[1:3]
root = Path(json.loads((P / 'snapshot-paths.json').read_text())['baseline']).resolve()
assert not (P / f'{label}-command.json').exists()
argv = [sys.executable, '-u', str(P / script), str(P), *sys.argv[3:]]
(P / f'{label}-command.json').write_text(json.dumps({'argv': argv, 'cwd': str(root),
    'PYTHONPATH': str(root), 'started_unix': time.time()}, indent=2))
start = time.monotonic()
with (P / f'{label}.log').open('w') as log:
    proc = subprocess.Popen(argv, cwd=root, env={**os.environ, 'PYTHONPATH': str(root)},
                            stdout=log, stderr=subprocess.STDOUT)
    cp = json.loads((P / 'checkpoint.json').read_text())
    cp['processes'].append({'label': label, 'pid': proc.pid, 'root': str(root),
                            'log': str(P / f'{label}.log'), 'status': 'running'})
    (P / 'checkpoint.json').write_text(json.dumps(cp, ensure_ascii=False, indent=2))
    code = proc.wait()
cp = json.loads((P / 'checkpoint.json').read_text())
for item in cp['processes']:
    if item['label'] == label:
        item.update(status='completed', exit=code, wall_s=time.monotonic() - start)
(P / 'checkpoint.json').write_text(json.dumps(cp, ensure_ascii=False, indent=2))
print(label, 'exit', code, flush=True)
raise SystemExit(code)
