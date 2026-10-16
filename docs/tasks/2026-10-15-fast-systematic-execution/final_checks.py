"""M7 only: run current suite, isolated prepared prototype checks, static gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
REPO = P.parents[2]
paths = json.loads((P / 'snapshot-paths.json').read_text())
commands = [
    ('pytest', [sys.executable, '-m', 'pytest', '-q'], REPO, {}),
    ('prototype-contract', [sys.executable, '-m', 'pytest', '-q', '-c', '/dev/null',
         '-o', 'pythonpath=', '--import-mode=importlib', str(P / 'regression_draft.py')],
         Path(paths['prototype']), {'PYTHONPATH': paths['prototype'] + os.pathsep + str(REPO / 'tests')}),
    ('ruff', [str(REPO / '.venv/bin/ruff'), 'check', str(P)], REPO, {}),
    ('diff-check', ['git', 'diff', '--check'], REPO, {}),
    ('agents-sync', ['cmp', 'AGENTS.md', 'CLAUDE.md'], REPO, {}),
]
cp = json.loads((P / 'checkpoint.json').read_text())
cp['stage'] = 'M7'
cp['next'] = 'Concluding suite; inspect all outcomes, final guards/report/status.'
(P / 'checkpoint.json').write_text(json.dumps(cp, ensure_ascii=False, indent=2))
results = []
for label, argv, cwd, env in commands:
    assert not (P / f'{label}-command.json').exists(), 'Never overwrite a concluded stage'
    (P / f'{label}-command.json').write_text(json.dumps({'argv': argv, 'cwd': str(cwd), 'env': env}, indent=2))
    start = time.monotonic()
    with (P / f'{label}.log').open('w') as log:
        proc = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
        code = proc.wait()
    item = {'label': label, 'pid': proc.pid, 'exit': code, 'seconds': time.monotonic() - start}
    print(item, flush=True)
    results.append(item)
    (P / 'final-checks.json').write_text(json.dumps(results, indent=2))
