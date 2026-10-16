"""Guarded production adoption, followed by the common final test phase."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
REPO = P.parents[2]

def dump(name, obj):
    (P / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def run(label, argv):
    assert not (P / (label + '-command.json')).exists(), 'Never repeat a launched ID'
    dump(label + '-command.json', {'argv': list(map(str, argv)), 'cwd': str(REPO),
                                  'started_unix': time.time(), 'PYTHONPATH': str(REPO)})
    start = time.monotonic()
    with (P / (label + '.log')).open('w') as out:
        process = subprocess.Popen(list(map(str, argv)), cwd=REPO,
            env={**os.environ, 'PYTHONPATH': str(REPO)}, stdout=out, stderr=subprocess.STDOUT)
        code = process.wait()
    dump(label + '-completion.json', {'exit': code, 'wall_s': time.monotonic() - start})
    print(label, 'exit', code, flush=True)
    return code

while not (P / 'final-gate.json').exists():
    time.sleep(1)
replay = json.loads((P / 'f-wide-replay.json').read_text())
assert len(replay) == 114 and all(r['rows_allowed'] and r['windows_equal'] for r in replay)
for gate_name in ['corpus-gate.json', 'f-wide-gate.json', 'final-gate.json']:
    assert json.loads((P / gate_name).read_text())['passed'], gate_name
source = json.loads((P / 'wide-baseline.summary.json').read_text())['source_sha256']
reviewed = json.loads((P / 'foreign-source-acceptance.json').read_text()) if (P / 'foreign-source-acceptance.json').exists() else {}
for name, expected in source.items():
    if name in reviewed:
        assert reviewed[name]['baseline_sha256'] == expected
        expected = reviewed[name]['current_sha256']
    assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name + ' changed concurrently'
for name, item in json.loads((P / 'pretransfer-text-data-guard.json').read_text()).items():
    assert item['equal'] and hashlib.sha256((REPO / name).read_bytes()).hexdigest() == item['measured']
assert (REPO / 'AGENTS.md').read_bytes() == (REPO / 'CLAUDE.md').read_bytes()
assert not (REPO / 'tests/test_fast_short_spoken_prefix.py').exists()
subprocess.run(['git', 'apply', '--check', str(P / 'candidate-f.patch')], cwd=REPO, check=True)
subprocess.run(['git', 'apply', str(P / 'candidate-f.patch')], cwd=REPO, check=True)
executor = REPO / 'surah_cutter/core/fast_executor.py'
old_version = 'sequential-fast-v3-sparse-12-split-opening'
new_version = 'sequential-fast-v3-sparse-13-local-prefix-review'
code = executor.read_text()
assert code.count(old_version) == 1
executor.write_text(code.replace(old_version, new_version))
roots = json.loads((P / 'snapshot-paths.json').read_text())
measured = (Path(roots['prototype-f']) / 'surah_cutter/core/fast_executor.py').read_text()
assert code == measured, 'Production differs from checked F beyond the version label'
assert ast.dump(ast.parse(executor.read_text().replace(new_version, old_version))) == ast.dump(ast.parse(measured))
tests = (P / 'regression_draft.py').read_text().replace(
    'Prepared behavioral checks; execute only in the final testing phase.',
    'Short spoken prefixes request evidence without weakening whole-word proofs.')
(REPO / 'tests/test_fast_short_spoken_prefix.py').write_text(tests)
roots['production'] = str(REPO)
dump('snapshot-paths.json', roots)
dump('production-transfer.json', {'executor_sha256': hashlib.sha256(executor.read_bytes()).hexdigest(),
    'measured_executor_sha256': hashlib.sha256(measured.encode()).hexdigest(),
    'difference': 'VERSION only; exact remaining source equality checked',
    'version': new_version, 'started_unix': time.time()})
print('Production implementation applied after all corpus gates', flush=True)
code = run('final-pytest', [sys.executable, '-m', 'pytest', '-q'])
ruff = run('final-ruff', [sys.executable, '-m', 'ruff', 'check',
                       'surah_cutter/core/fast_executor.py', 'tests/test_fast_short_spoken_prefix.py'])
if code or ruff:
    raise SystemExit(1)
code = subprocess.run([sys.executable, '-u', str(P / 'run.py'), 'production-smoke',
    '--variant', 'production', '--models', 'whisper-tarteel-base',
    '--surahs', '28', '41', '65', '72', '73', '82', '84'], cwd=REPO).returncode
if code:
    raise SystemExit(code)
expected = {(c['model'], c['surah']): c for c in json.loads((P / 'f-replay-predictions.json').read_text())}
for case in json.loads((P / 'production-smoke.json').read_text()):
    prior = expected[(case['model'], case['surah'])]
    assert case['rows'] == prior['rows']
    assert case['actual_dispatches'] == prior['actual_dispatches']
dump('production-smoke-gate.json', {'passed': True, 'cases': len(json.loads((P / 'production-smoke.json').read_text())),
      'rows_and_dispatches_equal_to_checked_F': True})
print('Implementation and common final tests complete; report review remains', flush=True)
