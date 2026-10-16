"""Complete smaller-candidate evidence; production remains under the lead's control."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())


def load(name):
    return json.loads((P / name).read_text())


def wait_for(name):
    while not (P / name).exists():
        time.sleep(1)
    return load(name)


def run(variant, label, surahs):
    print('fresh model run', label, surahs, flush=True)
    subprocess.run([sys.executable, '-u', str(P / 'run.py'), variant, label,
                    '--surahs', *map(str, surahs)], check=True)


def signature(case):
    return [(a, ms, q > 0) for a, ms, q in case['rows']]


status = wait_for('SF-wide-reuse-final-status.json')
assert status['expected'] == 114
known_fresh = {c['surah']: c for c in load('SF-development.json')}
needed = {c['surah'] for c in status['missing']}
additional = sorted(needed - known_fresh.keys())
# Cross-reciter model jobs finish first; never launch competing cost/model jobs.
wait_for('SF-transfer-complete.json')
if additional:
    subprocess.run([sys.executable, '-u', str(P / 'run.py'), 'PF', 'PF-wide-missing-observations',
        '--surahs', *map(str, additional), '--fill-missing'], check=True)
    assert load('PF-wide-missing-observations-status.json')['missing'] == []
    known_fresh.update({c['surah']: c for c in load('PF-wide-missing-observations.json')})
fresh_missing = [known_fresh[s] for s in sorted(needed)]
(P / 'SF-wide-fresh-missing.json').write_text(json.dumps(fresh_missing, ensure_ascii=False) + '\n')
combined = {c['surah']: c for c in load('SF-wide-reuse-final.json')}
for surah, fresh in known_fresh.items():
    if surah in combined:
        assert signature(fresh) == signature(combined[surah]), ('fresh_replay_disagreement', surah)
    combined[surah] = fresh
assert sorted(combined) == list(range(1, 115))
(P / 'SF-wide-source-combined.json').write_text(json.dumps(
    [combined[s] for s in range(1, 115)], ensure_ascii=False) + '\n')
with (P / 'PF-wide-replay.log').open('w') as log:
    subprocess.run([sys.executable, '-u', str(P / 'ablation_replay.py'), 'PF', 'A-wide',
                    'SF-wide-source-combined', 'PF-wide-replay'], cwd=roots['PF'],
                   env={**os.environ, 'PYTHONPATH': roots['PF']}, stdout=log,
                   stderr=subprocess.STDOUT, check=True)
assert load('PF-wide-replay-status.json')['missing'] == []
for reciter in ('saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi'):
    label = 'final-SF-' + reciter
    with (P / ('PF-' + label + '-replay.log')).open('w') as log:
        subprocess.run([sys.executable, '-u', str(P / 'selected_replay.py'), label, 'PF'],
                       cwd=roots['PF'], env={**os.environ, 'PYTHONPATH': roots['PF']},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
subprocess.run([sys.executable, str(P / 'analyze_sf.py')], check=True)
assert load('SF-corpus-gate.json')['passed'], 'quality gate failed; do not transfer to production'
(P / 'PF-final-corpus-complete.json').write_text(json.dumps({'passed': True}) + '\n')
with (P / 'PF-negative-audio.log').open('w') as log:
    subprocess.run([sys.executable, '-u', str(P / 'negative_audio_final.py')],
                   cwd=roots['PF'], env={**os.environ, 'PYTHONPATH': roots['PF']},
                   stdout=log, stderr=subprocess.STDOUT, check=True)
for variant in ('A', 'PF'):
    run(variant, 'final-warm-' + variant, [2, 79, 80, 2, 79, 80, 2, 79, 80])
for variant in ('AP', 'PFP'):
    label = 'final-profile-' + variant
    subprocess.run([sys.executable, '-u', str(P / 'run.py'), variant, label,
        '--surahs', '2', '79', '80'], check=True,
        env={**os.environ, 'FAST_PROFILE_DIR': str(P / (label + '-calls'))})
(P / 'PF-final-costs-complete.json').write_text(json.dumps({'serial': True, 'warm_repeats': 3}) + '\n')
