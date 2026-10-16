"""Serial warm executor costs and separately profiled model forward costs."""
import ast
import json
from pathlib import Path
import pstats

import numpy as np

P = Path(__file__).resolve().parent


def load(name):
    return json.loads((P / name).read_text())


def signature(case):
    return [(a, ms, q > 0) for a, ms, q in case['rows']]


before, after = load('final-warm-A.json'), load('final-warm-PF.json')
assert len(before) == len(after) == 9
rows = []
for index, (a, b) in enumerate(zip(before, after, strict=True)):
    assert a['surah'] == b['surah']
    for key in ('audio_sha256', 'initial_pauses_sha256'):
        assert a[key] == b[key]
    rows.append({'surah': a['surah'], 'repeat': index // 3 + 1,
                 'before_seconds': a['search_seconds'], 'after_seconds': b['search_seconds'],
                 'ratio': b['search_seconds'] / a['search_seconds'],
                 'before_decode_seconds': a['decode_seconds'], 'after_decode_seconds': b['decode_seconds'],
                 'before_dsp_seconds': a['dsp_seconds'], 'after_dsp_seconds': b['dsp_seconds'],
                 'before_inference_seconds': a['trace']['inference_seconds'],
                 'after_inference_seconds': b['trace']['inference_seconds'],
                 'before_calls': len(a['actual_dispatches']), 'after_calls': len(b['actual_dispatches']),
                 'before_input_ms': a['trace']['audio_ms'], 'after_input_ms': b['trace']['audio_ms'],
                 'before_union_ms': a['trace']['covered_audio_ms'], 'after_union_ms': b['trace']['covered_audio_ms']})
for label, cases in [('A', before), ('PF', after)]:
    for surah in (2, 79, 80):
        repeated = [c for c in cases if c['surah'] == surah]
        assert all(signature(c) == signature(repeated[0]) for c in repeated)
        assert all(c['actual_dispatches'] == repeated[0]['actual_dispatches'] for c in repeated)
    canonical = {c['surah']: c for c in load('A-wide.json' if label == 'A' else 'PF-wide-replay.json')}
    assert all(signature(c) == signature(canonical[c['surah']]) for c in cases)

source = Path('/Users/ayub/.surah-cutter/model-runtime/lib/python3.11/site-packages/transformers/models/whisper/modeling_whisper.py')
classes = ast.parse(source.read_text())
forward_lines = {}
for cls in classes.body:
    if isinstance(cls, ast.ClassDef) and cls.name in ('WhisperEncoder', 'WhisperDecoder'):
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        forward_lines[cls.name] = (min([method.lineno] + [d.lineno for d in method.decorator_list]), method.lineno)
profiles = {}
for variant in ('AP', 'PFP'):
    files = sorted((P / ('final-profile-' + variant + '-calls')).glob('*.prof'))
    cases = load('final-profile-' + variant + '.json')
    warm = before if variant == 'AP' else after
    assert all(signature(c) == signature(next(w for w in warm if w['surah'] == c['surah'])) for c in cases)
    assert len(files) == sum(len(c['actual_dispatches']) for c in cases)
    seconds = {'encoder': 0., 'decoder': 0., 'total_local_read': 0.}
    calls = {'encoder': 0, 'decoder': 0}
    for path in files:
        for (filename, line, name), values in pstats.Stats(str(path)).stats.items():
            if name == 'whisper_local' and filename.endswith('/model_worker.py'):
                seconds['total_local_read'] += values[3]
            if name != 'forward' or not filename.endswith('/whisper/modeling_whisper.py'):
                continue
            for cls_name, (left, right) in forward_lines.items():
                if left <= line <= right:
                    key = 'encoder' if cls_name == 'WhisperEncoder' else 'decoder'
                    seconds[key] += values[3]
                    calls[key] += values[1]
    assert seconds['encoder'] > 0 and seconds['decoder'] > 0
    assert seconds['total_local_read'] >= seconds['encoder'] + seconds['decoder']
    seconds['other'] = seconds['total_local_read'] - seconds['encoder'] - seconds['decoder']
    profiles[variant] = {'local_reads': len(files), 'seconds': seconds, 'forward_calls': calls,
                         'measurement': 'separate cProfile run; not warm wall-time benchmark'}

ratios = [r['ratio'] for r in rows]
cost = {'serial_own_model_jobs': True, 'warm_repeats': 3, 'per_case': rows,
        'fresh_warm_and_profile_owners_times_match_full_replay': True,
        'warm_repeated_exact_PCM_dispatches_stable': True,
        'median_ratio': float(np.median(ratios)), 'p95_ratio': float(np.percentile(ratios, 95)),
        'proposed_soft_cost_target': {'median_max': 1.25, 'p95_max': 2.,
                                    'passed': np.median(ratios) <= 1.25 and np.percentile(ratios, 95) <= 2.},
        'process': {v: load('final-warm-' + v + '-process.json') for v in ('A', 'PF')},
        'cold_load_probe_seconds': {v: load('final-warm-' + v + '.summary.json')['summary']['whisper-tarteel-base']['load_probe_seconds']
                                    for v in ('A', 'PF')},
        'profiles': profiles, 'profiles_source_forward_lines': forward_lines,
        'host_caveat': 'other user applications can contend; host CPU snapshots retained, no speedup claim'}
# numpy.bool_ is deliberately converted for strict JSON portability.
cost['proposed_soft_cost_target']['passed'] = bool(cost['proposed_soft_cost_target']['passed'])
(P / 'PF-costs.json').write_text(json.dumps(cost, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: cost[k] for k in ('median_ratio', 'p95_ratio', 'proposed_soft_cost_target', 'profiles')}, indent=2))
