"""Read-only analysis of completed sparse12 artifacts; no inference or tests."""
import collections
import hashlib
import json
import math
from pathlib import Path
from surah_cutter.core.fast_executor import Policy

p = Path(__file__).parent
prior = p.parent / '2026-10-15-fast-next-iteration'
cases = json.loads((prior / 'after-raw.json').read_text())
assert len(cases) == 48 and (prior / 'after-raw.summary.json').exists()
out = {'artifact_source': str(prior / 'after-raw.json'),
       'source_sha256': hashlib.sha256((prior / 'after-raw.json').read_bytes()).hexdigest(),
       'policy': vars(Policy()), 'models': [], 'local_resources': []}
for model in dict.fromkeys(c['model'] for c in cases):
    subset = [c for c in cases if c['model'] == model]
    assert len(subset) == 12
    rows = []
    for c in subset:
        t = c['trace']
        stop = next((x for x in t['transitions']
                     if x['state'] in ('RESOLVE_BLOCK', 'REVIEW_REQUIRED')), None)
        rejections = [r for read in t['acoustic_reads']
                      for r in read.get('prefix_rejections', [])]
        rows.append({'surah': c['surah'], 'ayahs': len(c['rows']),
                     'confirmed': sum(r[2] > 0 for r in c['rows']),
                     'first_stop': stop, 'calls': len(t['acoustic_reads']),
                     'prefix_rejection_count': len(rejections),
                     'max_rejected_target_probability': max((math.exp(r['expected_log_probability'])
                         for r in rejections if r.get('expected_log_probability') is not None), default=None),
                     'denied_calls': len(t['denied_acoustic_reads'])})
    out['models'].append({'model': model, 'rows': rows,
        'confirmed': sum(r['confirmed'] for r in rows),
        'first_stop_ayah_1_cases': sum(r['first_stop'] is not None and r['first_stop']['ayah'] == 1 for r in rows),
        'zero_confirmed_cases': sum(r['confirmed'] == 0 for r in rows),
        'first_stop_reasons': dict(collections.Counter(r['first_stop']['reason']
            if r['first_stop'] else 'all_resolved' for r in rows)),
        'calls': sum(r['calls'] for r in rows),
        'prefix_rejection_count': sum(r['prefix_rejection_count'] for r in rows),
        'denied_calls': sum(r['denied_calls'] for r in rows)})
cache = Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit')
for name in ['saad_al_ghamidi', 'mohammed_siddiq_al_minshawi',
             'mahmoud_khalil_al_hussary', 'abdul_basit_mujawwad']:
    folder = cache / name
    files = [folder / f'{s:03}.mp3' for s in [80, 98, 112]] + [folder / f'{name}.db']
    out['local_resources'].append({'name': name, 'surahs': [80, 98, 112],
        'files': [{'path': str(f), 'exists': f.exists(),
                   'sha256': hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None}
                  for f in files]})
paths = [Path('surah_cutter/core') / f for f in ['fast_executor.py', 'fast_adapters.py',
    'fast_ctc.py', 'fast_reads.py', 'candidates.py', 'emissions.py']]
paths += [Path('surah_cutter/data/transliteration.py'), Path('tools/model_sources.json'),
          Path('tools/bench_fast_models.py'), Path('AGENTS.md'), Path('CLAUDE.md')]
paths += list(Path('data').rglob('*.db'))
out['protected_source_data_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
(p / 'research-data.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print([{k:v for k,v in m.items() if k != 'rows'} for m in out['models']])
print('local recordings present', sum(all(f['exists'] for f in r['files']) for r in out['local_resources']))
