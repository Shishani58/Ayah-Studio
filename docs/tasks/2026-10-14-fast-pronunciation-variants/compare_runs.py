"""Compare retained prospective runs; no inference or tests."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('old', type=Path)
parser.add_argument('new', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
old = {c['surah']: c for c in json.loads(args.old.read_text())}
new = json.loads(args.new.read_text())
result = {'inputs': [str(args.old.resolve()), str(args.new.resolve())],
          'cases': [], 'lost': [], 'added': [], 'confident_outliers': [],
          'totals': {}}
for label, data in [('old', [old[c['surah']] for c in new]), ('new', new)]:
    result['totals'][label] = {
        'cases': len(data), 'rows': sum(len(c['rows']) for c in data),
        'proofs': sum(sum(r[2] > 0 for r in c['rows']) for c in data),
        'calls': sum(len(c['actual_dispatches']) for c in data),
        'inference_seconds': sum(c['trace']['inference_seconds'] for c in data),
        'search_seconds': sum(c['search_seconds'] for c in data),
    }
for b in new:
    s = b['surah']
    a = old[s]
    assert a['audio_sha256'] == b['audio_sha256']
    assert a['reference_sha256'] == b['reference_sha256']
    ar = {r[0]: r for r in a['rows'] if r[2] > 0}
    br = {r[0]: r for r in b['rows'] if r[2] > 0}
    result['lost'].extend([[s, k, ar[k]] for k in sorted(ar.keys() - br.keys())])
    result['added'].extend([[s, k, br[k]] for k in sorted(br.keys() - ar.keys())])
    assert len(b['rows']) == len(b['errors_ms'])
    result['confident_outliers'].extend([[s, r, e] for r, e in zip(b['rows'], b['errors_ms'])
                                        if r[2] > 0 and e > 500])
    trace = b['trace']
    # Validate actual dispatch sizes, not just planned transition windows.
    max_read = max((d['samples'] / 16 for d in b['actual_dispatches']), default=0)
    ratio = trace['audio_ms'] / b['duration_ms']
    assert max_read <= 6000.001 and trace['coverage'] <= .650001 and ratio <= 2.000001
    assert len(b['actual_dispatches']) == trace['model_passes']
    assert not b['sequence_violations']
    result['cases'].append({'surah': s, 'old': len(ar), 'new': len(br),
                            'new_calls': len(b['actual_dispatches']), 'coverage': trace['coverage'],
                            'max_actual_read_ms': max_read, 'input_ratio': ratio})
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['added', 'inputs']},
                 ensure_ascii=False, indent=2))
print('added:', len(result['added']))
