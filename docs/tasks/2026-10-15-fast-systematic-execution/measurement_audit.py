"""Summarize observed cost/coverage, check immutable data and source guards."""
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
REPO = P.parents[2]


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def describe(errors):
    return {'n': len(errors), 'median': float(np.median(errors)) if errors else None,
            'p95': float(np.percentile(errors, 95)) if errors else None,
            'max': max(errors, default=None), 'over500': sum(e > 500 for e in errors)}


def main():
    results = {}
    for label in ['baseline-a2', 'after-a', 'baseline-e', 'after-e', 'repeat-baseline', 'repeat-prototype']:
        if not (P / f'{label}.summary.json').exists():
            continue
        cases = json.loads((P / f'{label}.json').read_text())
        models = {}
        for model in dict.fromkeys(c['model'] for c in cases):
            subset = [c for c in cases if c['model'] == model]
            confirmed = [e for c in subset for r, e in zip(c['rows'], c['errors_ms']) if r[2] > 0]
            all_errors = [e for c in subset for e in c['errors_ms']]
            models[model] = {'confirmed_reference_errors_ms': describe(confirmed),
                'all_reference_errors_ms': describe(all_errors),
                'audio_ms': sum(c['trace']['audio_ms'] for c in subset),
                'union_ms': sum(c['trace']['covered_audio_ms'] for c in subset),
                'duration_ms': sum(c['duration_ms'] for c in subset),
                'calls': sum(len(c['actual_dispatches']) for c in subset),
                'inference_s': sum(c['trace']['inference_seconds'] for c in subset),
                'executor_s': sum(c['seconds'] for c in subset),
                'cache_peak_bytes': max(c['trace']['peak_cache_bytes'] for c in subset),
                'cache_hits': sum(c['trace']['cache_hits'] for c in subset),
                'max_union_ratio': max(c['trace']['coverage'] for c in subset),
                'max_input_ratio': max(c['trace']['audio_ms'] / c['duration_ms'] for c in subset),
                'risk_bins': [{'quality_min': lower, 'quality_max': upper,
                    'confirmed': sum(lower <= r[2] <= upper for c in subset for r in c['rows'] if r[2] > 0),
                    'reference_outlier': sum(lower <= r[2] <= upper and e > 500 for c in subset
                         for r, e in zip(c['rows'], c['errors_ms']) if r[2] > 0)}
                    for lower, upper in [(0, .599999), (.6, .899999), (.9, 1.)]]}
            if label.startswith('repeat'):
                models[model]['warm'] = {str(s): {
                    'executor_median_s': float(np.median([c['seconds'] for c in subset if c['surah'] == s])),
                    'inference_median_s': float(np.median([c['trace']['inference_seconds'] for c in subset if c['surah'] == s])),
                    'all_rows_identical': len({json.dumps(c['rows']) for c in subset if c['surah'] == s}) == 1,
                    'all_dispatches_identical': len({json.dumps(c['actual_dispatches']) for c in subset if c['surah'] == s}) == 1}
                    for s in [112, 114]}
        process = json.loads((P / f'{label}-process.json').read_text())
        results[label] = {'models': models, 'tree_peak_MiB': process['tree_peak_kib'] / 1024,
                          'tree_sampling_s': process['sampling_s'], 'wall_s': process['wall_s']}
    initial = json.loads((P / 'source-manifests.json').read_text())['baseline']
    guarded = {name: {'initial': digest, 'current': sha(REPO / name)}
               for name, digest in initial.items() if name.startswith('surah_cutter/') or name.startswith('tools/')}
    changed = {name: d for name, d in guarded.items() if d['initial'] != d['current']}
    data = json.loads((P / 'data-manifest.json').read_text())
    changed_data = {name: {'initial': item['sha256'], 'current': sha(Path(name))}
                    for name, item in data.items() if sha(Path(name)) != item['sha256']}
    (P / 'measurement-audit.json').write_text(json.dumps({'results': results,
        'production_source_changes_since_initial': changed, 'data_files_guarded': len(data),
        'data_changes': changed_data, 'agents_sync': (REPO / 'AGENTS.md').read_bytes() == (REPO / 'CLAUDE.md').read_bytes(),
        'physical_label_evaluation': None,
        'notes': ['Risk bins are model score strata, not calibrated error probabilities.',
                  'Non-outlier reference timing does not establish correct physical owner/onset.',
                  'Tree RSS is concurrent sum sampled at200ms; short peaks may be missed.']},
        ensure_ascii=False, indent=2) + '\n')
    print('source drift', list(changed), 'data drift', len(changed_data))
    print({label: {m: s['confirmed_reference_errors_ms']['n'] for m, s in r['models'].items()}
           for label, r in results.items()})


if __name__ == '__main__':
    main()
