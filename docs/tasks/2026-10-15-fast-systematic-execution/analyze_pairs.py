"""Independent per-row paired summaries, not a test suite or inference."""
import csv
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent


def load(label):
    assert (P / f'{label}.summary.json').exists(), f'incomplete {label}'
    return json.loads((P / f'{label}.json').read_text())


def compare(before, after):
    b = {(c['model'], c['surah']): c for c in load(before)}
    pairs = []
    for c in load(after):
        old = b[(c['model'], c['surah'])]
        old_rows = {r[0]: r for r in old['rows']}
        added, lost, changed = [], [], []
        for row, error in zip(c['rows'], c['errors_ms']):
            previous = old_rows[row[0]]
            if row[2] > 0 and previous[2] == 0:
                added.append({'ayah': row[0], 'row': row, 'reference_error_ms': error})
            if previous[2] > 0 and row[2] == 0:
                lost.append(previous)
            if previous[2] > 0 and previous != row:
                changed.append({'before': previous, 'after': row})
        events = c['trace']['transitions']
        direct = [e['ayah'] for e in events if e['state'] == 'SHORT_CONTEXT_RECHECK']
        pairs.append({'model': c['model'], 'surah': c['surah'], 'before_label': before,
            'after_label': after, 'added': added, 'lost': lost, 'changed_previous': changed,
            'direct_short_proofs': [a['ayah'] for a in added if a['ayah'] in direct],
            'downstream_added': [a['ayah'] for a in added if a['ayah'] not in direct],
            'rows_identical': c['rows'] == old['rows'],
            'dispatches_identical': c['actual_dispatches'] == old['actual_dispatches'],
            'calls': [len(old['actual_dispatches']), len(c['actual_dispatches'])],
            'input_ms': [old['trace']['audio_ms'], c['trace']['audio_ms']],
            'union_ms': [old['trace']['covered_audio_ms'], c['trace']['covered_audio_ms']],
            'sequence_violations': c['sequence_violations'],
            'confirmed': [sum(r[2] > 0 for r in old['rows']), sum(r[2] > 0 for r in c['rows'])]})
    return pairs


def main():
    current = load('baseline-a2')
    prior = json.loads((P.parent / '2026-10-15-fast-next-iteration/after-raw.json').read_text())
    lookup = {(c['model'], c['surah']): c for c in prior}
    parity = [{'model': c['model'], 'surah': c['surah'],
        'rows_identical': c['rows'] == lookup[(c['model'], c['surah'])]['rows'],
        'dispatches_identical': c['actual_dispatches'] == lookup[(c['model'], c['surah'])]['actual_dispatches']}
        for c in current]
    (P / 'historical-parity.json').write_text(json.dumps(parity, indent=2))
    flat = []
    for c in current:
        conf_errors = [e for r, e in zip(c['rows'], c['errors_ms']) if r[2] > 0]
        reads = c['trace']['acoustic_reads']
        flat.append({'model': c['model'], 'surah': c['surah'], 'ayahs': len(c['rows']),
            'confirmed': len(conf_errors), 'review': sum(r[2] == 0 for r in c['rows']),
            'confirmed_median_ms': float(np.median(conf_errors)) if conf_errors else None,
            'confirmed_p95_ms': float(np.percentile(conf_errors, 95)) if conf_errors else None,
            'confirmed_max_ms': max(conf_errors, default=None),
            'all_median_ms': c['median_ms'], 'all_p95_ms': c['p95_ms'], 'all_max_ms': c['max_ms'],
            'confirmed_over500': sum(e > 500 for e in conf_errors),
            'calls': len(reads), 'input_ms': c['trace']['audio_ms'],
            'union_ms': c['trace']['covered_audio_ms'], 'coverage': c['trace']['coverage'],
            'input_ratio': c['trace']['audio_ms'] / c['duration_ms'],
            'cache_hits': c['trace']['cache_hits'], 'peak_cache_bytes': c['trace']['peak_cache_bytes'],
            'inference_s': c['trace']['inference_seconds'],
            'executor_s': c['seconds'],
            'matcher_other_s': c['seconds'] - c['trace']['inference_seconds'],
            'denied': len(c['trace']['denied_acoustic_reads'])})
    with (P / 'baseline.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=flat[0].keys())
        w.writeheader()
        w.writerows(flat)
    (P / 'baseline.json').write_text(json.dumps(flat, ensure_ascii=False, indent=2))
    pairs = []
    for before, after in [('baseline-a2', 'after-a'), ('baseline-e', 'after-e')] + [
        (f'baseline-{n}', f'prototype-{n}') for n in ['saad_al_ghamidi',
        'mohammed_siddiq_al_minshawi', 'mahmoud_khalil_al_hussary', 'abdul_basit_mujawwad']]:
        if (P / f'{after}.summary.json').exists():
            pairs.extend(compare(before, after))
    (P / 'comparison.json').write_text(json.dumps(pairs, ensure_ascii=False, indent=2))
    print('historical exact', sum(c['rows_identical'] and c['dispatches_identical'] for c in parity), '/', len(parity))
    for model in dict.fromkeys(c['model'] for c in pairs):
        subset = [c for c in pairs if c['model'] == model]
        print(model, 'pairs', len(subset), 'added', sum(len(c['added']) for c in subset),
              'lost', sum(len(c['lost']) for c in subset),
              'changed', sum(len(c['changed_previous']) for c in subset))


if __name__ == '__main__':
    main()
