"""Independent row/call audit. Does not call a model or execute test suites."""
import json
from pathlib import Path
import sys

from causal_evidence import cue_evidence

P = Path(__file__).resolve().parent
stage = sys.argv[1]
pairs = [('corpus-a-baseline', 'corpus-a-prototype-c'),
         ('corpus-e-baseline', 'corpus-e-prototype-c'),
         ('h9b-baseline-saad_al_ghamidi', 'h9c-prototype-saad_al_ghamidi'),
         ('h9b-baseline-mahmoud_khalil_al_hussary', 'h9c-prototype-mahmoud_khalil_al_hussary'),
         ('h9c-baseline-mohammed_siddiq_al_minshawi', 'h9c-prototype-mohammed_siddiq_al_minshawi')]
if stage == 'f-wide':
    pairs = [('wide-baseline', 'f-replay-predictions')]
elif stage == 'wide':
    pairs = [('wide-baseline', 'wide-prototype-c')]
elif stage == 'final':
    pairs = [('wide-baseline', 'f-replay-predictions')]
    pairs += [(f'h9b-baseline-{name}', f'final-f-{name}')
              for name in ['saad_al_ghamidi', 'mahmoud_khalil_al_hussary']]
    pairs.append(('h9c-baseline-mohammed_siddiq_al_minshawi',
                  'final-f-mohammed_siddiq_al_minshawi'))
results, incomplete = [], []
for before, after in pairs:
    if not all((P / f'{name}.summary.json').exists() for name in [before, after]):
        incomplete.append([before, after])
        continue
    old = {(c['model'], c['surah']): c for c in json.loads((P / f'{before}.json').read_text())}
    for case in json.loads((P / f'{after}.json').read_text()):
        if stage in ('final', 'f-wide') and case['model'] != 'whisper-tarteel-base':
            continue
        prior = old[(case['model'], case['surah'])]
        indexed = {r[0]: r for r in prior['rows']}
        added, lost, changed, outliers = [], [], [], []
        for row, error in zip(case['rows'], case['errors_ms']):
            previous = indexed[row[0]]
            if row[2] > 0 and previous[2] == 0:
                added.append({'ayah': row[0], 'row': row, 'reference_error_ms': error})
                if error > 500:
                    outliers.append({'ayah': row[0], 'reference_error_ms': error})
            if previous[2] > 0 and row[2] == 0:
                lost.append(previous)
            if previous[2] > 0 and previous != row:
                changed.append({'before': previous, 'after': row})
        direct_ayahs = {a['ayah'] for a in added if cue_evidence(case, prior, a['ayah'])['direct']}
        results.append({'before': before, 'after': after, 'model': case['model'],
            'surah': case['surah'], 'added': added, 'lost': lost, 'changed': changed,
            'new_over500': outliers, 'sequence_violations': case['sequence_violations'],
            'direct_added': [a['ayah'] for a in added if a['ayah'] in direct_ayahs],
            'downstream_added': [a['ayah'] for a in added if a['ayah'] not in direct_ayahs],
            'confirmed': [sum(r[2] > 0 for r in prior['rows']), sum(r[2] > 0 for r in case['rows'])],
            'calls': [len(prior['actual_dispatches']), len(case['actual_dispatches'])],
            'pcm_ms': [prior['trace']['audio_ms'], case['trace']['audio_ms']],
            'identical_rows': prior['rows'] == case['rows'],
            'identical_dispatches': prior['actual_dispatches'] == case['actual_dispatches'],
            'max_union_ratio': case['trace']['coverage'],
            'input_ratio': case['trace']['audio_ms'] / case['duration_ms']})
gate = {'passed': not incomplete and not any(r['lost'] or r['changed'] or r['new_over500']
        or r['sequence_violations'] for r in results), 'incomplete': incomplete,
        'pairs': len(results), 'added': sum(len(r['added']) for r in results),
        'lost': sum(len(r['lost']) for r in results),
        'new_over500': sum(len(r['new_over500']) for r in results),
        'results': results}
temporary = P / f'{stage}-gate.json.pending'
temporary.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + '\n')
temporary.replace(P / f'{stage}-gate.json')
print({k: v for k, v in gate.items() if k != 'results'}, flush=True)
for result in results:
    if result['added'] or result['lost'] or result['changed'] or result['new_over500']:
        print(result, flush=True)
if not gate['passed']:
    raise SystemExit(1)
