"""Export each added proof with its actual cue/read and post-hoc reference error."""
import json
from pathlib import Path

from causal_evidence import cue_evidence

P = Path(__file__).resolve().parent
GATE = json.loads((P / 'final-gate.json').read_text())
assert GATE['passed']
cache = {}
def cases(label):
    if label not in cache:
        cache[label] = {(c['model'], c['surah']): c for c in json.loads((P / (label + '.json')).read_text())}
    return cache[label]

out = []
for result in GATE['results']:
    if not result['added']:
        continue
    key = result['model'], result['surah']
    case, prior = cases(result['after'])[key], cases(result['before'])[key]
    for added in result['added']:
        ayah = added['ayah']
        boundary = next(b for b in case['trace']['boundaries'] if b['ayah'] == ayah)
        evidence = cue_evidence(case, prior, ayah)
        proof_read, observations = evidence['proof_read'], evidence['cue_reads']
        out.append({'before': result['before'], 'after': result['after'], 'model': key[0],
            'surah': key[1], 'ayah': ayah, 'start_ms': boundary['start_ms'],
            'reference_error_ms': added['reference_error_ms'],
            'kind': 'direct_same_window_extension' if evidence['direct'] else 'downstream_or_recovery',
            'cue_reads': observations, 'proof_read_id': boundary['read_id'],
            'proof_window_ms': proof_read['window_ms'], 'proof_observed': proof_read['observed']['units'],
            'proof_quality': proof_read['observed']['unit_quality'], 'proof': boundary['proof'],
            'case_calls_before_after': result['calls'], 'case_pcm_ms_before_after': result['pcm_ms']})
(P / 'causal-gains.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print('added', len(out), 'direct', sum(r['kind'] == 'direct_same_window_extension' for r in out))
for r in out:
    print({k: r[k] for k in ['after', 'model', 'surah', 'ayah', 'start_ms', 'reference_error_ms', 'kind']})
