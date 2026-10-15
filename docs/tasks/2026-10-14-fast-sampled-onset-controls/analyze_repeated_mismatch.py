"""Offline hypothesis audit: repeated decoder mismatch versus third-read proofs."""
import json
from pathlib import Path
import re

task = Path(__file__).parent.resolve()
root = task.parents[2]
inputs = [root / 'docs/tasks/2026-10-14-fast-base-window-analysis/next_pause.json',
          root / 'docs/tasks/2026-10-14-fast-pronunciation-variants/sampled-onset-corpus.json']
findings = []
for source in inputs:
    for case in json.loads(source.read_text()):
        trace = case['trace']
        plans = {e['ayah']: e['phrase'] for e in trace['transitions']
                 if e['state'] == 'PLAN_PHRASE' and 'phrase' in e}
        grouped = {}
        threshold = .25 if trace['backend'] == 'ctc_greedy_probability' else .6
        proof_reads = {b['read_id']: b['ayah'] for b in trace['boundaries'] if b.get('read_id') is not None}
        for read in trace['acoustic_reads']:
            owner = re.match(r'ayah=(\d+):', read['reason'])
            if not owner or read.get('outcome') != 'completed':
                continue
            ayah = int(owner[1])
            if ':pause:' not in read['reason'] and ':first_onset:' not in read['reason']:
                continue
            grouped.setdefault((ayah, read['window_ms'][0]), []).append(read)
        for (ayah, start), reads in grouped.items():
            if len(reads) < 3:
                continue
            phrase = plans[ayah]
            pairs = []
            for read in reads[:2]:
                observed = read['observed']
                units = observed['units']
                matched = 0
                while matched < min(len(units), len(phrase)) and units[matched] == phrase[matched]:
                    matched += 1
                if not (0 < matched < len(phrase) and matched < len(units)):
                    break
                if min(observed['unit_quality'][:matched + 1], default=0) < threshold:
                    break
                pairs.append((matched, units[matched]))
            if len(pairs) != 2 or pairs[0] != pairs[1]:
                continue
            findings.append({'source': source.name, 'model': case['model'], 'surah': case['surah'],
                             'ayah': ayah, 'start': start, 'phrase': phrase,
                             'repeated_mismatch': pairs[0],
                             'reads': [{'id': r['id'], 'window': r['window_ms'],
                                        'observed': r['observed']['units'],
                                        'quality': r['observed']['unit_quality'],
                                        'used_by_proof': proof_reads.get(r['id'])}
                                       for r in reads]})
result = {'inputs': [str(p) for p in inputs], 'candidates': findings,
          'third_or_later_used_proof': [f for f in findings if any(r['used_by_proof'] is not None
                                                       for r in f['reads'][2:])]}
(task / 'repeated-mismatch-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('candidates', len(findings), 'later proof cases', len(result['third_or_later_used_proof']))
for case in result['third_or_later_used_proof']:
    print(case['source'], case['model'], case['surah'], case['ayah'], case['repeated_mismatch'])
