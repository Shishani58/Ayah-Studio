"""Reproduce the partial-gate analysis, including the production letter aliases."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sys.path.insert(0, str(ROOT))
from surah_cutter.core.fast_adapters import arabic_units  # noqa: E402
from surah_cutter.core.fast_executor import _strong_continuation_conflict  # noqa: E402
from surah_cutter.core.model_worker import whisper_alignment_text  # noqa: E402


def checks(h, case):
    units, quality = [], []
    if len(h['attribution']) != len(h['units']):
        return False
    for word, q, attr in zip(h['units'], h['unit_quality'], h['attribution']):
        names = arabic_units(whisper_alignment_text(attr.get('raw_word', word)))
        units.extend(names)
        quality.extend([q] * len(names))
    phrase = tuple(case['phrase'])
    return (tuple(units[:len(phrase)]) == phrase and min(quality[:len(phrase)], default=0.) >= .6
            and not h['truncated'] and not _strong_continuation_conflict(
                units, quality, case['expected'], len(phrase), .6))


summaries = {}
for prefix in ['deep', 'geometry', 'recovery']:
    manifest = json.loads((P / f'{prefix}-inputs.json').read_text())
    results = json.loads((P / f'{prefix}-results.json').read_text())
    assert len(manifest['cases']) == len(results)
    cases = {c['key']: c for c in manifest['cases']}
    groups, changed, extra, positives = {}, [], [], []
    for r in results:
        case = cases[r['key']]
        g = checks(r['readings']['1'][0], case)
        b = any(checks(h, case) for h in r['readings']['5'])
        previous_g = r['readings']['1'][0]['necessary_opening_checks']
        previous_b = any(h['necessary_opening_checks'] for h in r['readings']['5'])
        if (g, b) != (previous_g, previous_b):
            changed.append(r['key'])
        group = groups.setdefault(case['role'], dict(n=0, greedy=0, any_beam=0))
        group['n'] += 1
        group['greedy'] += g
        group['any_beam'] += b
        if b and not g:
            extra.append(r['key'])
        if g:
            positives.append(r['key'])
    summaries[prefix] = dict(groups=groups, alias_expansion_changed_results=changed,
        beam_additional_necessary_pass=extra, greedy_necessary_pass=positives,
        median_decoder_s={k: statistics.median(r['decoder_seconds'][k] for r in results) for k in ['1', '5']})
baseline = ROOT / 'docs/tasks/2026-10-15-fast-systematic-execution/baseline-a2.json'
rows = json.loads(baseline.read_text())
reasons = {}
for row in rows:
    if row['model'] != 'whisper-tarteel-base':
        continue
    for b in row['trace']['boundaries']:
        if b['confidence'] == 0:
            reason = b['review_reason']
            reasons[reason] = reasons.get(reason, 0) + 1
summaries['historical_unconfirmed_reasons'] = reasons
summaries['baseline_sha256'] = hashlib.sha256(baseline.read_bytes()).hexdigest()
summaries['scope'] = 'Necessary lexical/quality checks only: not production proof, temporal accuracy or representative rate'
summaries['source_sha256'] = {name: hashlib.sha256((P / name).read_bytes()).hexdigest()
    for name in ['deep_probe.py', 'geometry_prepare.py', 'recovery_prepare.py', 'analyze_deep.py']}
(P / 'deep-analysis.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summaries, ensure_ascii=False, indent=2))
