"""Historical 37-record comparison; accepted legacy marks are not strict proofs."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
decode_file = 'surah_cutter/core/decode.py'
assert hashlib.sha256((Path(roots['legacy']) / decode_file).read_bytes()).digest() == hashlib.sha256(
    (Path(roots['A']) / decode_file).read_bytes()).digest()
old = json.loads((P / 'legacy37.json').read_text())
baseline = {c['surah']: c for c in json.loads((P / 'A-wide.json').read_text())}
current = {c['surah']: c for c in json.loads((P / 'PF-wide-replay.json').read_text())}
records = []
for legacy in old:
    a, b = baseline[legacy['surah']], current[legacy['surah']]
    for field in ('audio_sha256', 'initial_pauses_sha256', 'reference_sha256'):
        assert legacy[field] == a[field] == b[field]
    assert legacy['runtime_identity'] == a['trace']['run_snapshot']['text_runtime'] == b['trace']['run_snapshot']['text_runtime']
    records.append({'surah': legacy['surah'], 'ayahs': len(legacy['rows']),
        **{name: {'positive_marks': sum(r[2] > 0 for r in c['rows']),
            'positive_over500': sum(r[2] > 0 and e > 500 for r, e in zip(c['rows'], c['errors_ms'], strict=True)),
            'all_over500': sum(e > 500 for e in c['errors_ms']),
            'calls': len(c['actual_dispatches']), 'input_ms': c['trace']['audio_ms']}
           for name, c in [('legacy', legacy), ('F13_baseline', a), ('PF14', b)]}})
summary = {name: {key: sum(r[name][key] for r in records)
                 for key in records[0][name]} for name in ('legacy', 'F13_baseline', 'PF14')}
output = {'records': len(records), 'ayahs': sum(r['ayahs'] for r in records), 'summary': summary,
    'comparison_limit': 'legacy accepted marks and current independently validated proofs have different semantics; this does not redefine historical M5',
    'matched': ['audio SHA', 'initial DSP decisions SHA', 'reference SHA', 'actual installed model assets/config/runtime identity', 'decoder source'],
    'records_detail': records}
(P / 'PF-legacy37-comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, indent=2))
