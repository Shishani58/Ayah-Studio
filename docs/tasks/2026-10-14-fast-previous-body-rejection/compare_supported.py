"""Late paired comparison with preservation, new error and dispatch gates."""
import json
from pathlib import Path

task = Path(__file__).parent.resolve()
reports = []
for model in ['quran-phonetics', 'quran-phonetics-fp32', 'whisper-tarteel-tiny', 'whisper-tarteel-base']:
    before, after = [json.loads((task / f'{model}-{v}.json').read_text()) for v in ['before', 'after']]
    assert len(before) == len(after) == 12
    report = {'model': model, 'lost': [], 'added': [], 'changed_rows': [], 'new_outliers': [],
              'cases': [], 'before': {}, 'after': {}, 'declines': []}
    for label, cases in [('before', before), ('after', after)]:
        report[label] = {'rows': sum(len(c['rows']) for c in cases),
                         'proofs': sum(sum(r[2] > 0 for r in c['rows']) for c in cases),
                         'calls': sum(len(c['actual_dispatches']) for c in cases),
                         'search_seconds': sum(c['search_seconds'] for c in cases)}
    for a, b in zip(before, after):
        assert a['model'] == b['model'] == model and a['surah'] == b['surah']
        for key in ['audio_sha256', 'reference_sha256', 'initial_pauses_sha256']:
            assert a[key] == b[key]
        for ar, br, ae, be in zip(a['rows'], b['rows'], a['errors_ms'], b['errors_ms']):
            assert ar[0] == br[0]
            if ar[2] > 0 and br[2] == 0:
                report['lost'].append([a['surah'], ar])
            if ar[2] == 0 and br[2] > 0:
                report['added'].append([a['surah'], br, be])
            if ar != br:
                report['changed_rows'].append([a['surah'], ar, br])
            if br[2] > 0 and be > 500 and (ar[2] == 0 or ae <= 500):
                report['new_outliers'].append([a['surah'], br, be])
        trace = b['trace']
        max_read = max((r['samples'] / 16 for r in b['actual_dispatches']), default=0)
        ratio = trace['audio_ms'] / b['duration_ms']
        assert max_read <= 6000.001 and trace['coverage'] <= .650001 and ratio <= 2.000001
        assert not b['sequence_violations']
        report['cases'].append({'surah': a['surah'], 'same_dispatches': a['actual_dispatches'] == b['actual_dispatches'],
                                'coverage': trace['coverage'], 'max_read_ms': max_read, 'input_ratio': ratio})
        report['declines'].extend([{'surah': a['surah'], **e} for e in trace['transitions']
                                   if e['state'] == 'DECLINE_CANDIDATE'])
    reports.append(report)
(task / 'supported-comparison.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
for r in reports:
    print(r['model'], 'proofs', r['before']['proofs'], r['after']['proofs'],
          'lost', len(r['lost']), 'new errors', len(r['new_outliers']),
          'changed rows', len(r['changed_rows']), 'declines', len(r['declines']),
          'search', round(r['before']['search_seconds'], 2), round(r['after']['search_seconds'], 2))
