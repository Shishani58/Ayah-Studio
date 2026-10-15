"""Late reference comparison of paired supported models and actual dispatches."""
import json
from pathlib import Path

task = Path(__file__).parent.resolve()
reports = []
for model in ['quran-phonetics', 'whisper-tarteel-tiny', 'whisper-tarteel-base']:
    paths = [task / f'{model}-{variant}.json' for variant in ['before', 'after']]
    if not all(p.exists() and p.with_suffix('.summary.json').exists() for p in paths):
        continue
    before, after = [json.loads(p.read_text()) for p in paths]
    assert len(before) == len(after) == 11
    report = {'model': model, 'cases': [], 'lost': [], 'added': [], 'changed_rows': [],
              'new_confident_outliers': [], 'before': {}, 'after': {}}
    for label, cases in [('before', before), ('after', after)]:
        report[label] = {
            'rows': sum(len(c['rows']) for c in cases),
            'proofs': sum(sum(r[2] > 0 for r in c['rows']) for c in cases),
            'calls': sum(len(c['actual_dispatches']) for c in cases),
            'search_seconds': sum(c['search_seconds'] for c in cases),
            'inference_seconds': sum(c['trace']['inference_seconds'] for c in cases),
        }
    for a, b in zip(before, after):
        assert a['surah'] == b['surah'] and a['model'] == b['model'] == model
        assert a['audio_sha256'] == b['audio_sha256']
        assert a['reference_sha256'] == b['reference_sha256']
        assert a['initial_pauses_sha256'] == b['initial_pauses_sha256']
        s = a['surah']
        old = {r[0]: r for r in a['rows']}
        new = {r[0]: r for r in b['rows']}
        assert old.keys() == new.keys()
        same_dispatches = a['actual_dispatches'] == b['actual_dispatches']
        for ayah in old:
            ar, br = old[ayah], new[ayah]
            if ar != br:
                report['changed_rows'].append([s, ar, br])
            if ar[2] > 0 and br[2] == 0:
                report['lost'].append([s, ayah])
            if ar[2] == 0 and br[2] > 0:
                report['added'].append([s, ayah])
        for ar, br, ae, be in zip(a['rows'], b['rows'], a['errors_ms'], b['errors_ms']):
            if br[2] > 0 and be > 500 and (ar[2] == 0 or ae <= 500):
                report['new_confident_outliers'].append([s, br, be])
        trace = b['trace']
        assert not b['sequence_violations']
        max_read = max((r['samples'] / 16 for r in b['actual_dispatches']), default=0)
        ratio = trace['audio_ms'] / b['duration_ms']
        assert max_read <= 6000.001 and trace['coverage'] <= .650001 and ratio <= 2.000001
        report['cases'].append({'surah': s, 'same_dispatches': same_dispatches,
                                'coverage': trace['coverage'], 'max_read_ms': max_read,
                                'input_ratio': ratio})
    reports.append(report)
(task / 'paired-comparison.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
for report in reports:
    print(report['model'], 'proofs', report['before']['proofs'], report['after']['proofs'],
          'changed', len(report['changed_rows']), 'lost', len(report['lost']),
          'new outliers', len(report['new_confident_outliers']),
          'same dispatches', all(c['same_dispatches'] for c in report['cases']),
          'search', round(report['before']['search_seconds'], 2), round(report['after']['search_seconds'], 2))
