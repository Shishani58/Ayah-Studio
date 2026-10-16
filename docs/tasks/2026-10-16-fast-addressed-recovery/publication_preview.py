"""Interim publication projection; final acceptance requires full exact-PCM replay."""
import hashlib
import json
from pathlib import Path
import sqlite3

from surah_cutter.core.fast_executor import Boundary, Policy, _review_addressed_recovery_onsets

P = Path(__file__).resolve().parent
baseline = {x['surah']: x for x in json.loads((P / 'A-wide.json').read_text())}
cases = json.loads((P / 'BS-wide.json').read_text())
reference = P.parents[2] / 'data/references/abu_bakr_ash_shatri.db'
before_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
results = []
with sqlite3.connect(f'file:{reference.resolve()}?mode=ro', uri=True) as conn:
    for case in cases:
        trace = case['trace']
        boundaries, reviews = _review_addressed_recovery_onsets(
            [Boundary(**b) for b in trace['boundaries']], trace['pause_geometry'], Policy())
        truth = dict(conn.execute('SELECT ayah,time FROM timings WHERE sura=?', (case['surah'],)))
        gains, losses, changed, bad = [], [], [], []
        for old, new in zip(baseline[case['surah']]['rows'], boundaries, strict=True):
            if old[2] == 0 and new.confidence > 0:
                gains.append(new.ayah)
                if abs(new.start_ms - truth[new.ayah]) > 500:
                    bad.append([new.ayah, abs(new.start_ms - truth[new.ayah])])
            if old[2] > 0 and new.confidence == 0:
                losses.append(new.ayah)
            if old[2] > 0 and old[1] != new.start_ms:
                changed.append(new.ayah)
        results.append({'surah': case['surah'], 'gains': gains, 'losses': losses,
                        'changed': changed, 'bad': bad, 'reviewed': [r['ayah'] for r in reviews]})
assert before_hash == hashlib.sha256(reference.read_bytes()).hexdigest()
output = {'projection_only': True, 'records': len(cases), 'cases': results}
(P / 'BS2-publication-preview.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print('records', len(cases), 'gain', sum(len(r['gains']) for r in results),
      'exceptions', [r for r in results if r['losses'] or r['changed'] or r['bad']])
