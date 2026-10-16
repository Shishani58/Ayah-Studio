"""Read reference only for completed predictions from a frozen candidate."""
import hashlib
import json
from pathlib import Path
import sqlite3

P = Path(__file__).resolve().parent
old = {c['surah']: c for c in json.loads((P / 'A-wide.json').read_text())}
new = json.loads((P / 'BS-wide.json').read_text())
reference = P.parents[2] / 'data/references/abu_bakr_ash_shatri.db'
digest = hashlib.sha256(reference.read_bytes()).hexdigest()
out = []
with sqlite3.connect(f'file:{reference.resolve()}?mode=ro', uri=True) as conn:
    for case in new:
        a = old[case['surah']]
        truth = dict(conn.execute('SELECT ayah,time FROM timings WHERE sura=?', (case['surah'],)))
        gained = [r[0] for r, before in zip(case['rows'], a['rows']) if r[2] > 0 and before[2] == 0]
        lost = [r[0] for r, before in zip(case['rows'], a['rows']) if r[2] == 0 and before[2] > 0]
        changed = [[before, r] for r, before in zip(case['rows'], a['rows']) if before[2] > 0 and r[1] != before[1]]
        bad = [[r[0], abs(r[1]-truth[r[0]])] for r in case['rows'] if r[0] in gained and abs(r[1]-truth[r[0]]) > 500]
        out.append({'surah': case['surah'], 'gained': gained, 'lost': lost,
                    'changed_old_times': changed, 'new_over500': bad})
assert digest == hashlib.sha256(reference.read_bytes()).hexdigest()
(P / 'BS-wide-checkpoint-evaluation.json').write_text(json.dumps({'predicted_records': len(new),
    'reference_sha256': digest, 'method': 'read-only evaluation after each completed recording; frozen source unchanged',
    'cases': out}, ensure_ascii=False, indent=2) + '\n')
print('completed', len(new), 'gain', sum(len(x['gained']) for x in out))
print('exceptions', [x for x in out if x['lost'] or x['changed_old_times'] or x['new_over500']])
