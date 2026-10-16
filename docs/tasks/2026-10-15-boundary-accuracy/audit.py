"""Read-only replay. Reference agreement is NOT human onset accuracy.

Run: rtk proxy .venv/bin/python -m docs.tasks.2026-10-15-boundary-accuracy.audit
Split declared before measurement: Shatri 2 development; other Shatri surahs
and all four other reciters held out. Never split neighbouring ayahs randomly.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

from surah_cutter.data.onset_evidence import onset_evidence, saved_onset_evidence

HERE = Path(__file__).parent
PROJECT = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri')
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def reference(reciter):
    path = Path('data/references') / f'{reciter}.db'
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        return {(s, a): t for s, a, t in db.execute('select sura,ayah,time from timings')}


def rows(reciter, surah, trace, ref):
    result = []
    for c in trace['ayahs']:
        evidence = onset_evidence(c, c['onset_ms'], c['conf'], trace=trace)
        if evidence is None:
            continue
        accuracy = evidence['timing_accuracy']
        result.append(dict(reciter=reciter, surah=surah, ayah=c['ayah'],
                           time=c['onset_ms'], reference=ref[surah, c['ayah']],
                           error=abs(c['onset_ms'] - ref[surah, c['ayah']]),
                           level=evidence['boundary']['level'], **accuracy))
    return result


def metrics(data):
    flagged = [r for r in data if r['review_priority'] == 'suspect']
    errors = [r for r in data if r['error'] > 500]
    found = [r for r in flagged if r['error'] > 500]
    old = [r for r in data if r['level'] == 'low']
    return dict(n=len(data), reference_over500=len(errors), review=len(flagged),
                found=len(found), missed=len(errors)-len(found),
                review_fraction=len(flagged)/len(data) if data else None,
                recall=len(found)/len(errors) if errors else None,
                precision=len(found)/len(flagged) if flagged else None,
                old_low_review=len(old), old_low_found=sum(r['error'] > 500 for r in old),
                unassessed=sum(r['review_priority'] == 'unassessed' for r in data))


ref = reference('abu_bakr_ash_shatri')
development, heldout = [], []
db_path = PROJECT / 'Abu_Bakr_Ash-Shatri.db'
db_before = hashlib.sha256(db_path.read_bytes()).hexdigest()
for path in sorted((PROJECT / 'Abu_Bakr_Ash-Shatri.logs').glob('sura_*_full.json')):
    d = read(path)
    batch = rows('abu_bakr_ash_shatri', d['surah'], d['trace'], ref)
    (development if d['surah'] == 2 else heldout).extend(batch)

stability = []
for reciter in ('saad_al_ghamidi', 'mohammed_siddiq_al_minshawi',
                'mahmoud_khalil_al_hussary', 'abdul_basit_mujawwad'):
    d = read(Path('docs/tasks/2026-10-14-full-other-reciters') / f'{reciter}.json')
    ref = reference(reciter)
    # One preselected production-window run per recording, no duplicate rows
    # from the seven configurations in calibration evaluation.
    case = next(c for c in d['cases'] if c['case'] == 'strong:20:3:0')
    for s in case['surahs']:
        checks = s['onset_checks']
        heldout.extend(rows(reciter, s['surah'], {'ayahs': checks}, ref))
        variants = [next(x for x in c['surahs'] if x['surah'] == s['surah'])
                    for c in d['cases']]
        assert len({x['audio_sha256'] for x in variants}) == 1
        for a, t, _ in s['rows']:
            times = [next(row[1] for row in v['rows'] if row[0] == a) for v in variants]
            stability.append(dict(reciter=reciter, surah=s['surah'], ayah=a,
                                  spread=max(times)-min(times), error=abs(t-ref[s['surah'], a])))

# Empirical grade calibration experiment; NOT installed as product probability.
bins = {}
for level in ('high', 'medium', 'low', 'unknown'):
    train = [r for r in development if r['level'] == level]
    test = [r for r in heldout if r['level'] == level]
    p = sum(r['error'] <= 500 for r in train)/len(train) if train else None
    bins[level] = dict(train_n=len(train), fitted_reference_agreement=p, test_n=len(test),
                       test_reference_agreement=sum(r['error'] <= 500 for r in test)/len(test) if test else None)
scored = [(bins[r['level']]['fitted_reference_agreement'], int(r['error'] <= 500))
          for r in heldout if bins[r['level']]['fitted_reference_agreement'] is not None]
constant = sum(r['error'] <= 500 for r in development)/len(development)
ece = sum(b['test_n']*abs(b['fitted_reference_agreement']-b['test_reference_agreement'])
          for b in bins.values() if b['test_n'] and b['fitted_reference_agreement'] is not None)/len(scored)
stable_wrong = [r for r in stability if r['spread'] <= 100 and r['error'] > 500]
latest_path = max((PROJECT / 'Abu_Bakr_Ash-Shatri.logs').glob('sura_002_*.json'),
                  key=lambda p: p.stat().st_mtime)
latest = read(latest_path)
with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
    db.row_factory = sqlite3.Row
    timings = dict(db.execute('select ayah,time from timings where sura=2'))
    reviews = {r['ayah']: dict(r) for r in db.execute('select * from review_meta where sura=2')}
signature = latest['trace']['audio_signature']
audio = PROJECT / signature['name']
stat = audio.stat()
assert signature == dict(name=audio.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
evidence = saved_onset_evidence(db_path, 2, timings, reviews, signature)
current_ref = reference('abu_bakr_ash_shatri')
db_after = hashlib.sha256(db_path.read_bytes()).hexdigest()
assert db_before == db_after, 'DB changed during read-only audit (possibly concurrent task)'
report = dict(target='reference agreement within 500ms; not human onset accuracy',
              probability_release='rejected: no human labels/convention; domain/policy mismatch',
              inputs_sha256=inputs, development=metrics(development), heldout=metrics(heldout),
              heldout_by_recording={f'{reciter}:{s}': metrics([r for r in heldout if r['reciter']==reciter and r['surah']==s])
                 for reciter,s in sorted({(r['reciter'],r['surah']) for r in heldout})},
              empirical_grade_calibration=dict(bins=bins, heldout_ece=ece,
                 scored_rows=len(scored), total_heldout_rows=len(heldout),
                 heldout_brier=sum((p-y)**2 for p,y in scored)/len(scored),
                 constant_train_rate=constant,
                 constant_heldout_brier=sum((constant-y)**2 for _,y in scored)/len(scored)),
              current_project=dict(latest_log=str(latest_path), mode=latest.get('mode'),
                 rows=len(timings), matching_evidence=len(evidence),
                 reference_over500=sum(abs(t-current_ref[2,a]) > 500 for a,t in timings.items()),
                 db_sha256_before=db_before, db_sha256_after=db_after,
                 note='Latest FAST lacks first_token_conf; abstain, do not reuse older FULL evidence'),
              window_stability=dict(n=len(stability), spread_over500=sum(r['spread']>500 for r in stability),
                 stable_within100_but_wrong_over500=len(stable_wrong), examples=stable_wrong[:10]),
              rows=development+heldout)
(HERE/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','inputs_sha256','heldout_by_recording')}, indent=2))
