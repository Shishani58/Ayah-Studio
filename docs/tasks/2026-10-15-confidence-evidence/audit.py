"""Read-only replay of saved evidence; no model calls or database writes."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from surah_cutter.data.alignment_runs import load_run
from surah_cutter.data.onset_evidence import onset_evidence, saved_onset_evidence

db_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
payload = json.loads(log_path.read_text())
trace = payload['trace']
digest = lambda: hashlib.sha256(db_path.read_bytes()).hexdigest()
before = digest()
old = {'__name__': 'baseline'}
exec(subprocess.check_output([
    'rtk', 'proxy', 'git', 'show', 'HEAD:surah_cutter/data/onset_evidence.py',
], text=True), old)
rows = []
for c in trace['ayahs']:
    prior = old['onset_evidence'](c, c['onset_ms'], c['conf'], trace=trace)
    current = onset_evidence(c, c['onset_ms'], c['conf'], trace=trace)
    rows.append(dict(ayah=c['ayah'], time=c['onset_ms'],
                     before=prior['boundary'], after=current['boundary']))

with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
    db.row_factory = sqlite3.Row
    timings = dict(db.execute('SELECT ayah,time FROM timings WHERE sura=2'))
    reviews = {r['ayah']: dict(r) for r in db.execute('SELECT * FROM review_meta WHERE sura=2')}
signature = trace['audio_signature']
audio = db_path.parent / signature['name']
stat = audio.stat()
assert signature == dict(name=audio.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
evidence = saved_onset_evidence(db_path, 2, timings, reviews, signature)
snapshot_path = Path(json.loads(Path(
    'docs/tasks/2026-10-15-full-local-pause/saved.json').read_text())['log_paths']['run'])
snapshot = load_run(log_path.parent, 2, snapshot_path.stem)
assert len(evidence) == len(rows) == len(snapshot['ayahs']) == 286
assert all(r['onset_evidence'] == evidence[r['ayah']] for r in snapshot['ayahs'])
assert evidence[220]['boundary']['level'] == 'low'
controls = json.loads(Path('docs/tasks/2026-10-14-boundary-confidence/verified-controls.json').read_text())
for control in controls:
    c = control['check']
    assert onset_evidence(c, c['onset_ms'], c['conf'], trace={
        'full_boundary_reads': control['reads'],
    }) == control['evidence']
assert before == digest()
comparison = json.loads(Path('docs/tasks/2026-10-15-full-local-pause/comparison.json').read_text())
reference = {c['ayah']: c['reference_ms'] for c in comparison['comparison']}
reference_counts = {}
for level in ('high', 'medium', 'low'):
    differences = [abs(r['time'] - reference[r['ayah']]) for r in rows if r['after']['level'] == level]
    reference_counts[level] = dict(rows=len(differences), over500=sum(d > 500 for d in differences))
report = dict(
    input=str(log_path), input_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
    db_sha256_before=before, db_sha256_after=digest(),
    before=dict(collections.Counter(r['before']['level'] for r in rows)),
    after=dict(collections.Counter(r['after']['level'] for r in rows)),
    reasons=dict(collections.Counter(r['after']['reason'] for r in rows)),
    saved_reader_rows=len(evidence), snapshot_rows=len(snapshot['ayahs']),
    phrase_controls_passed=len(controls), known_error_220=evidence[220]['boundary'],
    reference_comparison=reference_counts, rows=rows,
)
Path(__file__).with_name('audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
