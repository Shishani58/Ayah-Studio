"""External evaluator: open references only after loading saved predictions.

Never imported by the application or the measurement program.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('labels', nargs='+')
args = parser.parse_args()
records = []
for label in args.labels:
    path = P / (label + '.jsonl')
    for line in path.read_text().splitlines():
        item = json.loads(line)
        # Predictions have already been materialized; references cannot
        # affect nominations, inference, thresholds or the saved rows.
        reciter = item['reciter']
        reference = (ROOT / 'data/references/abu_bakr_ash_shatri.db' if reciter == 'shatri' else
            Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / reciter / (reciter + '.db'))
        digest = hashlib.sha256(reference.read_bytes()).hexdigest()
        with sqlite3.connect(reference.resolve().as_uri() + '?mode=ro', uri=True) as conn:
            truth = dict(conn.execute('SELECT ayah,time FROM timings WHERE sura=?', (item['surah'],)))
        proven = [(a, t, c, abs(t-truth[a])) for a, t, c in item['rows'] if c > 0]
        assert digest == hashlib.sha256(reference.read_bytes()).hexdigest()
        records.append(dict(label=label, reciter=reciter, model=item['model'], surah=item['surah'],
            canonical_ayahs=len(truth), proven=len(proven), unresolved=len(truth)-len(proven),
            search_anchors=len(item['trace'].get('search_anchors', [])),
            positive_over500=[r for r in proven if r[3]>500],
            errors_ms=[r[3] for r in proven],
            seconds=item['seconds'], inference_seconds=item['trace'].get('inference_seconds'),
            calls=len(item['dispatches']), model_input_ms=item['trace']['audio_ms'],
            covered_ms=item['trace']['covered_audio_ms'], duration_ms=item['trace']['total_ms'],
            reference_sha256=digest, evidence_source_sha256=hashlib.sha256(line.encode()).hexdigest()))
(P / 'comparison.json').write_text(json.dumps(records, indent=2) + '\n')
for r in records:
    print(r['label'], r['reciter'], r['model'], r['surah'], 'proof',r['proven'], 'unknown',r['unresolved'],
          'bad',r['positive_over500'], 'calls',r['calls'], 'seconds',round(r['seconds'],2))
