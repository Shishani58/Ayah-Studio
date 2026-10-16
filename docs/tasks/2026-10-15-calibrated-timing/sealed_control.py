"""Additional whole recording checked after freezing the calibration table."""
import hashlib
import json
from pathlib import Path
import sqlite3

from surah_cutter.data.onset_evidence import onset_evidence
from surah_cutter.data.timing_probability import MODEL_SHA256

here = Path(__file__).parent
artifact = Path('surah_cutter/data/timing_calibration.json')
before = hashlib.sha256(artifact.read_bytes()).hexdigest()
source = Path('docs/tasks/2026-10-14-full-pause-windows/surah67-20-vs-60-2-pause-60-2-0.json')
payload = json.loads(source.read_text())
assert payload['model_sha256'] == MODEL_SHA256
data = payload['surahs'][0]
assert data['surah'] == 67
training = json.loads((here/'evaluation.json').read_text())
assert not any(r['group'] == 'abu_bakr_ash_shatri:67' for r in training['rows'])
ref = Path('data/references/abu_bakr_ash_shatri.db')
with sqlite3.connect(f'file:{ref}?mode=ro', uri=True) as db:
    times = dict(db.execute('select ayah,time from timings where sura=67'))
checks = [dict(ayah=a, onset_ms=t, conf=c) for a,t,c in data['rows']]
trace = dict(model_source=f'onnx-ctc:quran-phonetics:sha256={MODEL_SHA256}',
             mode='full', full_windows=data['windows'], ayahs=checks)
rows = []
for c in checks:
    p = onset_evidence(c,c['onset_ms'],c['conf'],trace=trace)['timing_accuracy']['probability']
    rows.append(dict(ayah=c['ayah'],probability=p,error_ms=abs(c['onset_ms']-times[c['ayah']])))
report = dict(calibration_sha256=before,source=str(source),
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              n=len(rows),over500=sum(r['error_ms']>500 for r in rows),
              predicted_rate=sum(r['probability'] for r in rows)/len(rows),
              observed_rate=sum(r['error_ms']<=500 for r in rows)/len(rows),rows=rows)
assert before == hashlib.sha256(artifact.read_bytes()).hexdigest()
(here/'sealed-67.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
