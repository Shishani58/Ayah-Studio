"""DSP timing hypotheses fixed before reference scoring; no acoustic model calls."""
import csv,json,hashlib,sqlite3,wave
from pathlib import Path
import numpy as np
from surah_cutter.core import decode
p=Path(__file__).parent
# Same four wide Base controls, fixed by prior pause width >=800ms.
controls=[(1,2),(73,20),(77,25),(79,26)]
prior=json.loads((p.parent/'2026-10-15-fast-head-word-boundary/whisper-tarteel-base-after.json').read_text())
results=[]
for surah,ayah in controls:
 c=next(c for c in prior if c['surah']==surah)
 b=next(b for b in c['trace']['boundaries'] if b['ayah']==ayah)
 left,right=b['proof']['pause_ms'];audio=Path(f'/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03}.mp3')
 assert hashlib.sha256(audio.read_bytes()).hexdigest()==c['audio_sha256']
 pcm=decode.decode(str(audio));start=max(0,left-2000);end=min(len(pcm)/16,right+2000)
 crop=pcm[round(start*16):round(end*16)];width=320
 n=len(crop)//width;rms=np.sqrt(np.mean(crop[:n*width].reshape(-1,width)**2,axis=1))
 ts=start+(np.arange(n)+.5)*20
 candidates=[]
 for fraction in [.01,.02,.05]:
  threshold=float(np.max(rms)*fraction)
  l=ts[(ts<=(left+right)/2)&(rms>threshold)];r=ts[(ts>(left+right)/2)&(rms>threshold)]
  le=float(l[-1]+10) if len(l) else None;ri=float(r[0]-10) if len(r) else None
  candidates.append(dict(peak_fraction=fraction,threshold=threshold,previous_energy_end_ms=le,current_energy_start_ms=ri,midpoint_ms=(le+ri)/2 if le and ri else None))
 with (p/f'energy-{surah}-{ayah}.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['time_ms','rms']);w.writerows(zip(ts,rms))
 # Shareable local evidence, never claimed as actually listened to.
 with wave.open(str(p/f'local-{surah}-{ayah}.wav'),'wb') as f:
  f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes((np.clip(crop,-1,1)*32767).astype('<i2').tobytes())
 results.append(dict(surah=surah,ayah=ayah,audio_sha256=c['audio_sha256'],crop_ms=[start,end],dsp_pause_ms=[left,right],original_ms=b['start_ms'],hypotheses=candidates))
(p/'temporal-unscored.json').write_text(json.dumps(results,indent=2))
reference=Path('data/references/abu_bakr_ash_shatri.db')
with sqlite3.connect(f'file:{reference.resolve()}?mode=ro',uri=True) as db:
 for r in results:
  truth=db.execute('select time from timings where sura=? and ayah=?',(r['surah'],r['ayah'])).fetchone()[0]
  r.update(reference_ms=truth,original_signed_error_ms=r['original_ms']-truth,reference_fraction_in_dsp_pause=(truth-r['dsp_pause_ms'][0])/(r['dsp_pause_ms'][1]-r['dsp_pause_ms'][0]))
  for h in r['hypotheses']:h['signed_error_ms']=h['midpoint_ms']-truth if h['midpoint_ms'] else None
catalog=Path('/Users/ayub/Developer/quran_flutter/assets/data/reciters/reciters-base.json')
reciter=next(r for r in json.loads(catalog.read_text())['data'] if 'ash_shatri' in r.get('timecodes_db_url',''))
(p/'temporal-evaluation.json').write_text(json.dumps({'controls':results,'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'catalog_sha256':hashlib.sha256(catalog.read_bytes()).hexdigest(),'catalog_entry':reciter,'metadata_tables':['timings(sura,ayah,time)','db_version(code,version)'],'semantics':'App contract calls time start of ayah; DB/catalog contain no annotation method, listener identity or physical landmark definition. Waveform thresholds are diagnostic, not ground truth. No listening performed.'},ensure_ascii=False,indent=2))
print([(r['surah'],r['ayah'],r['original_signed_error_ms'],[round(h['signed_error_ms']) for h in r['hypotheses']]) for r in results])
