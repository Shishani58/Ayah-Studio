import json
from pathlib import Path
p=Path(__file__).parent

def compare(before,after,label,expected):
 a=json.loads((p/f'{before}.json').read_text());b=json.loads((p/f'{after}.json').read_text());assert len(a)==len(b)==expected
 aa={(c['model'],c['surah']):c for c in a};results=[]
 for c in b:
  old=aa[c['model'],c['surah']];t=c['trace'];ot=old['trace']
  for k in ['audio_sha256','reference_sha256','initial_pauses_sha256']:assert c[k]==old[k]
  ar={r[0]:r for r in old['rows'] if r[2]>0};br={r[0]:r for r in c['rows'] if r[2]>0}
  r=dict(model=c['model'],surah=c['surah'],before=len(ar),after=len(br),lost=[a for a in ar if a not in br],changed_prior=[a for a in ar if a in br and ar[a]!=br[a]],added=[{'row':r,'error_ms':e,'review':r[0] in t['review_required_ayahs']} for r,e in zip(c['rows'],c['errors_ms']) if r[0] in br and r[0] not in ar],new_outliers=[{'row':r,'error_ms':e} for r,e in zip(c['rows'],c['errors_ms']) if r[2]>0 and e>500 and (r[0] not in ar or r!=ar[r[0]])],calls=[len(ot['acoustic_reads']),len(t['acoustic_reads'])],inference_s=[ot['inference_seconds'],t['inference_seconds']],executor_s=[old['seconds'],c['seconds']],coverage=[ot['coverage'],t['coverage']],input_ratio=[ot['audio_ms']/old['duration_ms'],t['audio_ms']/c['duration_ms']],peak_rss=[old['peak_rss_self'],c['peak_rss_self']],dispatches_equal=old['actual_dispatches']==c['actual_dispatches'],rows_equal=old['rows']==c['rows'])
  assert not c['sequence_violations']
  assert t['coverage']<=.650001 and t['audio_ms']/c['duration_ms']<=2.000001
  assert all((r['window_ms'][1]-r['window_ms'][0])<=6000.001 for r in t['acoustic_reads'])
  assert len(t['acoustic_reads'])==len(c['actual_dispatches'])
  results.append(r)
 (p/f'{label}.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
 summary=[]
 for model in dict.fromkeys(r['model'] for r in results):
  rs=[r for r in results if r['model']==model]
  summary.append(dict(model=model,before=sum(r['before'] for r in rs),after=sum(r['after'] for r in rs),lost=sum(len(r['lost']) for r in rs),changed=sum(len(r['changed_prior']) for r in rs),added=sum(len(r['added']) for r in rs),new_outliers=sum(len(r['new_outliers']) for r in rs),calls=[sum(r['calls'][i] for r in rs) for i in range(2)],inference_s=[sum(r['inference_s'][i] for r in rs) for i in range(2)],executor_s=[sum(r['executor_s'][i] for r in rs) for i in range(2)]))
 (p/f'{label}-totals.json').write_text(json.dumps(summary,indent=2));print(label,summary)
if (p/'after-raw.summary.json').exists():compare('baseline-raw','after-raw','comparison',48)
if (p/'transfer-prototype.summary.json').exists():compare('transfer-baseline','transfer-prototype','transfer-comparison',12)
