"""Current production FULL on the same cached matrices, plus fresh Fatiha."""
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
sys.path.insert(0,str(ROOT))
from surah_cutter.core import align,emissions,full_windows,model_store,pipeline  # noqa: E402

model_store.ACTIVE_ID='quran-phonetics'
bundle=align._OnnxBundle(emissions.load_emitter(ROOT/'models/wav2vec2-quran-phonetics.onnx'))
align._get_onnx_bundle=lambda:bundle
cache=Path.home()/'.cache/surah-cutter/full-accuracy'
cases=json.loads((TASK/'qpc-probe.json').read_text())
original=align._full_emissions
results=[]
for case in cases:
    stem=f"{case['reciter']}-{case['surah']}"
    matrix=np.load(cache/(stem+'.npy'))
    windows=json.loads((cache/(stem+'-windows.json')).read_text())
    def cached(_bundle,_pcm,_np,*,trace=None):
        trace['full_windows']=copy.deepcopy(windows)
        trace['full_windows'].update(algorithm=full_windows.FULL_POLICY,
                  review_policy=full_windows.FULL_REVIEW_POLICY,onset_policy=full_windows.FULL_ONSET_POLICY)
        return matrix
    align._full_emissions=cached
    trace={}
    actual=pipeline.process_surah(case['audio'],case['surah'],mode='full',trace=trace)
    verified={c['ayah']:c for c in trace['ayahs'] if c['reason']=='split-word-muaalem-proof'}
    expected=[(a,round(verified[a]['pause_end_ms']-100) if a in verified else ms)
              for a,ms,_ in case['after']]
    assert [(a,ms) for _,a,ms in actual]==expected
    assert all(c['conf']==old[2] for c,old in zip(trace['ayahs'],case['after']) if c['ayah'] not in verified)
    assert [c['reliable'] for c in trace['ayahs']]==[c['reliable'] for c in case['checks']]
    truth={a:ms-e for (a,ms,_),e in zip(case['before'],case['before_errors_ms'])}
    results.append({'reciter':case['reciter'],'surah':case['surah'],'rows':actual,
                    'checks':trace['ayahs'],'reads':trace.get('full_boundary_reads',[]),
                    'verifier_source':trace.get('full_boundary_verifier_source'),
                    'verifier_identity':trace.get('full_boundary_verifier_identity'),
                    'before':case['before'],'before_errors_ms':case['before_errors_ms'],
                    'after_errors_ms':[ms-truth[a] for _,a,ms in actual],
                    'matches_primary_probe':not verified,
                    'matches_probe':True,'cached_main_emissions':True})
    (TASK/'production.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(stem,'production matches',flush=True)
align._full_emissions=original
trace={}
started=time.perf_counter()
case=cases[0]
actual=pipeline.process_surah(case['audio'],1,mode='full',trace=trace)
assert [(a,ms) for _,a,ms in actual]==[(a,ms) for a,ms,_ in case['after']]
(TASK/'fresh-fatiha.json').write_text(json.dumps({'rows':actual,'trace':trace,
        'elapsed_s':time.perf_counter()-started,'fresh_main_emissions':True},ensure_ascii=False,indent=2)+'\n')
print('fresh Fatiha matches',flush=True)
case=next(s for s in cases if s['reciter']=='abdul_basit_mujawwad' and s['surah']==98)
trace={}
started=time.perf_counter()
actual=pipeline.process_surah(case['audio'],98,mode='full',trace=trace)
expected=next(s for s in results if s['reciter']==case['reciter'] and s['surah']==98)
assert actual==[tuple(r) for r in expected['rows']]
assert trace['full_boundary_verifier_source'].startswith('muaalem@')
assert model_store.ACTIVE_ID=='quran-phonetics'
(TASK/'fresh-mujawwad98.json').write_text(json.dumps({'rows':actual,'trace':trace,
        'elapsed_s':time.perf_counter()-started,'fresh_main_emissions':True},ensure_ascii=False,indent=2)+'\n')
print('fresh mujawwad98 matches, selected model remains QPC',flush=True)
