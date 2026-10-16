"""Compare saved keys/times and causal inputs; no reference or inference."""
import copy
import json
from pathlib import Path

P = Path(__file__).resolve().parent
baseline = [json.loads(line) for line in (P/'baseline.jsonl').read_text().splitlines()]
frozen_path = P.parent/'2026-10-16-fast-addressed-recovery/PF-final-SF-saad_al_ghamidi-replay.json'
frozen = json.loads(frozen_path.read_text())
results = []
for label in ('current-shatri', 'current-ghamidi', 'final-replay2', 'final-regression80', 'final-smoke', 'final-frozen-replay2'):
    for line in (P/(label+'.jsonl')).read_text().splitlines():
        after = json.loads(line)
        pool = frozen if after['reciter']=='saad_al_ghamidi' else baseline
        before = next((i for i in pool if i.get('model')==after['model'] and i['surah']==after['surah']), None)
        result = dict(label=label,reciter=after['reciter'],model=after['model'],surah=after['surah'])
        if before is None:
            results.append({**result,'comparison':'current_only'})
            continue
        b, a = before['trace']['run_snapshot'], after['trace']['run_snapshot']
        equal = {k:b.get(k)==a.get(k) for k in ('pcm_sha256','text_sha256','text_runtime','dsp_config','dsp_pauses_sha256')}
        old_dsp,new_dsp = copy.deepcopy(b['dsp_models']),copy.deepcopy(a['dsp_models'])
        # A checkout's pathname is provenance, not different Silero weights.
        old_dsp.get('silero',{}).pop('path',None)
        new_dsp.get('silero',{}).pop('path',None)
        equal['dsp_effective_identity'] = old_dsp==new_dsp
        old = {r[0]:r[1] for r in before['rows'] if r[2]>0}
        new = {r[0]:r[1] for r in after['rows'] if r[2]>0}
        results.append({**result,'causal_inputs_equal':equal,'before':len(old),'after':len(new),
            'added':sorted(new.keys()-old.keys()),'lost':sorted(old.keys()-new.keys()),
            'shifted':[dict(ayah=k,before=t,after=new[k]) for k,t in old.items() if k in new and new[k]!=t],
            'before_times':old,'after_times':new,'before_kind':'frozen PF14' if pool is frozen else 'fresh baseline',
            'after_kind':after.get('measurement_kind','fresh inference')})
(P/'paired-deltas.json').write_text(json.dumps(results,indent=2)+'\n')
for r in results:
    print(r['label'],r['model'],r['surah'],r.get('before'),'->',r.get('after'),
          'lost',r.get('lost'),'shifted',r.get('shifted'),'inputs',r.get('causal_inputs_equal'))
