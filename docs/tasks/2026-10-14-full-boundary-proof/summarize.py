"""Same-recording before/after and model comparisons; no inference GT input."""
import hashlib
import json
from pathlib import Path
import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
qpc=json.loads((TASK/'qpc-probe.json').read_text())
muaalem=json.loads((TASK/'muaalem.json').read_text())
production=json.loads((TASK/'production.json').read_text())
# Current production can add an independently confirmed Muaalem opening.
for case in qpc:
    actual=next(s for s in production if (s['reciter'],s['surah'])==(case['reciter'],case['surah']))
    case['after']=[(a,ms,c['conf']) for (_,a,ms),c in zip(actual['rows'],actual['checks'])]
    case['after_errors_ms']=actual['after_errors_ms']
    case['checks']=actual['checks']

def stats(values):
    a=np.abs(values)
    return {'n':len(values),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),
            'max_ms':int(max(a)),'over500':int(sum(a>500))}

summary={'qpc_all':{},'comparison':{},'qpc_changes':[],'qpc_new_large':[],
         'model_new_large':[],'model_removed_large':[],'failures':[]}
for name in ['all',*dict.fromkeys(s['reciter'] for s in qpc)]:
    cases=[s for s in qpc if name=='all' or s['reciter']==name]
    summary['qpc_all'][name]={'before':stats([e for s in cases for e in s['before_errors_ms']]),
                            'after':stats([e for s in cases for e in s['after_errors_ms']]),
                            'unflagged_after':sum(abs(e)>500 and c['reliable'] for s in cases
                                                 for e,c in zip(s['after_errors_ms'],s['checks']))}
for s in qpc:
    for a,(before,after) in enumerate(zip(s['before_errors_ms'],s['after_errors_ms']),1):
        event={'reciter':s['reciter'],'surah':s['surah'],'ayah':a,'before_ms':before,'after_ms':after}
        if before!=after:
            summary['qpc_changes'].append(event)
        if abs(before)<=500<abs(after):
            summary['qpc_new_large'].append(event)

pairs=[]
for m in muaalem:
    if 'error' in m:
        summary['failures'].append(m)
        continue
    q=next(s for s in qpc if (s['reciter'],s['surah'])==(m['reciter'],m['surah']))
    assert len(q['after'])==len(m['rows'])
    assert m['audio_sha256']==hashlib.sha256(Path(q['audio']).read_bytes()).hexdigest()
    pairs.append((q,m))
    for a,(qe,me) in enumerate(zip(q['after_errors_ms'],m['errors_ms']),1):
        event={'reciter':m['reciter'],'surah':m['surah'],'ayah':a,'qpc_error_ms':qe,'muaalem_error_ms':me,
               'muaalem_flagged':not m['trace']['ayahs'][a-1]['reliable']}
        if abs(qe)<=500<abs(me):
            summary['model_new_large'].append(event)
        if abs(me)<=500<abs(qe):
            summary['model_removed_large'].append(event)
for name in ['all',*dict.fromkeys(m['reciter'] for _,m in pairs)]:
    group=[(q,m) for q,m in pairs if name=='all' or m['reciter']==name]
    summary['comparison'][name]={'files':len(group),'qpc':stats([e for q,_ in group for e in q['after_errors_ms']]),
        'muaalem':stats([e for _,m in group for e in m['errors_ms']]),
        'muaalem_unflagged_large':sum(abs(e)>500 and c['reliable'] for _,m in group
                                     for e,c in zip(m['errors_ms'],m['trace']['ayahs']))}
summary['production_matches_probe']=len(production)==len(qpc) and all(s['matches_probe'] for s in production)
summary['extra_reads']=sum(len(s['reads']) for s in production)
summary['extra_audio_ms']=sum(r['end_ms']-r['start_ms'] for s in production for r in s['reads'])
(TASK/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
manifest=json.loads((TASK/'runtime-manifest.json').read_text())
unchanged=all(hashlib.sha256((Path(manifest['runtime'])/p).read_bytes()).hexdigest()==sha
              for p,sha in manifest['files'].items())
assert unchanged
source_files=['surah_cutter/core/full_boundary.py','surah_cutter/core/align.py',
              'surah_cutter/core/align_tiled.py','surah_cutter/core/full_windows.py',
              'surah_cutter/data/alignment_identity.py','surah_cutter/data/alignment_result.py',
              'surah_cutter/webgui/server.py','surah_cutter/gui/dashboard_view.py',
              'tests/test_full_boundary_verifier.py','tests/test_full_resume.py',
              'tests/test_full_phrase_boundary.py','tests/test_alignment_audio.py','tests/test_webgui_server.py']
(TASK/'source-manifest.json').write_text(json.dumps({'frozen_runtime_unchanged':unchanged,
    'production':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files}},indent=2)+'\n')
print(json.dumps(summary['comparison'],ensure_ascii=False,indent=2))
print('QPC all',summary['qpc_all']['all'],'reads',summary['extra_reads'],summary['extra_audio_ms'])
print('Muaalem new/removed',len(summary['model_new_large']),len(summary['model_removed_large']))
