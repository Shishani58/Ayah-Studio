"""Frozen baseline + candidate on cached strong20 matrices, all controls."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
sys.path.insert(0,json.loads((TASK/'runtime-manifest.json').read_text())['runtime'])
from surah_cutter.core import align,decode,emissions,full_windows  # noqa: E402
from surah_cutter.data import quran_text,references  # noqa: E402
from prototype import recheck  # noqa: E402

emitter=emissions.load_emitter(ROOT/'models/wav2vec2-quran-phonetics.onnx')
real_bundle=align._OnnxBundle(emitter)
cache=Path.home()/'.cache/surah-cutter/full-accuracy'
previous=ROOT/'docs/tasks/2026-10-14-full-other-reciters'
recs=json.loads((previous/'data-manifest.json').read_text())['recordings']
cases=[{'reciter':'abu_bakr_ash_shatri','surah':s,
        'audio':str(Path.home()/f'Desktop/Abu_Bakr_Ash-Shatri/{s:03d}.mp3'),
        'reference':'abu_bakr_ash_shatri'} for s in [1,42,67,*range(78,115)]]
cases += [{'reciter':r['name'],'surah':s,'audio':str(Path(r['audio_dir'])/f'{s:03d}.mp3'),
           'reference':r['reference']} for r in recs for s in [80,98,112]]
results=[]
original=align._check_ayah_onsets
for case in cases:
    pcm=decode.decode(case['audio'])
    stem=f"{case['reciter']}-{case['surah']}"
    matrix_path=cache/(stem+'.npy')
    windows={}
    if matrix_path.exists():
        matrix=np.load(matrix_path)
        windows=json.loads((cache/(stem+'-windows.json')).read_text())
    else:
        matrix=full_windows.generate_pause_emissions(emitter,pcm,window_s=20,workers=3,
                        strong_pauses=True,max_window_s=60,trace=windows)
        np.save(matrix_path,matrix)
        (cache/(stem+'-windows.json')).write_text(json.dumps(windows))
    bundle=align._OnnxBundle(SimpleNamespace(vocab=emitter.vocab,blank_id=emitter.blank_id,
                                           emissions=lambda _:matrix))
    reads=[]
    snapshots=[]
    def patched(words,word_spans,target_spans,scores,stride,**kw):
        kw.pop('pcm',None)
        kw.pop('sr',None)
        checks=original(words,word_spans,target_spans,scores,stride,pcm=pcm,**kw)
        snapshots.extend([c.copy() for c in checks.values()])
        align._check_ayah_onsets=original
        try:
            return recheck(real_bundle,pcm,16000,words,word_spans,target_spans,stride,
                           checks,kw['pauses'],frame_scores=scores,base_ms=kw.get('base_ms',0),reads=reads)
        finally:
            align._check_ayah_onsets=patched
    align._check_ayah_onsets=patched
    # Local recheck calls the original checker, not the instrumented wrapper.
    trace={}
    try:
        rows=align.align_surah(pcm,16000,quran_text.surah_words(case['surah']),model=bundle,
                               surah=case['surah'],trace=trace)
    finally:
        align._check_ayah_onsets=original
    before={c['ayah']:next(x for x in reversed(snapshots) if x['ayah']==c['ayah']
                         and x['raw_onset_ms']==c['raw_onset_ms']) for c in trace['ayahs']}
    before_rows=[(a,before[a]['onset_ms'],before[a]['conf']) for a,_,_ in rows]
    align._review_pause_full_onsets(trace['ayahs'],windows)
    gt=references.load_reference(case['reference'])
    result={**case,'before':before_rows,'after':rows,'reads':reads,'checks':trace['ayahs'],
            'before_errors_ms':[ms-gt[case['surah'],a] for a,ms,_ in before_rows],
            'after_errors_ms':[ms-gt[case['surah'],a] for a,ms,_ in rows]}
    results.append(result)
    (TASK/'qpc-probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(stem,'changes',[(a,old[1],ms) for old,(a,ms,_) in zip(before_rows,rows) if old[1]!=ms],
          'reads',len(reads),flush=True)
