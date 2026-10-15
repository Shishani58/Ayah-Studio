"""Read-only, frozen-runtime Muaalem FULL audit, GT opened after predictions."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
runtime=Path(json.loads((TASK/'runtime-manifest.json').read_text())['runtime'])
sys.path.insert(0,str(runtime))
from surah_cutter.core import align,decode,model_store  # noqa: E402
from surah_cutter.core.model_runtime import release_runtimes  # noqa: E402
from surah_cutter.data import quran_text,references  # noqa: E402

model_store.ACTIVE_ID='muaalem'
started=time.perf_counter()
bundle=align._get_onnx_bundle()
load_s=time.perf_counter()-started
previous=ROOT/'docs/tasks/2026-10-14-full-other-reciters'
recs=json.loads((previous/'data-manifest.json').read_text())['recordings']
cases=[{'reciter':'abu_bakr_ash_shatri','surah':s,
        'audio':str(Path.home()/f'Desktop/Abu_Bakr_Ash-Shatri/{s:03d}.mp3'),
        'reference':'abu_bakr_ash_shatri'} for s in [1,83,95,78,89,67,112,114,42]]
cases += [{'reciter':r['name'],'surah':s,'audio':str(Path(r['audio_dir'])/f'{s:03d}.mp3'),
           'reference':r['reference']} for s in [98,112,80] for r in recs]
results=[]
for case in cases:
    item={**case,'model':'muaalem','load_s':load_s}
    started=time.perf_counter()
    try:
        audio=Path(case['audio'])
        pcm=decode.decode(str(audio))
        trace={}
        item['audio_sha256']=hashlib.sha256(audio.read_bytes()).hexdigest()
        stage=time.perf_counter()
        rows=align.align_surah(pcm,16000,quran_text.surah_words(case['surah']),
                               model=bundle,surah=case['surah'],trace=trace)
        item['full_s']=time.perf_counter()-stage
        assert [r[0] for r in rows]==list(range(1,quran_text.ayah_count(case['surah'])+1))
        assert all(b[1]>a[1] for a,b in zip(rows,rows[1:]))
        gt=references.load_reference(case['reference'])
        item.update(rows=rows,errors_ms=[ms-gt[case['surah'],a] for a,ms,_ in rows],trace=trace,
                    source=model_store.source())
    except Exception as exc:
        item['error']=f'{type(exc).__name__}: {exc}'
    item['elapsed_s']=time.perf_counter()-started
    item['peak_rss_mib_parent_only']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    results.append(item)
    (TASK/'muaalem.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(case['reciter'],case['surah'],item.get('error') or
          f"{item['full_s']:.1f}s max={max(map(abs,item['errors_ms']))} over500={sum(abs(e)>500 for e in item['errors_ms'])}",flush=True)
release_runtimes()
