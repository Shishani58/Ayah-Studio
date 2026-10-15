"""Independent ASR of model-selected starts and DSP opening, no GT prompts."""
import json
from pathlib import Path
import sys

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
sys.path.insert(0,str(ROOT))
from surah_cutter.core import align,candidates,decode,model_store  # noqa: E402
from surah_cutter.core.model_runtime import release_runtimes  # noqa: E402

qpc=json.loads((TASK/'qpc-probe.json').read_text())
muaalem=json.loads((TASK/'muaalem.json').read_text())
model_store.ACTIVE_ID='whisper-tarteel-base'
b=align._get_onnx_bundle()
results=[]
for name,surah,ayah in [('abu_bakr_ash_shatri',67,30),('abu_bakr_ash_shatri',83,10),
                       ('abu_bakr_ash_shatri',95,8),('abdul_basit_mujawwad',98,7)]:
    q=next(s for s in qpc if (s['reciter'],s['surah'])==(name,surah))
    pcm=decode.decode(q['audio'])
    sources=[('qpc',q['after'][ayah-1][1])]
    if surah==67:
        m=next(s for s in muaalem if (s['reciter'],s['surah'])==(name,surah))
        sources.append(('muaalem',m['rows'][ayah-1][1]))
    if name=='abdul_basit_mujawwad':
        pauses=candidates.detect_energy_pauses(pcm,16000,use_ten_vad=False)
        first=q['checks'][ayah-1]['raw_onset_ms']+200
        pause=next(p for p in pauses if first<p.start_ms<first+6000)
        sources.append(('right_dsp_opening',round(pause.end_ms-200)))
    for label,start in sources:
        end=min(len(pcm)/16,start+6000)
        r=b.emitter.recognize_local(pcm[round(start*16):round(end*16)],total_samples=len(pcm))
        result={'reciter':name,'surah':surah,'ayah':ayah,'label':label,
                'crop_ms':[start,end],'result':r}
        results.append(result)
        (TASK/'independent-problem-checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print(name,surah,ayah,label,r['text'],r['token_confidence'],flush=True)
release_runtimes()
