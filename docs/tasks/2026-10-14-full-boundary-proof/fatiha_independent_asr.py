"""Independent Tarteel decode on DSP-selected Fatiha clips. No canonical prompt/GT."""
import json
from pathlib import Path
import sys
import time

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
sys.path.insert(0,str(ROOT))
from surah_cutter.core import align,candidates,decode,model_store  # noqa: E402
from surah_cutter.core.model_runtime import release_runtimes  # noqa: E402

pcm=decode.decode(str(Path.home()/'Desktop/Abu_Bakr_Ash-Shatri/001.mp3'))
pause=candidates.detect_energy_pauses(pcm,16000,use_ten_vad=False)[0]
clips=[('left_tail',pause.start_ms-2000,pause.start_ms+200),
       ('right_opening',pause.end_ms-200,pause.end_ms+2000),
       ('joint',pause.start_ms-2500,pause.end_ms+2000)]
results=[]
for model_id in ['whisper-tarteel-tiny','whisper-tarteel-base']:
    model_store.ACTIVE_ID=model_id
    bundle=align._get_onnx_bundle()
    for label,lo,hi in clips:
        clip=pcm[round(lo*16):round(hi*16)]
        started=time.perf_counter()
        result=bundle.emitter.recognize_local(clip,total_samples=len(pcm))
        results.append({'model':model_id,'label':label,'crop_ms':[lo,hi],
                        'elapsed_s':time.perf_counter()-started,'result':result})
        print(model_id,label,result['text'],result['token_confidence'],flush=True)
        (TASK/'fatiha-independent-asr.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    align._ONNX_CACHE.clear()
    release_runtimes()
