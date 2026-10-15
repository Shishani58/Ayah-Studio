"""Read-only FULL comparison of installed alternatives on the same Fatiha."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
sys.path.insert(0,str(ROOT))
from surah_cutter.core import align,decode,model_store  # noqa: E402
from surah_cutter.core.model_runtime import release_runtimes  # noqa: E402
from surah_cutter.data import quran_text,references  # noqa: E402

audio=Path.home()/"Desktop/Abu_Bakr_Ash-Shatri/001.mp3"
pcm=decode.decode(str(audio))
results=[]
for model_id in ['muaalem','fastconformer-arabic','whisper-large-v3','whisper-tarteel-base']:
    item={'model':model_id,'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest()}
    started=time.perf_counter()
    try:
        if model_store.runtime_path(model_id) is None:
            raise RuntimeError('Model is not installed; downloads are not part of this audit')
        model_store.ACTIVE_ID=model_id
        align._ONNX_CACHE.clear()
        release_runtimes()
        stage=time.perf_counter()
        bundle=align._get_onnx_bundle()
        item['load_s']=time.perf_counter()-stage
        trace={}
        stage=time.perf_counter()
        rows=align.align_surah(pcm,16000,quran_text.surah_words(1),model=bundle,surah=1,trace=trace)
        item['full_s']=time.perf_counter()-stage
        # Reference is for evaluation, never target/window construction.
        gt=references.load_reference('abu_bakr_ash_shatri')
        errors=[ms-gt[1,a] for a,ms,_ in rows]
        assert [a for a,_,_ in rows]==list(range(1,8))
        assert all(b[1]>a[1] for a,b in zip(rows,rows[1:]))
        item.update(rows=rows,errors_ms=errors,trace=trace,source=model_store.source(),
                    median_ms=float(np.median(np.abs(errors))),max_ms=max(map(abs,errors)),
                    over500=sum(abs(e)>500 for e in errors))
    except Exception as exc:
        item['error']=f'{type(exc).__name__}: {exc}'
    finally:
        release_runtimes()
    item['elapsed_s']=time.perf_counter()-started
    results.append(item)
    (TASK/'fatiha-model-comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in item.items() if k not in ('rows','trace','source')},ensure_ascii=False),flush=True)
