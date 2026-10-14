"""Bounded exploratory Whisper timing; each ASR/attention PCM pass is charged."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
from time import monotonic
from unittest.mock import patch

import numpy as np

from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_adapters import TarteelAdapter
from surah_cutter.core.fast_reads import LocalAcousticReads
from surah_cutter.core.model_runtime import RuntimeEmitter, release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import verify_dispatch


def main():
    original=decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/108.mp3")
    cropped=original[101120:]
    cases={"original_intro":original}
    for silence in (0,200,1200,2500):
        cases[f"derived_crop6320_silence{silence}"]=np.r_[np.zeros(silence*16,np.float32),cropped]
    words=quran_text.surah_words(108)
    output=[]
    for model in ("whisper-tarteel-tiny","whisper-tarteel-base"):
        model_store.ACTIVE_ID=model
        align._ONNX_CACHE.clear()
        release_runtimes()
        bundle=align.default_bundle()
        adapter=TarteelAdapter(bundle)
        for name,pcm in cases.items():
            silence=int(name.split("silence")[-1]) if "silence" in name else 0
            end=min(6000,2700+silence)
            reads=LocalAcousticReads(pcm,16000,None)
            calls=[]
            original_asr=RuntimeEmitter.recognize_local
            original_align=RuntimeEmitter.align_words
            def record(local,method):
                assert len(local)<=96000 and len(local)<len(pcm)
                calls.append({"method":method,"samples":len(local),"sha256":hashlib.sha256(memoryview(np.ascontiguousarray(local,np.float32))).hexdigest()})
            def asr(emitter,local,**kw):
                record(local,"independent_asr")
                return original_asr(emitter,local,**kw)
            def timing(emitter,local,target,**kw):
                record(local,"teacher_forced_attention")
                return original_align(emitter,local,target,**kw)
            def dispatch(operation,phase):
                s0,s1=reads._window(0,end)
                permitted=reads._permit(s0,s1)
                if permitted is None:
                    return None
                attempt=reads._begin_read(s0,s1,permitted,phase=phase)
                started=monotonic()
                try:
                    result=operation(np.ascontiguousarray(pcm[s0:s1],np.float32))
                    attempt["outcome"]="completed"
                    return result
                except BaseException:
                    attempt["outcome"]="failed"
                    raise
                finally:
                    attempt["seconds"]=monotonic()-started
            with ExitStack() as stack:
                stack.enter_context(patch.object(RuntimeEmitter,"recognize_local",asr))
                stack.enter_context(patch.object(RuntimeEmitter,"align_words",timing))
                obs=dispatch(lambda local:adapter.observe(local,16000,total_samples=len(pcm)),"independent_asr")
                raw=obs.units
                recognized=[]
                for word, rendering in zip(words,adapter.represent(words)):
                    offset=sum(len(u) for u in adapter.represent(recognized)) if recognized else 0
                    if tuple(raw[offset:offset+len(rendering)])!=rendering:
                        break
                    if min(obs.unit_quality[offset:offset+len(rendering)],default=0.)<.6:
                        break
                    recognized.append(word)
                    if len(recognized)==2:
                        break
                timings={}
                if recognized:
                    for count in range(1,len(recognized)+1):
                        timings[str(count)]=dispatch(lambda local:bundle.emitter.align_words(local,recognized[:count],strict_words=False),f"attention_{count}_independently_proved_words")
            trace=reads.diagnostics()
            verify_dispatch(calls,pcm,trace)
            output.append({"case":name,"model":model,"silence_ms":silence,"text":obs.text,
                           "units":obs.units,"unit_quality":obs.unit_quality,"proved_words":len(recognized),
                           "timings":timings,"ledger":trace,"actual_dispatches":calls,"runtime_identity":adapter.runtime_identity})
            print(model,name,"proved",len(recognized),"timings",timings,flush=True)
    release_runtimes()
    reference=Path("data/references/abu_bakr_ash_shatri.db")
    with sqlite3.connect(reference.resolve().as_uri()+"?mode=ro",uri=True) as db:
        reference_ms=db.execute("SELECT time FROM timings WHERE sura=108 AND ayah=1").fetchone()[0]
    report={"results":output,"reference_108_1_ms":reference_ms,
            "reference_sha256":hashlib.sha256(reference.read_bytes()).hexdigest(),
            "limitation":"Derived shifts test equivariance, not independently marked word onset/wasl. Runtime timing capability remains false."}
    Path(__file__).with_name("temporal-probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
