"""Current production pipeline replay (cached emissions) and fresh Shatri42 run."""
import copy
import json
from pathlib import Path
import sys
import time
import resource
import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
sys.path.insert(0,str(ROOT))
from surah_cutter.core import align, emissions, full_windows, model_store, pipeline  # noqa: E402

model_store.ACTIVE_ID="quran-phonetics"
bundle=align._OnnxBundle(emissions.load_emitter(ROOT/"models/wav2vec2-quran-phonetics.onnx"))
align._get_onnx_bundle=lambda:bundle
sources=json.loads((TASK/"diagnostics.json").read_text())
trial=json.loads((TASK/"local-pause-probe.json").read_text())
results=[]
original=align._full_emissions
for s in sources:
    stem=f"{s['reciter']}-{s['surah']}"
    matrix=np.load(Path.home()/f".cache/surah-cutter/full-accuracy/{stem}.npy")
    def cached(_bundle, _pcm, _np, *, trace=None):
        trace["full_windows"]=copy.deepcopy(s["windows"])
        trace["full_windows"].update(algorithm=full_windows.FULL_POLICY,
                        review_policy=full_windows.FULL_REVIEW_POLICY,onset_policy=full_windows.FULL_ONSET_POLICY)
        return matrix
    align._full_emissions=cached
    trace={}
    started=time.perf_counter()
    rows=pipeline.process_surah(s["audio"],s["surah"],mode="full",trace=trace)
    expected=next(x for x in trial if x["reciter"]==s["reciter"] and x["surah"]==s["surah"])
    changes={c["ayah"]:c["after_ms"] for c in expected["changes"]}
    assert [(a,ms) for _,a,ms in rows]==[(a,changes.get(a,ms)) for a,ms,_ in s["rows"]]
    assert [c["conf"] for c in trace["ayahs"]]==[c["conf"] for c in s["checks"]]
    assert [c["reliable"] for c in trace["ayahs"]]==[c["reliable"] for c in s["checks"]]
    results.append({"reciter":s["reciter"],"surah":s["surah"],"cached_emissions":True,
                    "elapsed_s":time.perf_counter()-started,"rows":rows,"checks":trace["ayahs"],
                    "errors_ms":expected["errors_ms"],"matches_prospective_probe":True})
    print(stem,"production replay matches",flush=True)
(TASK/"verified-fix.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
align._full_emissions=original
s=next(s for s in sources if s["surah"]==42)
trace={}
started=time.perf_counter()
rows=pipeline.process_surah(s["audio"],42,mode="full",trace=trace)
expected=next(x for x in results if x["surah"]==42)
assert rows==[tuple(r) for r in expected["rows"]]
fresh={"cached_emissions":False,"elapsed_s":time.perf_counter()-started,"rows":rows,"trace":trace,
       "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
       "matches_cached_replay":True}
(TASK/"fresh-42.json").write_text(json.dumps(fresh,ensure_ascii=False,indent=2)+"\n")
print("fresh42 matches",fresh["elapsed_s"],flush=True)
