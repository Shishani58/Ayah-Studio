"""Unique-boundary summary; no duplicate Shatri83/95/112 counting."""
import hashlib
import json
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
initial=json.loads((TASK/"diagnostics.json").read_text())
fixed=json.loads((TASK/"verified-fix.json").read_text())
broad=json.loads((TASK/"broad-audit.json").read_text())

def stats(values):
    a=np.abs(values)
    return {"n":len(values),"median_ms":float(np.median(a)),"p95_ms":float(np.percentile(a,95)),
            "max_ms":int(max(a)),"over500":int(sum(a>500))}

records=[]
for s in initial:
    if s["reciter"]=="abu_bakr_ash_shatri" and s["surah"]!=42:
        continue
    f=next(x for x in fixed if x["reciter"]==s["reciter"] and x["surah"]==s["surah"])
    records.append({"reciter":s["reciter"],"surah":s["surah"],"before_errors":s["errors_ms"],
                    "after_errors":f["errors_ms"],"checks":f["checks"]})
for s in broad:
    records.append({"reciter":"abu_bakr_ash_shatri","surah":s["surah"],
                    "before_errors":s["before_errors_ms"],"after_errors":s["after_errors_ms"],"checks":s["checks"]})
result={"recordings":len(records),"groups":{},"changed":[],"new_large":[],"removed_large":[]}
for name in [*dict.fromkeys(r["reciter"] for r in records),"all"]:
    group=[r for r in records if name=="all" or r["reciter"]==name]
    result["groups"][name]={"before":stats([e for r in group for e in r["before_errors"]]),
                          "after":stats([e for r in group for e in r["after_errors"]]),
                          "unflagged_after":sum(abs(e)>500 and c["reliable"] for r in group
                                               for e,c in zip(r["after_errors"],r["checks"]))}
for r in records:
    for ayah,(before,after) in enumerate(zip(r["before_errors"],r["after_errors"]),1):
        event={"reciter":r["reciter"],"surah":r["surah"],"ayah":ayah,"before_ms":before,"after_ms":after}
        if before!=after:
            result["changed"].append(event)
        if abs(before)<=500<abs(after):
            result["new_large"].append(event)
        if abs(after)<=500<abs(before):
            result["removed_large"].append(event)
(TASK/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
source_paths=["surah_cutter/core/align.py","surah_cutter/core/align_tiled.py",
              "surah_cutter/core/full_windows.py","surah_cutter/data/alignment_result.py",
              "tests/test_full_local_pause.py"]
manifest={"production":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
          "models":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                    ["models/wav2vec2-quran-phonetics.onnx","models/wav2vec2-quran-phonetics.fp32.onnx",
                     "models/wav2vec2-quran-phonetics.onnx.vocab.json"]},
          "decision":"Human delegated pause policy choice: retain speech-start contract; no GT-based early shift."}
frozen=json.loads((TASK.parent/"2026-10-14-full-other-reciters/runtime-manifest.json").read_text())
manifest["frozen_runtime_unchanged"]=all(hashlib.sha256((Path(frozen["runtime"])/p).read_bytes()).hexdigest()==sha
                                         for p,sha in frozen["files"].items())
assert manifest["frozen_runtime_unchanged"]
(TASK/"source-manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result["groups"],ensure_ascii=False,indent=2))
