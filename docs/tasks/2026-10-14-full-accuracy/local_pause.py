"""Prospective local DSP repair: weak prefix + strong suffix of same first word."""
import json
from pathlib import Path
import sys
TASK = Path(__file__).resolve().parent
sys.path.insert(0,str(Path.home()/".cache/surah-cutter/full-reciter-audit/runtime-20261014"))
from surah_cutter.core import candidates, decode  # noqa: E402
data=json.loads((TASK/"diagnostics.json").read_text())
results=[]
for s in data:
    pcm=decode.decode(s["audio"])
    changes=[]
    for row,check,detail in zip(s["rows"],s["checks"],s["details"]):
        if check["repaired"] or check["first_token_conf"] >= .25:
            continue
        tokens=detail["tokens"][:detail["first_word_tokens"]]
        strong=next((t for t in tokens if t["conf"] >= .45),None)
        if strong is None:
            continue
        first=tokens[0]["start_ms"]
        stop=strong["start_ms"]
        if not 300 <= stop-first <= 4000:
            continue
        lo=max(0,first-1000)
        hi=min(len(pcm)/16,stop+1000)
        crop=pcm[round(lo*16):round(hi*16)]
        accepted=[d.pause for d in candidates.pause_pipeline_decisions(crop,16000) if d.accepted]
        supported=[p for p in accepted if first < lo+p.start_ms < lo+p.end_ms <= stop
                   and all(t["conf"]<.25 for t in tokens if t["start_ms"] < lo+p.end_ms)]
        if len(supported)==1:
            p=supported[0]
            changes.append({"ayah":row[0],"before_ms":row[1],"after_ms":round(lo+p.center_ms),
                            "pause_start_ms":lo+p.start_ms,"pause_end_ms":lo+p.end_ms,
                            "crop_start_ms":lo,"crop_end_ms":hi})
    # GT evaluation comes after decisions, not into crop selection/detection.
    proposed={c["ayah"]:c["after_ms"] for c in changes}
    errors=[proposed.get(r[0],r[1])-(r[1]-e) for r,e in zip(s["rows"],s["errors_ms"])]
    results.append({"reciter":s["reciter"],"surah":s["surah"],"changes":changes,"errors_ms":errors})
    print(s["reciter"],s["surah"],changes,flush=True)
(TASK/"local-pause-probe.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
