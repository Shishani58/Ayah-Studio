"""Supported ASR body cue uses a measured read interval, never word timing."""
import hashlib
import json
import math
from pathlib import Path
from surah_cutter.core import align,decode,model_store
from surah_cutter.core.fast_adapters import TarteelAdapter
from surah_cutter.core.fast_executor import plan_phrase,_expansion_evidence
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy,verify_dispatch

task = Path(__file__).parent.resolve()
prior = task.parent / "2026-10-14-fast-previous-body-rejection"
case = next(c for c in json.loads((prior / "whisper-tarteel-base-after.json").read_text()) if c["surah"]==59)
audio = Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/059.mp3")
assert hashlib.sha256(audio.read_bytes()).hexdigest()==case["audio_sha256"]
pcm = decode.decode(str(audio))
model_store.ACTIVE_ID = "whisper-tarteel-base"
bundle = align._get_onnx_bundle()
adapter = TarteelAdapter(bundle)
words = quran_text.surah_words(59)
rendered = adapter.represent(words)
proof_ends = adapter.proof_word_ends(words,rendered)
groups,ends,weights = {},{},{}
for w,u,e in zip(words,rendered,proof_ends):
    base = len(groups.get(w.ayah,()))
    groups.setdefault(w.ayah,[]).extend(u)
    ends.setdefault(w.ayah,[]).extend(base+x for x in e)
    weights[w.ayah] = weights.get(w.ayah,0)+sum(map(len,u))
groups = {a:tuple(u) for a,u in groups.items()}
cue = case["trace"]["acoustic_reads"][42]
wi = 213
assert words[wi].ayah==10 and words[wi+1].ayah==10
pair = rendered[wi]+rendered[wi+1]
assert [i for i in range(len(words)-1) if rendered[i]+rendered[i+1]==pair]==[wi]
o = cue["observed"]
assert not o["truncated"] and o["positions_ms"] is None
matches = [i for i in range(len(o["units"])-len(pair)+1) if tuple(o["units"][i:i+len(pair)])==pair
           and min(o["unit_quality"][i:i+len(pair)])>=.25]
assert len(matches)==1
cue_ms = sum(cue["window_ms"])/2
anchors = [b for b in case["trace"]["boundaries"] if b["confidence"]>0 and b["ayah"]<10]
assert anchors[-1]["ayah"]==9 and len(anchors)>=2
tempo = (anchors[-1]["start_ms"]-anchors[0]["start_ms"])/sum(weights[a] for a in weights if anchors[0]["ayah"]<=a<9)
remaining = sum(sum(map(len,u)) for w,u in zip(words,rendered) if w.ayah==10 and w.index>=words[wi].index)
span = tempo*remaining
forecasts = {11:cue_ms+span,12:cue_ms+span+tempo*weights[11]}
left,right = cue["window_ms"][0]+span*.2, min(len(pcm)/16,cue["window_ms"][1]+span*2.5)
pool = [(i,p) for i,p in enumerate(case["trace"]["phase1_pauses"])
        if cue["window_ms"][1]<p["center_ms"] and left<=p["center_ms"]<=right
        and case["trace"]["pause_geometry"][i]["accepted"]]
reserved = {}
for f in forecasts.values():
    reserved.update(sorted(pool,key=lambda ip:(abs(ip[1]["center_ms"]-f),ip[0]))[:4])
for i,p in sorted(pool,key=lambda ip:(min(abs(ip[1]["center_ms"]-f) for f in forecasts.values()),ip[0])):
    if len(reserved)>=8:
        break
    reserved[i]=p
selected = sorted(reserved.items(),key=lambda ip:ip[1]["center_ms"])
plans = {a:plan_phrase(groups,a,ends) for a in [11,12]}
reads = LocalEvidenceReads(pcm,16000,adapter,cache_limit_bytes=32*1024*1024)
out = {"scope":"ASR cue window for search only; no current10onset","cue_window_ms":cue["window_ms"],
       "body_pair":pair,"word_indices":[wi,wi+1],"forecasts":forecasts,"region":[left,right],
       "selected":selected,"found":None,"evaluations":[],"audio_sha256":case["audio_sha256"],
       "runtime_identity":adapter.runtime_identity}
try:
    with dispatch_spy(bundle,pcm) as dispatches:
        for index,pause in selected:
            for speech in [1300,2500,4000]:
                obs = reads.read(max(pause["start_ms"],pause["end_ms"]-200),min(len(pcm)/16,pause["end_ms"]+speech),reason=f"ayah=10:body_future:pause={index}")
                if obs is None:
                    break
                expand = False
                for ayah,plan in plans.items():
                    if plan["ambiguous"]:
                        continue
                    phrase = tuple(plan["units"])
                    quality = obs.unit_quality[:len(phrase)]
                    complete = tuple(obs.units[:len(phrase)])==phrase
                    alias_conflict = any("spoken_alias" in item and tuple(item["source_units"])!=rendered[next(i for i,w in enumerate(words) if w.ayah==ayah)] for item in obs.attribution[:len(phrase)])
                    out["evaluations"].append({"pause_id":index,"speech_ms":speech,"ayah":ayah,"complete":complete,"units":obs.units,"min_quality":min(quality,default=0)})
                    if complete and not obs.truncated and not alias_conflict and len(quality)==len(phrase) and all(math.isfinite(q) for q in quality) and min(quality)>=.25:
                        out["found"]={"ayah":ayah,"read_id":reads.last_read_id,"phrase":phrase,"min_quality":min(quality),"pause_id":index,"geometry":case["trace"]["pause_geometry"][index]}
                        break
                    if not alias_conflict and _expansion_evidence(obs,phrase,ends[ayah],.25,groups,ayah,ends):
                        expand=True
                out.update(reads=reads.calls,actual_dispatches=dispatches)
                (task / "probe-base-59.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
                if out["found"] or not expand:
                    break
            if out["found"]:
                break
    verify_dispatch(dispatches,pcm,{"acoustic_reads":reads.calls})
finally:
    release_runtimes()
print("calls",len(dispatches),"found",out["found"])
