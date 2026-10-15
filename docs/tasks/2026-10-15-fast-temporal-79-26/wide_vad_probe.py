"""All four public Base pauses >=800ms, fixed local VAD context controls."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from surah_cutter.core import candidates,decode

task=Path(__file__).parent.resolve()
source=task.parent/"2026-10-15-fast-head-word-boundary/whisper-tarteel-base-after.json"
cases=json.loads(source.read_text())
result=[]
for c in cases:
    selected=[b for b in c["trace"]["boundaries"] if b["confidence"]>0 and b["evidence_kind"]=="pause"
              and b["proof"]["pause_ms"][1]-b["proof"]["pause_ms"][0]>=800]
    if not selected:
        continue
    audio=Path(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{c['surah']:03}.mp3")
    assert hashlib.sha256(audio.read_bytes()).hexdigest()==c["audio_sha256"]
    pcm=decode.decode(str(audio))
    for boundary in selected:
        left,right=boundary["proof"]["pause_ms"]
        controls=[]
        for context in [1200,2500,4000]:
            a,b=round(max(0,left-context)*16),round(min(len(pcm)/16,right+1500)*16)
            pauses=candidates.detect_silero_pauses(pcm[a:b],16000,min_pause_ms=200)
            absolute=[{**asdict(p),"start_ms":p.start_ms+a/16,"end_ms":p.end_ms+a/16,
                       "midpoint_ms":p.center_ms+a/16} for p in pauses]
            overlapping=[p for p in absolute if min(p["end_ms"],right)>max(p["start_ms"],left)]
            best=max(overlapping,key=lambda p:min(p["end_ms"],right)-max(p["start_ms"],left),default=None)
            controls.append({"left_context_ms":context,"window_ms":[a/16,b/16],"all_pauses":absolute,"overlap":best})
        result.append({"surah":c["surah"],"ayah":boundary["ayah"],"original_ms":boundary["start_ms"],
                       "original_interval_ms":[left,right],"audio_sha256":c["audio_sha256"],"controls":controls})
        (task/"wide-vad-controls.json").write_text(json.dumps({"scope":"DSP/VAD only, no text model or referenceinput","cases":result},ensure_ascii=False,indent=2)+"\n")
print([(c["surah"],c["ayah"],[(r["left_context_ms"],r["overlap"]) for r in c["controls"]]) for c in result])
