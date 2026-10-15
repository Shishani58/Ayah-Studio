"""Local VAD controls around the same suspect DSP interval; no text model."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from surah_cutter.core import candidates,decode

task=Path(__file__).parent.resolve()
audio=Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/079.mp3")
case=json.loads((task/"seam-79-26.json").read_text())[0]
assert hashlib.sha256(audio.read_bytes()).hexdigest()==case["audio_sha256"]
pcm=decode.decode(str(audio))
left,right=case["boundary"]["proof"]["pause_ms"]
a,b=round((left-2500)*16),round((right+1500)*16)
result={"audio_sha256":case["audio_sha256"],"window_ms":[a/16,b/16],
        "scope":"VAD/DSP only, no known text or reference provided","detectors":[]}
for name,fn in [("silero",candidates.detect_silero_pauses),("ten",candidates.detect_ten_pauses)]:
    try:
        pauses=fn(pcm[a:b],16000,min_pause_ms=200)
        row={"name":name,"pauses":[{**asdict(p),"absolute_start_ms":p.start_ms+a/16,
                                     "absolute_end_ms":p.end_ms+a/16,"absolute_midpoint_ms":p.center_ms+a/16} for p in pauses]}
    except RuntimeError as exc:
        row={"name":name,"unavailable":str(exc)}
    result["detectors"].append(row)
(task/"vad-controls.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
