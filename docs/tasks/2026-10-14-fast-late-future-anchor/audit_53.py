"""Read-only DSP evidence for the newly reachable 2:53; no neural calls."""
import hashlib
import json
from pathlib import Path
import wave
import numpy as np
from surah_cutter.core.decode import decode

task = Path(__file__).parent.resolve()
case = json.loads((task / "late-2.json").read_text())[0]
evidence = json.loads((task / "2-53-outlier-proof.json").read_text())
audio = Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3")
assert hashlib.sha256(audio.read_bytes()).hexdigest() == case["audio_sha256"]
pcm = decode(str(audio))
left, right = evidence["geometry"]["interval_ms"]
frames = []
for t in range(int(left) - 200, int(right) + 600, 20):
    samples = pcm[t * 16:(t + 20) * 16]
    frames.append({"start_ms": t, "rms": float(np.sqrt(np.mean(samples * samples))),
                   "peak": float(np.max(np.abs(samples)))})
start, end = int(left) - 1000, int(right) + 2500
clip = (np.clip(pcm[start * 16:end * 16], -1, 1) * 32767).astype("<i2")
with wave.open(str(task / "2-53-context.wav"), "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(16000)
    output.writeframes(clip.tobytes())
result = {"scope": "DSP only; no human listening claimed", "audio_sha256": case["audio_sha256"],
          "reference_sha256": case["reference_sha256"], "reference_ms": evidence["reference_ms"],
          "boundary_ms": evidence["boundary"]["start_ms"], "signed_error_ms": evidence["signed_error_ms"],
          "geometry": evidence["geometry"], "signal_frames": frames, "clip_start_ms": start}
(task / "2-53-signal.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
for label, t in [("reference", result["reference_ms"]), ("midpoint", result["boundary_ms"])]:
    print(label, [f for f in frames if abs(f["start_ms"] - t) <= 30])
