"""Read-only local waveform probe for the first ten boundaries of the saved run."""
import json
from pathlib import Path

from surah_cutter.core import candidates, decode

root = Path.home() / "Desktop/Abu_Bakr_Ash-Shatri"
saved = json.loads((root / "Abu_Bakr_Ash-Shatri.logs/sura_002_full.json").read_text())
pcm = decode.decode_range(str(root / "002.mp3"), start_ms=0, end_ms=150000)
for i, c in enumerate(saved["trace"]["ayahs"][:10]):
    first = c["full_ctc_first_ms"]
    tail = saved["trace"]["ayahs"][i - 1]["full_ctc_end_ms"] if i else first - 2000
    lo = max(0, first - 6000, tail - 800)
    hi = min(len(pcm) / 16, first + 1000)
    crop = pcm[round(lo * 16):round(hi * 16)]
    found = []
    for p in candidates.detect_energy_pauses(crop, 16000, min_pause_ms=120,
                                             frame_ms=10, smooth_frames=3,
                                             use_ten_vad=False):
        metrics = candidates._dsp_pause_metrics(crop, 16000, p)
        found.append(dict(start=round(lo + p.start_ms), end=round(lo + p.end_ms),
                          accepted=candidates._dsp_accepts_pause(metrics), metrics=metrics))
    print(json.dumps(dict(ayah=c["ayah"], tail=round(tail), first=round(first), pauses=found)))
