"""Read-only measurements of every saved Fatiha cut; no inference."""
import json
from pathlib import Path

import numpy as np

from surah_cutter.core import decode

TASK = Path(__file__).resolve().parent
pcm = decode.decode(Path.home() / "Desktop/Abu_Bakr_Ash-Shatri/001.mp3")
trace = json.loads((TASK / "user-full.json").read_text())["trace"]
rows = trace["ayahs"]
measurements = []
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1200">',
       '<rect width="100%" height="100%" fill="white"/>']
for i, row in enumerate(rows[1:]):
    cut = row["onset_ms"]
    lo, hi = cut - 800, cut + 500
    frames = pcm[lo * 16:hi * 16].reshape(-1, 80)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peaks = np.max(np.abs(frames), axis=1)
    times = lo + np.arange(len(frames)) * 5
    scale = max(float(np.percentile(peaks, 95)), 1e-8)
    y = 60 + i * 185

    def x(ms):
        return 70 + (ms - lo)

    svg.append(f'<text x="30" y="{y - 25}" font-family="Arial" font-size="20">'
               f'1:{row["ayah"] - 1} → 1:{row["ayah"]}   {cut} ms</text>')
    for ms, a in zip(times, peaks):
        h = min(float(a) / scale, 1) * 48
        svg.append(f'<line x1="{x(ms)}" x2="{x(ms)}" y1="{y + 50 - h}" '
                   f'y2="{y + 50 + h}" stroke="#65717d"/>')
    for ms in range(lo, hi + 1, 100):
        svg.append(f'<text x="{x(ms)}" y="{y + 120}" text-anchor="middle" '
                   f'font-family="Arial" font-size="11">{ms}</text>')
    svg.append(f'<path d="M{x(cut)} {y}v100" stroke="#d62728" stroke-width="2"/>')
    measurements.append(dict(ayah=row["ayah"], cut_ms=cut, times_ms=times.tolist(),
                             rms=rms.tolist(), peak=peaks.tolist()))
svg.append('</svg>')
(TASK / "waveforms.svg").write_text("\n".join(svg))
(TASK / "signal.json").write_text(json.dumps(measurements) + "\n")
# A small sample near each cut lets the browser test measure decoder offset.
windows = [dict(start_ms=r["onset_ms"] - 100,
                samples=pcm[(r["onset_ms"] - 100) * 16:(r["onset_ms"] + 100) * 16].tolist())
           for r in rows[1:]]
(TASK / "decoder-windows.json").write_text(json.dumps(windows) + "\n")
print(json.dumps(dict(duration_ms=len(pcm) / 16, cuts=[r["onset_ms"] for r in rows])))
