"""Standalone SVG waveform envelopes: user run versus fresh FULL."""
import html
import json
from pathlib import Path

import numpy as np

from surah_cutter.core import decode

TASK = Path(__file__).resolve().parent
rows = json.loads((TASK / "comparison.json").read_text())["comparison"][:10]
pcm = decode.decode_range(str(Path.home() / "Desktop/Abu_Bakr_Ash-Shatri/002.mp3"),
                          start_ms=0, end_ms=145000)
for start in (0, 5):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1200" viewBox="0 0 1200 1200">',
           '<rect width="1200" height="1200" fill="#fff"/>',
           '<g font-family="Arial,sans-serif" fill="#18212b">',
           '<text x="40" y="36" font-size="24">Бахара — первые границы FULL</text>',
           '<text x="40" y="65" font-size="15">Серая волна: пиковая огибающая, 5 мс. Зелёный фон: выбранная тишина.</text>',
           '<text x="40" y="89" font-size="15">Оранжевая пунктирная линия: предыдущий прогон. Синяя: новый.</text>']
    for i, row in enumerate(rows[start:start + 5]):
        old, new = row["old_ms"], row["new_ms"]
        left = min(old, new, row["pause_start_ms"] or new) - 400
        right = max(old, new, row["pause_end_ms"] or new) + 600
        top = 120 + i * 210
        center = top + 70

        def x(ms):
            return 60 + (ms - left) / (right - left) * 1080

        a, b = max(0, round(left * 16)), min(len(pcm), round(right * 16))
        samples = pcm[a:b]
        frame = 80
        n = len(samples) // frame
        peak = np.max(np.abs(samples[:n * frame].reshape(n, frame)), axis=1)
        scale = max(float(np.percentile(peak, 98)), .001)
        upper, lower = [], []
        for j, amplitude in enumerate(peak):
            xx = x(a / 16 + j * 5)
            height = min(float(amplitude) / scale, 1) * 34
            upper.append(f"{xx:.1f},{center - height:.1f}")
            lower.append(f"{xx:.1f},{center + height:.1f}")
        label = f"Аят {row['ayah']}   было {old / 1000:.3f} с → стало {new / 1000:.3f} с"
        svg.append(f'<text x="40" y="{top + 5}" font-size="17">{html.escape(label)}</text>')
        if row["pause_start_ms"] is not None:
            p0, p1 = row["pause_start_ms"], row["pause_end_ms"]
            svg.append(f'<rect x="{x(p0):.1f}" y="{top + 23}" width="{x(p1) - x(p0):.1f}" height="95" fill="#d9f4df"/>')
            detail = f"Тишина {(p1 - p0) / 1000:.3f} с; до её конца {p1 - new:.0f} мс"
            svg.append(f'<text x="1140" y="{top + 5}" text-anchor="end" font-size="14">{detail}</text>')
        svg.append(f'<polygon points="{" ".join(upper + lower[::-1])}" fill="#81909d"/>')
        for ms, color, dash in ((old, "#c26716", 'stroke-dasharray="6 4"'), (new, "#1264bf", "")):
            svg.append(f'<line x1="{x(ms):.1f}" x2="{x(ms):.1f}" y1="{top + 20}" y2="{top + 122}" stroke="{color}" stroke-width="2.5" {dash}/>')
        for ms in np.linspace(left, right, 5):
            svg.append(f'<text x="{x(ms):.1f}" y="{top + 143}" text-anchor="middle" font-size="12">{ms / 1000:.2f} с</text>')
    svg.append("</g></svg>")
    (TASK / f"waveforms-{start + 1}-{start + 5}.svg").write_text("\n".join(svg))
