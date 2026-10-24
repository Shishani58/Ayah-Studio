"""Draw the reported late cuts against real PCM, without model calls."""

import json
from pathlib import Path

import numpy as np

from surah_cutter.core import decode

TASK = Path(__file__).resolve().parent


def main():
    data = json.loads((TASK / 'baqara-replay.json').read_text())
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1200" viewBox="0 0 1200 1200">',
           '<rect width="1200" height="1200" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#18212b">',
           '<text x="40" y="36" font-size="24">Бахара: оставшиеся поздние границы</text>',
           '<text x="40" y="67" font-size="16">Зелёный — общая тишина двух окон. Синий — новая метка; оранжевый пунктир — старая.</text>']
    for i, ayah in enumerate((6, 8, 9, 12)):
        change = next(c for c in data['changes'] if c['ayah'] == ayah)
        start, end = change['pause_ms']
        old, new = change['before_ms'], change['after_ms']
        left, right = start - 450, old + 600
        pcm = decode.decode_range(data['audio'], start_ms=left, end_ms=right)
        peaks = np.max(np.abs(pcm[:len(pcm) // 80 * 80].reshape(-1, 80)), axis=1)
        scale = max(float(np.percentile(peaks, 98)), .001)
        top = 125 + i * 220

        def x(ms):
            return 60 + (ms - left) * 1080 / (right - left)

        upper, lower = [], []
        for j, peak in enumerate(peaks):
            height = min(float(peak) / scale, 1) * 42
            upper.append(f'{x(left + j * 5):.1f},{top + 78 - height:.1f}')
            lower.append(f'{x(left + j * 5):.1f},{top + 78 + height:.1f}')
        svg.append(f'<text x="40" y="{top}" font-size="18">Аят {ayah}: {old / 1000:.3f} → {new / 1000:.3f} с (−{old - new} мс)</text>')
        svg.append(f'<rect x="{x(start):.1f}" y="{top + 20}" width="{x(end) - x(start):.1f}" height="115" fill="#d9f4df"/>')
        svg.append(f'<polygon points="{" ".join(upper + lower[::-1])}" fill="#81909d"/>')
        for ms, color, dash in ((old, '#c26716', 'stroke-dasharray="6 4"'), (new, '#1264bf', '')):
            svg.append(f'<line x1="{x(ms):.1f}" x2="{x(ms):.1f}" y1="{top + 20}" y2="{top + 137}" stroke="{color}" stroke-width="3" {dash}/>')
        for ms in np.linspace(left, right, 5):
            svg.append(f'<text x="{x(ms):.1f}" y="{top + 158}" text-anchor="middle" font-size="13">{ms / 1000:.2f} с</text>')
    svg.append('</g></svg>')
    (TASK / 'waveform.svg').write_text('\n'.join(svg))


if __name__ == '__main__':
    main()
