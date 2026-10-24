"""Plot actual PCM envelopes and saved old/new cuts for the reported boundaries."""

import html
import json
from pathlib import Path

import numpy as np

from surah_cutter.core import decode

TASK = Path(__file__).resolve().parent


def main():
    result = json.loads((TASK / 'baqara-replay.json').read_text())
    pcm = decode.decode_range(result['audio'], start_ms=204000, end_ms=255000)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1200" viewBox="0 0 1200 1200">',
           '<rect width="1200" height="1200" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#18212b">',
           '<text x="40" y="36" font-size="24">Аль-Бакара: границы перед аятами 15–18</text>',
           '<text x="40" y="67" font-size="16">Оранжевый пунктир — было; синий — стало; зелёный фон — DSP-пауза.</text>',
           '<text x="40" y="91" font-size="16">Серая волна — реальные пиковые амплитуды каждых 5 мс.</text>']
    for i, ayah in enumerate([15, 16, 17, 18]):
        change = next(c for c in result['changes'] if c['ayah'] == ayah)
        pause_start, pause_end = change['pause_ms']
        old, new = change['before_ms'], change['after_ms']
        left, right = pause_start - 450, old + 600
        top = 135 + i * 210

        def x(ms):
            return 60 + (ms - left) * 1080 / (right - left)

        lo, hi = round((left - 204000) * 16), round((right - 204000) * 16)
        samples = pcm[lo:hi]
        peak = np.max(np.abs(samples[:len(samples) // 80 * 80].reshape(-1, 80)), axis=1)
        scale = max(float(np.percentile(peak, 98)), .001)
        upper, lower = [], []
        for j, amplitude in enumerate(peak):
            xx = x(204000 + lo / 16 + j * 5)
            height = min(float(amplitude) / scale, 1) * 42
            upper.append(f'{xx:.1f},{top + 78 - height:.1f}')
            lower.append(f'{xx:.1f},{top + 78 + height:.1f}')
        label = f'Аят {ayah}: {old / 1000:.3f} → {new / 1000:.3f} с (−{old - new} мс)'
        svg.append(f'<text x="40" y="{top}" font-size="18">{html.escape(label)}</text>')
        svg.append(f'<rect x="{x(pause_start):.1f}" y="{top + 20}" width="{x(pause_end) - x(pause_start):.1f}" height="115" fill="#d9f4df"/>')
        svg.append(f'<polygon points="{" ".join(upper + lower[::-1])}" fill="#81909d"/>')
        for ms, color, dash in ((old, '#c26716', 'stroke-dasharray="6 4"'), (new, '#1264bf', '')):
            svg.append(f'<line x1="{x(ms):.1f}" x2="{x(ms):.1f}" y1="{top + 20}" y2="{top + 137}" stroke="{color}" stroke-width="3" {dash}/>')
        for ms in np.linspace(left, right, 5):
            svg.append(f'<text x="{x(ms):.1f}" y="{top + 158}" text-anchor="middle" font-size="13">{ms / 1000:.2f} с</text>')
    svg.append('</g></svg>')
    (TASK / 'waveform.svg').write_text('\n'.join(svg))


if __name__ == '__main__':
    main()
