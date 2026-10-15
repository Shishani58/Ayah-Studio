"""Reviewable before/after clips; no inference or reference fitting."""
import base64
import io
import json
from pathlib import Path
import sys
import wave
import numpy as np
TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK.parents[2]))
from surah_cutter.core import decode
cases = json.loads((TASK / "qpc-probe.json").read_text())
cache = Path.home() / ".cache/surah-cutter/full-boundary-proof/audio"
cache.mkdir(parents=True, exist_ok=True)
items = [
    ("Фатиха Шатри: конец первого аята до", "abu_bakr_ash_shatri", 1, 120, 5339),
    ("Фатиха Шатри: конец первого аята после", "abu_bakr_ash_shatri", 1, 120, 6483),
    ("Фатиха Шатри: второй аят до", "abu_bakr_ash_shatri", 1, 5339, 11598),
    ("Фатиха Шатри: второй аят после", "abu_bakr_ash_shatri", 1, 6483, 11598),
    ("Mujawwad 98:7 до", "abdul_basit_mujawwad", 98, 166571, 196889),
    ("Mujawwad 98:7 после", "abdul_basit_mujawwad", 98, 173500, 196889),
    ("Шатри 67:30, QPC: первая фраза", "abu_bakr_ash_shatri", 67, 357398, 364398),
    ("Шатри 67:30, Muaalem: поздняя фраза", "abu_bakr_ash_shatri", 67, 371857, 378857),
]
html = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>FULL: границы до/после</title>',
        '<style>body{font:17px system-ui;max-width:850px;margin:40px auto;padding:0 20px;background:#f5f5f5;color:#222}article{background:white;padding:20px;margin:20px 0;border-radius:12px}audio{width:100%}p{line-height:1.5}</style>',
        '<h1>FULL: границы до/после</h1><p>Фатиха: 5,339 → 6,483 с. Mujawwad 98:7: 166,571 → 173,500 с. Последнее расхождение с GT увеличилось: эталон 169,383 с внутри тишины 168,340–173,600 с. Новый срез сохраняет окончание шестого аята в предыдущем клипе. Автоматическая проверка фраз выполнена; человеческое прослушивание не заявляется.</p>',
        '<p>Шатри 67:30: два фрагмента для проверки повторения. Независимое ASR распознаёт одну и ту же начальную фразу в обоих; Muaalem FULL выбирает поздний участок. Нужна слуховая оценка полного контекста.</p>']
for i, (title, reciter, surah, lo, hi) in enumerate(items):
    case = next(c for c in cases if c['reciter'] == reciter and c['surah'] == surah)
    pcm = decode.decode_range(case['audio'], start_ms=lo, end_ms=hi)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(decode.DEFAULT_SR)
        wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())
    payload = buf.getvalue()
    (cache / f'{i + 1}.wav').write_bytes(payload)
    html.append(f'<article><h2>{title}</h2><p>{lo / 1000:.3f}–{hi / 1000:.3f} с</p><audio controls preload="none" src="data:audio/wav;base64,{base64.b64encode(payload).decode()}"></audio></article>')
html.append('</html>')
(TASK / 'examples.html').write_text('\n'.join(html))
print(cache)
