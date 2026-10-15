"""Retain reference, native proof and signal separately; never infer reference sign."""
import hashlib
import json
from pathlib import Path
import sqlite3
import wave

import numpy as np

from surah_cutter.core.decode import decode
from surah_cutter.data import quran_text

task = Path(__file__).parent.resolve()
root = task.parents[2]
reference = root / 'data/references/abu_bakr_ash_shatri.db'
case = json.loads((task / 'native-onset-2.json').read_text())[0]
audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3')
assert hashlib.sha256(audio.read_bytes()).hexdigest() == case['audio_sha256']
assert hashlib.sha256(reference.read_bytes()).hexdigest() == case['reference_sha256']
with sqlite3.connect(f'file:{reference.resolve()}?mode=ro', uri=True) as connection:
    truth = dict(connection.execute('SELECT ayah,time FROM timings WHERE sura=2'))
pcm = decode(str(audio))
words = quran_text.surah_words(2)
outliers = []
for ayah, start_ms, confidence in case['rows']:
    if confidence <= 0 or abs(start_ms - truth[ayah]) <= 500:
        continue
    boundary = next(b for b in case['trace']['boundaries'] if b['ayah'] == ayah)
    read = case['trace']['acoustic_reads'][boundary['read_id']]
    left, right = boundary['proof']['pause_ms']
    frames = []
    for t in range(int(left), int(right) + 600, 20):
        samples = pcm[t * 16:(t + 20) * 16]
        frames.append({'start_ms': t, 'rms': float(np.sqrt(np.mean(samples * samples))),
                       'peak': float(np.max(np.abs(samples)))})
    clip_start, clip_end = int(left) - 1000, int(right) + 2500
    clip = (np.clip(pcm[clip_start * 16:clip_end * 16], -1, 1) * 32767).astype('<i2')
    with wave.open(str(task / f'2-{ayah}-context.wav'), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(clip.tobytes())
    outliers.append({'ayah': ayah, 'canonical_opening': [w.text for w in words if w.ayah == ayah][:4],
                     'boundary': boundary, 'read': read, 'reference_ms': truth[ayah],
                     'signed_error_ms': start_ms - truth[ayah],
                     'pause_geometry': case['trace']['pause_geometry'][boundary['proof']['pause_id']],
                     'reference_frames': [f for f in frames if abs(f['start_ms'] - truth[ayah]) <= 40],
                     'native_frames': [f for f in frames if abs(f['start_ms'] - start_ms) <= 40],
                     'signal_frames': frames, 'audio_clip_start_ms': clip_start})
result = {'audio_sha256': case['audio_sha256'], 'reference_sha256': case['reference_sha256'],
          'outliers': outliers, 'scope': 'DSP only; no model calls; no human listening claimed'}
(task / 'native-onset-2-outlier-evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print([(o['ayah'], o['signed_error_ms'], o['canonical_opening']) for o in outliers])
