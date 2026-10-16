"""Prepare blind human audio/waveform package. Does not create human labels."""
import csv
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

from surah_cutter.core.decode import decode
from surah_cutter.data.quran_text import surah_words

P = Path(__file__).resolve().parent
OUT = P / 'annotation-package'
OUT.mkdir(exist_ok=True)
prior = json.loads((P.parent / '2026-10-15-fast-next-iteration/after-raw.json').read_text())
selection = [('shatri', s, a) for s, a in [(1, 2), (73, 20), (77, 25), (79, 26), (80, 2), (112, 2)]]
# Other-reciter labels use the same canonical structural controls. Locate clips
# from independent DSP alone, not packaged reference or a newly seen error.
selection += [(name, s, 2) for name in ['saad_al_ghamidi', 'mohammed_siddiq_al_minshawi',
             'mahmoud_khalil_al_hussary', 'abdul_basit_mujawwad'] for s in [80, 98, 112]]
private, public = [], []
for number, (reciter, surah, ayah) in enumerate(selection, 1):
    case_id = f'C{number:02}'
    if reciter == 'shatri':
        audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri') / f'{surah:03}.mp3'
        old = next(c for c in prior if c['surah'] == surah and c['model'] == 'whisper-tarteel-base')
        center = next(row[1] for row in old['rows'] if row[0] == ayah)
        location_method = 'historical-prediction-selected-before-current-errors'
    else:
        from surah_cutter.core.candidates import pause_pipeline_decisions
        audio = Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / reciter / f'{surah:03}.mp3'
        pcm = decode(str(audio))
        decisions = pause_pipeline_decisions(pcm, 16000, surah=surah)
        # First non-intro strong DSP interval; listener must identify content.
        pauses = [d.pause for d in decisions if d.accepted and d.pause.center_ms > 4000]
        center = pauses[0].center_ms if pauses else 6000
        location_method = 'DSP-first-pause-after4s-owner-unverified'
    pcm = decode(str(audio))
    start, end = max(0, round((center - 3500) * 16)), min(len(pcm), round((center + 4500) * 16))
    clip = pcm[start:end]
    wav_path = OUT / f'{case_id}.wav'
    with wave.open(str(wav_path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes((np.clip(clip, -1, 1) * 32767).astype('<i2').tobytes())
    chunks = np.array_split(clip, min(1000, len(clip)))
    peaks = [float(np.max(np.abs(chunk))) for chunk in chunks]
    scale = max(peaks, default=1) or 1
    lines = ''.join(f'<line x1="{i}" x2="{i}" y1="{100-v/scale*95:.1f}" y2="{100+v/scale*95:.1f}"/>'
                    for i, v in enumerate(peaks))
    (OUT / f'{case_id}.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 220">'
        '<rect width="1000" height="220" fill="white"/><g stroke="#215a88" stroke-width="1">' + lines +
        f'</g><text x="10" y="215" font-size="14">{case_id}: 0–{len(clip)/16000:.3f}s (crop-relative)</text></svg>')
    target = ' '.join(w.text for w in surah_words(surah) if w.ayah == ayah)
    public.append({'case_id': case_id, 'requested_surah': surah, 'requested_ayah': ayah,
        'target': target, 'owner_verified': False, 'previous_speech_end_s': None,
        'physical_target_onset_s': None, 'playback_boundary_lower_s': None,
        'playback_boundary_upper_s': None, 'uncertainty_s': None, 'listener': None,
        'method': None, 'date': None, 'notes': None})
    private.append({'case_id': case_id, 'reciter': reciter, 'audio': str(audio),
        'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
        'samples': [start, end], 'crop_origin_ms': start / 16, 'location_method': location_method,
        'wav_sha256': hashlib.sha256(wav_path.read_bytes()).hexdigest()})
(P / 'annotation-private-manifest.json').write_text(json.dumps(private, ensure_ascii=False, indent=2) + '\n')
(OUT / 'labels.json').write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')
with (OUT / 'labels.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=public[0].keys())
    writer.writeheader()
    writer.writerows(public)
print('prepared', len(public), 'unlabelled clips; no listening claimed', flush=True)
