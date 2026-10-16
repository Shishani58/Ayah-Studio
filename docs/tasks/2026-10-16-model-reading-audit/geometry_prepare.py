"""Probe every rejected-geometry unresolved candidate in the frozen Base trace."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
from deep_probe import ROOT, dump, representation  # noqa: E402

sys.path.insert(0, str(ROOT))
from surah_cutter.core import decode, model_store  # noqa: E402

assert not (P / 'geometry-inputs.json').exists()
baseline = ROOT / 'docs/tasks/2026-10-15-fast-systematic-execution/baseline-a2.json'
configs, arrays = [], {}
for row in json.loads(baseline.read_text()):
    if row['model'] != 'whisper-tarteel-base':
        continue
    unresolved = {a for a, _, q in row['rows'] if q == 0}
    for event in row['trace']['transitions']:
        if not event.get('declined_geometry') or event.get('ayah') not in unresolved:
            continue
        surah, ayah = row['surah'], event['ayah']
        pause = row['trace']['phase1_pauses'][event['candidate']]
        path = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri') / f'{surah:03}.mp3'
        pcm = decode.decode(str(path))
        start = max(pause['start_ms'], pause['end_ms'] - 200)
        for length in [1500, 4000]:
            key = f'rejected-{surah}-{ayah}-{length}'
            end = min(len(pcm) / 16, start + length)
            crop = pcm[round(start * 16):round(end * 16)].copy()
            assert len(crop) <= 96000 and len(crop) < len(pcm)
            arrays[key] = crop
            configs.append(dict(key=key, role='rejected-geometry', surah=surah, ayah=ayah,
                                **representation(surah, ayah), event=event, pause=pause,
                                window_ms=[start, end], audio_path=str(path),
                                audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                pcm_sha256=hashlib.sha256(crop.tobytes()).hexdigest(), samples=len(crop)))
for surah in [1, 2]:
    source = next(c for c in configs if c['surah'] == surah)
    key = f'zero-{surah}'
    arrays[key] = np.zeros(24000, np.float32)
    configs.append(dict(key=key, role='negative', surah=surah, ayah=4,
                        **representation(surah, 4), samples=24000))
dump('geometry-inputs.json', dict(cases=configs,
    selection='All unresolved declined_geometry events in frozen Base12; no reference read',
    baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    weights=str(model_store.installed_path(model_store.spec('whisper-tarteel-base')))))
np.savez(P / 'geometry-inputs.npz', **arrays)
print('prepared', len(configs))
