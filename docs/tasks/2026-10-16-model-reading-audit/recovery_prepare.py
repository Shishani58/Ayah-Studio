"""Read existing adjacent pause candidates for a future anchor, no oracle locator."""
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

assert not (P / 'recovery-inputs.json').exists()
baseline = ROOT / 'docs/tasks/2026-10-15-fast-systematic-execution/baseline-a2.json'
row = next(r for r in json.loads(baseline.read_text())
           if r['model'] == 'whisper-tarteel-base' and r['surah'] == 2)
path = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3')
pcm = decode.decode(str(path))
arrays, configs = {}, []
for read in row['trace']['acoustic_reads']:
    if not read['reason'].startswith('ayah=4:pause:'):
        continue
    for length in [1500, 4000]:
        start = read['window_ms'][0]
        end = start + length
        key = f'future-5-from-read-{read["id"]}-{length}'
        crop = pcm[round(start * 16):round(end * 16)].copy()
        arrays[key] = crop
        configs.append(dict(key=key, role='adjacent-recovery', surah=2, ayah=5,
                            **representation(2, 5), window_ms=[start, end],
                            source_read_id=read['id'], audio_path=str(path),
                            audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            pcm_sha256=hashlib.sha256(crop.tobytes()).hexdigest(), samples=len(crop)))
dump('recovery-inputs.json', dict(cases=configs,
    selection='Every existing ayah4 pause read; test immediate next ayah5 on same candidate',
    weights=str(model_store.installed_path(model_store.spec('whisper-tarteel-base')))))
np.savez(P / 'recovery-inputs.npz', **arrays)
print('prepared', len(configs))
