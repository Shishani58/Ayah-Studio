"""Assign whole unseen recordings before evaluating frozen variants."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

P = Path(__file__).resolve().parent
base = Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit')
reciters = {'saad_al_ghamidi': 'Saad-Al-Ghamidi',
           'mahmoud_khalil_al_hussary': 'Mahmoud_Khalil_Al-Hussary',
           'mohammed_siddiq_al_minshawi': 'Mohammed_Siddiq_Al-Minshawi'}
items = [{'reciter': name, 'surah': surah,
          'url': f'https://quran-audio.quranacademy.org/by-surah/{remote}/{surah:03}.mp3',
          'path': str(base / name / f'{surah:03}.mp3'),
          'role': 'whole-record control; no rule tuning after reading results',
          'assigned_unix': time.time()}
         for name, remote in reciters.items() for surah in (55, 79)]
manifest = P / 'control-manifest.json'
manifest.write_text(json.dumps({'initial_assignment_unavailable': [2, 3, 36, 67],
                               'frozen_rules': json.loads((P / 'variant-source-sha.json').read_text()),
                               'cases': items}, indent=2) + '\n')
for item in items:
    path = Path(item['path'])
    assert not path.exists(), path
    try:
        with urllib.request.urlopen(item['url'], timeout=60) as response:
            payload = response.read()
        path.write_bytes(payload)
        item.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), outcome='downloaded')
    except Exception as error:
        item.update(outcome='failed', error=str(error))
    manifest.write_text(json.dumps({'initial_assignment_unavailable': [2, 3, 36, 67],
        'frozen_rules': json.loads((P / 'variant-source-sha.json').read_text()), 'cases': items}, indent=2) + '\n')
    print(item['reciter'], item['surah'], item['outcome'], flush=True)
