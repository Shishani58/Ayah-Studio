"""Fresh whole-record holdout after candidate selection; previous55/79 now development."""
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
cases = [{'reciter': name, 'surah': n, 'path': str(base/name/f'{n:03}.mp3'),
    'url': f'https://quran-audio.quranacademy.org/by-surah/{remote}/{n:03}.mp3',
    'role': 'final holdout; no rule tuning', 'assigned_unix': time.time()}
    for name, remote in reciters.items() for n in (67, 77)]
manifest = {'selected': json.loads((P / 'selected-candidate.json').read_text()),
            'old_controls': '55/79 became development after B6 control failure / candidate selection', 'cases': cases}
out = P / 'final-control-manifest.json'
out.write_text(json.dumps(manifest, indent=2) + '\n')
for case in cases:
    path = Path(case['path'])
    assert not path.exists(), path
    try:
        with urllib.request.urlopen(case['url'], timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
        case.update(outcome='downloaded', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        case.update(outcome='failed', error=str(error))
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    print(case['reciter'], case['surah'], case['outcome'], flush=True)
