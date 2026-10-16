"""Fetch only four preselected small recordings; never touch existing DBs."""
import hashlib
import argparse
import json
from pathlib import Path
import time
import urllib.request

P = Path(__file__).resolve().parent
prior = P.parent / '2026-10-14-full-other-reciters/data-manifest.json'
source = json.loads(prior.read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--names', nargs='+', default=['saad_al_ghamidi', 'mahmoud_khalil_al_hussary'])
parser.add_argument('--label', default='transfer')
args = parser.parse_args()
names = args.names
manifest_path = P / f'{args.label}-manifest.json'
assert not manifest_path.exists(), 'Do not overwrite a previous download attempt'
manifest = []
for name in names:
    row = next(r for r in source['recordings'] if r['name'] == name)
    folder = Path(row['audio_dir'])
    for surah in [65, 73]:
        target = folder / f'{surah:03}.mp3'
        url = row['reciter']['url_template'].replace('{sura}', f'{surah:03}')
        entry = {'reciter': name, 'surah': surah, 'url': url, 'path': str(target),
                 'reference': row['reference'], 'selection': 'H9 confirmation fixed before results',
                 'started_unix': time.time()}
        manifest.append(entry)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        try:
            if target.exists():
                entry['outcome'] = 'reused-existing'
            else:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read(20 * 1024 * 1024 + 1)
                if len(data) > 20 * 1024 * 1024:
                    raise ValueError('selected recording exceeds20MiB cap')
                temp = target.with_suffix('.h9-download')
                temp.write_bytes(data)
                temp.rename(target)
                entry['outcome'] = 'downloaded'
            entry.update(bytes=target.stat().st_size,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        except Exception as error:
            entry.update(outcome='failed', error=repr(error))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(name, surah, entry['outcome'], entry.get('bytes'), flush=True)
