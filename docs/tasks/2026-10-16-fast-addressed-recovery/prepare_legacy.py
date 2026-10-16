"""Historical search/acceptance only; current pinned Base runtime and DSP."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(roots['A']).with_name('legacy')
assert not root.exists()
shutil.copytree(roots['A'], root, ignore=shutil.ignore_patterns('__pycache__'))
files = ('align_fast.py', 'align_fast_whisper.py', 'fast_reads.py')
manifest = {}
for name in files:
    relative = 'surah_cutter/core/' + name
    data = subprocess.check_output(['git', 'show', 'aac8f3a:' + relative], cwd=P.parents[2])
    (root / relative).write_bytes(data)
    manifest[relative] = hashlib.sha256(data).hexdigest()
roots['legacy'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(roots, indent=2) + '\n')
(P / 'legacy-source.json').write_text(json.dumps({'ref': 'aac8f3a', 'sha256': manifest,
    'scope': 'historical search/proof, same current Base runtime/weights/DSP; not production'}, indent=2) + '\n')
