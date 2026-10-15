"""Pair current production with exactly the isolated negative candidate predicate."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

task = Path(__file__).parent.resolve()
root = task.parents[2]
prototype = Path((task / 'snapshot-path.txt').read_text().strip())
text = (prototype / 'surah_cutter/core/fast_executor.py').read_text()
start = text.index('                    if (confirmed and confirmed[-1].ayah == ayah - 1 and not obs.truncated')
end = text.index('                    future_expansion = None', start)
predicate = text[start:end]
assert 'decoded_previous_ayah_body' in predicate
folder = Path(tempfile.mkdtemp(prefix='fast-body-supported-'))
sources = {}
for variant in ['before', 'after']:
    source = folder / variant
    source.mkdir()
    for group in ['surah_cutter', 'tools']:
        shutil.copytree(root / group, source / group, ignore=shutil.ignore_patterns('__pycache__'))
    for group in ['data', 'models']:
        (source / group).symlink_to(root / group, target_is_directory=True)
    shutil.copy2(root / 'pyproject.toml', source / 'pyproject.toml')
    file = source / 'surah_cutter/core/fast_executor.py'
    current = file.read_text()
    assert 'sequential-fast-v3-sparse-9' in current
    if variant == 'after':
        marker = '                    future_expansion = None\n'
        assert current.count(marker) == 1
        current = current.replace(marker, predicate + marker)
        current = current.replace('sequential-fast-v3-sparse-9', 'sequential-fast-v3-sparse-10-previous-body-probe')
        file.write_text(current)
    manifest = {}
    with zipfile.ZipFile(task / f'supported-{variant}-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        files = [p for group in ['surah_cutter', 'tools'] for p in (source / group).rglob('*.py')]
        files.append(source / 'pyproject.toml')
        for p in files:
            relative = p.relative_to(source).as_posix()
            archive.write(p, relative)
            manifest[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    (task / f'supported-{variant}-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    sources[variant] = str(source)
(task / 'supported-snapshot-paths.json').write_text(json.dumps(sources, indent=2) + '\n')
print(sources)
