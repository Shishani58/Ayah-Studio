"""Freeze dirty sources and read-only guards without touching user data."""
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

P = Path(__file__).resolve().parent
REPO = P.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    assert not (P / 'checkpoint.json').exists(), 'Do not overwrite checkpoint'
    assert (REPO / 'AGENTS.md').read_bytes() == (REPO / 'CLAUDE.md').read_bytes()
    for args, name in [(['git', 'status', '--short'], 'initial-status.log'),
                       (['git', 'diff', '--binary'], 'initial.diff')]:
        (P / name).write_bytes(subprocess.check_output(args, cwd=REPO))
    root = Path(tempfile.mkdtemp(prefix='fast-systematic-')).resolve()
    manifests = {}
    for variant in ['baseline', 'prototype']:
        dest = root / variant
        dest.mkdir()
        for folder in ['surah_cutter', 'tools', 'data']:
            shutil.copytree(REPO / folder, dest / folder,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ['pyproject.toml', 'AGENTS.md', 'CLAUDE.md']:
            shutil.copy2(REPO / name, dest / name)
        manifests[variant] = {str(f.relative_to(dest)): sha(f)
                              for f in sorted(dest.rglob('*')) if f.is_file()}
        with zipfile.ZipFile(P / f'{variant}-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for name in manifests[variant]:
                z.write(dest / name, name)
    dump('source-manifests.json', manifests)
    data = list((REPO / 'data').rglob('*.db'))
    cache = Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit')
    data += list(cache.rglob('*.db'))
    audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri')
    data += list(audio.glob('*.mp3'))
    data += list(cache.rglob('*.mp3'))
    dump('data-manifest.json', {str(f): {'sha256': sha(f), 'bytes': f.stat().st_size}
                                for f in sorted(set(data))})
    dump('snapshot-paths.json', {v: str(root / v) for v in manifests})
    dump('checkpoint.json', {'stage': 'M0', 'head': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'roots': {v: str(root / v) for v in manifests}, 'processes': [],
        'hypotheses': {}, 'next': 'Compare source to measured sparse12; fresh baseline.'})
    print(root, flush=True)
