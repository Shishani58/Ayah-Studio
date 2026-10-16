"""Read-only final guard for original canonical/reference/audio files."""
import hashlib
import json
from pathlib import Path
import sqlite3

P = Path(__file__).resolve().parent
PRIOR = P.parent / '2026-10-15-fast-systematic-execution'
manifest = json.loads((PRIOR / 'data-manifest.json').read_text())
for name in ['transfer-manifest.json', 'transfer-h9c-manifest.json']:
    for item in json.loads((P / name).read_text()):
        manifest[item['path']] = {'sha256': item['sha256'], 'bytes': item['bytes']}
errors = []
for name, expected in manifest.items():
    path = Path(name)
    actual = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    if actual != expected['sha256'] or path.stat().st_size != expected['bytes']:
        errors.append({'path': name, 'expected': expected, 'actual_sha256': actual,
                       'actual_bytes': path.stat().st_size})
repo = P.parents[2]
canonical = repo / 'data/quran_mushaf.db'
# Schema inspection is read-only, and canonical count/order is already exercised
# by the shared executor tests. This guard records immutable bytes, not labels.
with sqlite3.connect(f'file:{canonical}?mode=ro', uri=True) as con:
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
result = {'passed': not errors, 'files': len(manifest),
          'db_files': sum(f.endswith('.db') for f in manifest),
          'mp3_files': sum(f.endswith('.mp3') for f in manifest),
          'changed': errors, 'canonical_tables': tables,
          'AGENTS_CLAUDE_identical': (repo / 'AGENTS.md').read_bytes() == (repo / 'CLAUDE.md').read_bytes()}
targets = json.loads((P / 'pretransfer-text-data-guard.json').read_text())
result['target_tables_unchanged'] = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() == item['measured']
                                   for name, item in targets.items()}
result['passed'] &= all(result['target_tables_unchanged'].values())
(P / 'final-data-guard.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(result)
assert result['passed'] and result['AGENTS_CLAUDE_identical']
