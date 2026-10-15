"""Isolate lexical normalization from the established duration-bearing tempo."""
from pathlib import Path
import shutil
import tempfile

task = Path(__file__).parent.resolve()
old = Path((task.parent / "2026-10-14-fast-muaalem-local-analysis/edge-snapshot-path.txt").read_text().strip())
source = Path(tempfile.mkdtemp(prefix="muaalem-raw-tempo-")) / "source"
shutil.copytree(old, source)
executor = source / "surah_cutter/core/fast_executor.py"
text = executor.read_text()
assert 'groups = {a: tuple(units) for a, units in groups.items()}' in text
text = text.replace('groups = {a: tuple(units) for a, units in groups.items()}',
                    'groups = {a: tuple(units) for a, units in groups.items()}\n'
                    '    tempo_groups = getattr(adapter, "tempo_groups", groups)')
# Only forecast weight calculations use raw phones. Proofs still use normalized
# independent observations, including the canonical-word identity lower bound.
for expression in (
    'for u in groups[a]', 'sum(map(len, groups[prev]))',
    'sum(map(len, groups[prior_ayah]))', 'sum(map(len, groups[ayah]))',
):
    assert expression in text, expression
    text = text.replace(expression, expression.replace('groups[', 'tempo_groups['))
text = text.replace('sequential-fast-v3-sparse-9-wasl-edge-probe',
                    'sequential-fast-v3-sparse-10-raw-tempo-probe')
executor.write_text(text)
(task / 'raw-tempo-snapshot-path.txt').write_text(str(source) + '\n')
print(source)
