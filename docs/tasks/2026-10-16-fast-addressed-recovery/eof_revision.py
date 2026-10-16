"""Stop the new optional route when EOF cannot supply any additional samples."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(roots['B4']).with_name('B5')
assert not root.exists()
shutil.copytree(roots['B4'], root, ignore=shutil.ignore_patterns('__pycache__'))
file = root / 'surah_cutter/core/fast_executor.py'
code = file.read_text()
needle = '                    if identity in tried:\n                        raise RuntimeError("repeated_decision_without_new_data")'
assert needle in code
code = code.replace(needle, '''                    if identity in tried:
                        if partial_head_route is not None:
                            reason = "insufficient_local_context"
                            event("EVALUATE", ayah, candidate=index,
                                  reason="addressed_context_exhausted_at_audio_end")
                            break
                        raise RuntimeError("repeated_decision_without_new_data")''', 1)
file.write_text(code)
roots['B5'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(roots, indent=2) + '\n')
(P / 'B5-sha.json').write_text(json.dumps({'sha256': hashlib.sha256(code.encode()).hexdigest(),
    'parent': 'B4', 'change': 'new route only: identical EOF-clamped window leaves review, no duplicate inference',
    'completed_B4_traces_equivalence': 'guard unreachable in successfully completed B4 traces; replay required'}, indent=2) + '\n')
