"""H9b: a complete spoken unit can cue resolving an ambiguous canonical word."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
paths = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(paths['baseline']).parent / 'prototype-b'
assert not root.exists()
shutil.copytree(paths['baseline'], root, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
f = root / 'surah_cutter/core/fast_executor.py'
source = f.read_text()
old = '    matched = _prefix(observation.units, phrase)\n    if groups is not None and matched > 0'
new = '''    matched = _prefix(observation.units, phrase)
    # The model can spell one canonical word as multiple independently spoken
    # words (for example vocative ya + ayyuha). A complete short spoken word is
    # a cue to resolve the remaining ambiguity, never a canonical-word proof.
    if (groups is not None and matched == 1 and len(phrase[0]) == 2 and ends[0] > 1
            and not observation.truncated and observation.unit_quality
            and math.isfinite(observation.unit_quality[0])
            and observation.unit_quality[0] >= threshold
            and not _strong_continuation_conflict(observation.units, observation.unit_quality,
                                                groups[ayah], matched, threshold)):
        return "whole_spoken_prefix_incomplete_canonical_word"
    if groups is not None and matched > 0'''
assert source.count(old) == 1
f.write_text(source.replace(old, new))
paths['prototype-b'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(paths, indent=2))
(P / 'h9b-source-sha.json').write_text(json.dumps({'executor_sha256': hashlib.sha256(f.read_bytes()).hexdigest()}, indent=2))
print(root)
