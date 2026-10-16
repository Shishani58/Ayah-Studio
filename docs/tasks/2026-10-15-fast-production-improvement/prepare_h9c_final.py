"""Minimal H9c: decline the new permission before spending extra reads."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
paths = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(paths['baseline']).parent / 'prototype-c3'
assert not root.exists()
assert not (P / 'h9c-development-command.json').exists(), 'H9c must not be running'
shutil.copytree(paths['prototype-b'], root, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
f = root / 'surah_cutter/core/fast_executor.py'
source = f.read_text()
source = source.replace('class Policy:\n', 'class Policy:\n    short_word_pause_radius_ms: float = 500.\n', 1)
old = '''                                if anchor:
                                    future_expansion = (future_ayah, anchor)'''
new = '''                                if (anchor and (anchor != "whole_spoken_prefix_incomplete_canonical_word"
                                        or (pause.end_ms - pause.start_ms) / 2 <= policy.short_word_pause_radius_ms)):
                                    future_expansion = (future_ayah, anchor)'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '                    expansion = _expansion_evidence(obs, phrase, word_ends[ayah], threshold, groups, ayah, word_ends)\n'
new = old + '''                    if (expansion == "whole_spoken_prefix_incomplete_canonical_word" and pause
                            and (pause.end_ms - pause.start_ms) / 2 > policy.short_word_pause_radius_ms):
                        event("SEARCH_CUE_ONLY", ayah, read_id=opening_id,
                              reason="short_word_pause_time_uncertain", observed=obs.units,
                              pause_ms=[pause.start_ms, pause.end_ms])
                        expansion = None
'''
assert source.count(old) == 1
source = source.replace(old, new)
f.write_text(source)
paths['prototype-c'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(paths, indent=2))
(P / 'h9c-final-source-sha.json').write_text(json.dumps({'executor_sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
    'reason': 'Decline new permission before extra reads, preserving baseline candidate/budget routes.'}, indent=2))
print(root)
