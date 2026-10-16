"""The new short-word permission must not block existing candidate recovery."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
paths = json.loads((P / 'snapshot-paths.json').read_text())
assert not (P / 'wide-prototype-c-command.json').exists(), 'Never change a running candidate'
root = Path(paths['baseline']).parent / 'prototype-d'
assert not root.exists()
shutil.copytree(paths['prototype-c'], root, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
f = root / 'surah_cutter/core/fast_executor.py'
source = f.read_text()
old = '                had_prefix = False\n                speech_steps = list(policy.speech_ms)'
new = '''                had_prefix = False
                soft_short_word_route = False
                short_word_base_plan, short_word_base_phrase = plan, phrase
                speech_steps = list(policy.speech_ms)'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '                    current_expansion = bool(expansion)\n'
new = '''                    soft_short_word_route |= expansion == "whole_spoken_prefix_incomplete_canonical_word"
                    current_expansion = bool(expansion) and not soft_short_word_route
'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '                if boundary or blocked_prefix or future:\n                    break\n'
new = '''                if soft_short_word_route and boundary is None and future is None:
                    # This additional permission is provisional. Its failed
                    # reads cannot turn the original incomplete opening into
                    # a sticky block that hides a later independent anchor.
                    blocked_prefix = False
                    plan, phrase = short_word_base_plan, short_word_base_phrase
                    event("EVALUATE", ayah, candidate=index,
                          reason="short_word_extension_unresolved_resume_candidates")
                if boundary or blocked_prefix or future:
                    break
'''
assert source.count(old) == 1
source = source.replace(old, new)
f.write_text(source)
paths['prototype-d'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(paths, indent=2))
(P / 'h9d-source-sha.json').write_text(json.dumps({'executor_sha256': hashlib.sha256(f.read_bytes()).hexdigest()}, indent=2))
print(root)
