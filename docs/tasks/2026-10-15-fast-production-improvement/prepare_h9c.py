"""Separate identity from new pause-time claims after the H9 short-word cue."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
paths = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(paths['baseline']).parent / 'prototype-c2'
assert not root.exists()
shutil.copytree(paths['prototype-b'], root, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
f = root / 'surah_cutter/core/fast_executor.py'
source = f.read_text()
source = source.replace('class Policy:\n', 'class Policy:\n    short_word_pause_radius_ms: float = 500.\n', 1)
old = '                had_prefix = False\n                speech_steps = list(policy.speech_ms)'
new = '                had_prefix = False\n                short_word_cue_read_id = None\n                speech_steps = list(policy.speech_ms)'
assert source.count(old) == 1
source = source.replace(old, new)
old = '\n                        proof = {"phrase": phrase, "complete": True, "ambiguous": False,\n'
new = '''
                        if (pause and short_word_cue_read_id is not None
                                and (pause.end_ms - pause.start_ms) / 2 > policy.short_word_pause_radius_ms):
                            event("IDENTITY_ONLY", ayah, read_id=opening_id,
                                  cue_read_id=short_word_cue_read_id,
                                  reason="short_word_pause_time_uncertain", phrase=phrase,
                                  pause_ms=[pause.start_ms, pause.end_ms],
                                  interval_radius_ms=(pause.end_ms - pause.start_ms) / 2)
                            reason = "short_word_pause_time_uncertain"
                            blocked_prefix = True
                            break
                        proof = {"phrase": phrase, "complete": True, "ambiguous": False,
'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '                        if noisy_proof:\n                            proof.update(noisy_proof)'
new = '''                        if short_word_cue_read_id is not None:
                            proof["short_word_cue_read_id"] = short_word_cue_read_id
                        if noisy_proof:
                            proof.update(noisy_proof)'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '                    current_expansion = bool(expansion)\n'
new = '''                    if expansion == "whole_spoken_prefix_incomplete_canonical_word":
                        short_word_cue_read_id = opening_id
                    current_expansion = bool(expansion)
'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '            r0, r1 = read["window_ms"]\n            expected_left = max(left, right - 200.)'
new = '''            if "short_word_cue_read_id" in proof:
                from surah_cutter.core.fast_adapters import Observation
                cue_id = proof["short_word_cue_read_id"]
                if (type(cue_id) is not int or not 0 <= cue_id < boundary.read_id
                        or (right - left) / 2 > policy.short_word_pause_radius_ms
                        or groups is None or word_ends is None):
                    raise ValueError("invalid_short_word_cue_proof")
                cue = reads[cue_id]
                cue_data = cue.get("observed", {})
                cue_obs = Observation(**{**cue_data, "units": tuple(cue_data.get("units", ())),
                                         "unit_quality": tuple(cue_data.get("unit_quality", ()))})
                if (cue.get("outcome") != "completed"
                        or cue["window_ms"][0] != read["window_ms"][0]
                        or cue["window_ms"][1] >= read["window_ms"][1]
                        or _expansion_evidence(cue_obs, phrase, word_ends[boundary.ayah], threshold,
                            groups, boundary.ayah, word_ends)
                           != "whole_spoken_prefix_incomplete_canonical_word"):
                    raise ValueError("unheard_short_word_cue")
            r0, r1 = read["window_ms"]
            expected_left = max(left, right - 200.)'''
assert source.count(old) == 1
source = source.replace(old, new)
f.write_text(source)
paths['prototype-c'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(paths, indent=2))
(P / 'h9c-source-sha.json').write_text(json.dumps({'executor_sha256': hashlib.sha256(f.read_bytes()).hexdigest()}, indent=2))
print(root)
