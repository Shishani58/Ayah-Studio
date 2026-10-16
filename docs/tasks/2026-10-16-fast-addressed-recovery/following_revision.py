"""One bounded DSP hypothesis after an exhausted nonempty tempo region, no new scanner."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
for parent, variant in [('A', 'S'), ('B5', 'BS')]:
    root = Path(roots[parent]).with_name(variant)
    assert not root.exists()
    shutil.copytree(roots[parent], root, ignore=shutil.ignore_patterns('__pycache__'))
    file = root / 'surah_cutter/core/fast_executor.py'
    code = file.read_text()
    code = code.replace('            forecast = None\n', '            forecast = None\n            following_probe_id = None\n', 1)
    code = code.replace('                if not candidates_here and not adapter.timing:',
        '''                if (not adapter.timing and (not candidates_here
                        or (adapter.model_id == "whisper-tarteel-base" and len(candidates_here) < policy.max_candidates))):''', 1)
    code = code.replace('and geometry[c[1]]["accepted"]), None)',
        '''and geometry[c[1]]["accepted"]
                                      and (not candidates_here or geometry[c[1]]["midpoint_interval_radius_ms"]
                                           <= policy.short_word_pause_radius_ms)), None)''', 1)
    needle = '                        candidates_here = [following]\n'
    assert needle in code
    code = code.replace(needle, '''                        if candidates_here:
                            following_probe_id = following[1]
                        candidates_here.append(following)
''', 1)
    code = code.replace('reason="empty_tempo_region_next_dsp_pause", region_ms=[left, right],',
        '''reason=("empty_tempo_region_next_dsp_pause" if following_probe_id is None
                                      else "optional_after_unproved_tempo_candidates"), region_ms=[left, right],''', 1)
    code = code.replace('                # Geometry belongs to DSP.',
        '                following_probe = index == following_probe_id\n                # Geometry belongs to DSP.', 1)
    needle = '                        if noisy_proof:\n                            proof.update(noisy_proof)'
    assert needle in code
    code = code.replace(needle, '''                        if following_probe:
                            proof["bounded_following_origin_ayah"] = ayah
''' + needle, 1)
    # Apply to the full future proof, not just the cue permitting an expansion.
    needle = 'and not next_obs.truncated and quality >= threshold and not conflict'
    assert needle in code
    code = code.replace(needle, needle + '\n                                    and (not following_probe or future_ayah - ayah <= policy.expansion_lookahead)', 1)
    needle = '                                future = (future_ayah, next_phrase, pause, index, opening_id, quality, future_proof)'
    assert needle in code
    code = code.replace(needle, '''                                if following_probe:
                                    future_proof["bounded_following_origin_ayah"] = ayah
''' + needle, 1)
    if variant == 'BS':
        code = code.replace('recovered |= "partial_asr_head_cue_id" in proof',
            'recovered |= "partial_asr_head_cue_id" in proof or "bounded_following_origin_ayah" in proof', 1)
    needle = '        recovery = proof.get("opening_recovery")'
    code = code.replace(needle, '''        if "bounded_following_origin_ayah" in proof:
            origin = proof["bounded_following_origin_ayah"]
            if type(origin) is not int or not 0 <= boundary.ayah - origin <= policy.expansion_lookahead:
                raise ValueError("unbounded_following_owner")
''' + needle, 1)
    compile(code, str(file), 'exec')
    file.write_text(code)
    roots[variant] = str(root)
    (P / (variant + '-sha.json')).write_text(json.dumps({'sha256': hashlib.sha256(code.encode()).hexdigest(),
        'parent': parent, 'scope': 'Base nonempty-region fallback; exactly one next eligible DSP candidate, only if spare max8 slot',
        'owner_lookahead': 2, 'new_candidate_radius_cap_ms': 500,
        'preserves_earlier_hard_prefix_block': True, 'control_results_used_for_rule': False}, indent=2) + '\n')
(P / 'snapshot-paths.json').write_text(json.dumps(roots, indent=2) + '\n')
