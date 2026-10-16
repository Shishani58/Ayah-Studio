"""Isolated causal variants; never edit production or use reference to choose reads."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
original = (Path(roots['A']) / 'surah_cutter/core/fast_executor.py').read_text()

helper = '''def _partial_asr_head(observation, phrase, threshold):
    """A strong clipped spelling requests evidence, never chooses an owner."""
    if (observation.positions_ms is not None or observation.truncated
            or len(observation.units) != 1 or not phrase or len(phrase[0]) < 4
            or len(observation.unit_quality) != 1
            or not math.isfinite(observation.unit_quality[0])
            or observation.unit_quality[0] < threshold):
        return False
    heard = observation.units[0]
    common = 0
    for left, right in zip(heard, phrase[0]):
        if left != right:
            break
        common += 1
    return common >= 3 and common >= len(heard) - 1 and common < len(phrase[0])


'''

for variant in ('B', 'C', 'D'):
    code = original
    if variant in ('B', 'D'):
        code = code.replace('def _incomplete_speech_edge(', helper + 'def _incomplete_speech_edge(', 1)
        code = code.replace('word_ends, ayah, threshold):\n    """DSP explains',
                            'word_ends, ayah, threshold, *, asr_cue=False):\n    """DSP explains', 1)
        code = code.replace('if (observation.positions_ms is None or not observation.units or observation.truncated',
                            'if ((observation.positions_ms is None and not asr_cue) or not observation.units or observation.truncated', 1)
        code = code.replace('    if rivals:\n        return None\n    edge =',
                            '    if rivals and not asr_cue:\n        return None\n    edge =', 1)
        needle = '                                if (anchor and (anchor != "whole_spoken_prefix_incomplete_canonical_word"'
        assert needle in code
        code = code.replace(needle, '''                                if (not anchor and _partial_asr_head(next_obs, next_phrase, threshold)):
                                    edge = _incomplete_speech_edge(pcm, sr, end, next_obs, next_phrase,
                                        groups, word_ends, future_ayah, threshold, asr_cue=True)
                                    if edge:
                                        anchor = "partial_asr_head_at_speech_edge"
                                        event("SEARCH_CUE_ONLY", ayah, candidate=index, read_id=opening_id,
                                              possible_ayah=future_ayah, reason=anchor, speech_edge=edge)
''' + needle, 1)
        needle = '                    if not expansion and segmented_expansion:'
        code = code.replace(needle, '''                    if not expansion and _partial_asr_head(obs, phrase, threshold):
                        edge = _incomplete_speech_edge(pcm, sr, end, obs, phrase, groups,
                                                      word_ends, ayah, threshold, asr_cue=True)
                        if edge:
                            expansion = "partial_asr_head_at_speech_edge"
                            event("SEARCH_CUE_ONLY", ayah, candidate=index, read_id=opening_id,
                                  reason=expansion, speech_edge=edge)
''' + needle, 1)
        code = code.replace('soft_short_word_route |= (expansion == "whole_spoken_prefix_incomplete_canonical_word"',
                            'soft_short_word_route |= (expansion in {"whole_spoken_prefix_incomplete_canonical_word", "partial_asr_head_at_speech_edge"}', 1)
    if variant in ('C', 'D'):
        needle = '                        event("SELECT_CANDIDATE", ayah, candidate=index, declined_geometry=reason)\n                        continue'
        assert needle in code
        code = code.replace(needle, '''                        event("SELECT_CANDIDATE", ayah, candidate=index, declined_geometry=reason)
                        # Rejected temporal geometry remains rejected. The speech
                        # at its measured right edge may still explain identity.
                        if (reason == "decaying_tail_midpoint_unproved" and not adapter.timing):
                            for speech in policy.speech_ms:
                                end = min(duration, speech_start + speech)
                                obs = reads.read(start, end, reason=f"ayah={ayah}:rejected_midpoint_identity")
                                if obs is None:
                                    break
                                quality = obs.unit_quality[:len(phrase)]
                                if (not plan["ambiguous"] and not obs.truncated
                                        and _prefix(obs.units, phrase) == len(phrase)
                                        and len(quality) == len(phrase) and min(quality) >= threshold
                                        and not _strong_continuation_conflict(obs.units, obs.unit_quality,
                                                                            groups[ayah], len(phrase), threshold)):
                                    event("IDENTITY_ONLY", ayah, candidate=index, read_id=reads.last_read_id,
                                          phrase=phrase, temporal_reason=reason, onset_supported=False)
                                    break
                                if not _expansion_evidence(obs, phrase, word_ends[ayah], threshold,
                                                           groups, ayah, word_ends):
                                    break
                        continue''', 1)
    dest = Path(roots[variant]) / 'surah_cutter/core/fast_executor.py'
    dest.write_text(code)
    compile(code, str(dest), 'exec')
(P / 'variant-source-sha.json').write_text(json.dumps({v: hashlib.sha256(
    (Path(root) / 'surah_cutter/core/fast_executor.py').read_bytes()).hexdigest()
    for v, root in roots.items()}, indent=2) + '\n')
