"""B2: cover clipped and glued one-word plans, keeping proof quality unchanged."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(roots['B']).with_name('B2')
assert not root.exists()
shutil.copytree(roots['B'], root, ignore=shutil.ignore_patterns('__pycache__'))
file = root / 'surah_cutter/core/fast_executor.py'
code = file.read_text()
start = code.index('def _partial_asr_head(')
end = code.index('def _incomplete_speech_edge(', start)
code = code[:start] + '''def _partial_asr_head(observation, phrase, threshold):
    """A clipped/glued high-quality word is a cue, never an identity proof."""
    if (observation.positions_ms is not None or observation.truncated
            or len(observation.units) != 1 or not phrase or len(phrase[0]) < 3
            or len(observation.unit_quality) != 1
            or not math.isfinite(observation.unit_quality[0])
            or observation.unit_quality[0] < threshold
            or any("unsupported" in item or "spoken_alias" in item for item in observation.attribution)):
        return False
    heard, expected = observation.units[0], phrase[0]
    common = 0
    for left, right in zip(heard, expected):
        if left != right:
            break
        common += 1
    return common >= 3 and (heard.startswith(expected) or common >= len(heard) - 1)


def _asr_head_competitors(observation, groups, word_ends, threshold):
    # Starts and internal word boundaries get exactly the same cue rule.
    # No unique owner is inferred from this deliberately incomplete spelling.
    return [(owner, offset) for owner, units in groups.items()
            for offset in [0] + word_ends[owner][:-1]
            if _partial_asr_head(observation, units[offset:], threshold)]


''' + code[end:]
code = code.replace('or len(observation.units) >= len(phrase)\n',
                    'or (not asr_cue and len(observation.units) >= len(phrase))\n', 1)
code = code.replace('                soft_short_word_route = False',
                    '                partial_head_route = None\n                soft_short_word_route = False', 1)
code = code.replace('                                    if edge:\n                                        anchor =',
                    '                                    if edge:\n                                        partial_head_route = opening_id\n                                        anchor =', 1)
code = code.replace('                            expansion = "partial_asr_head_at_speech_edge"',
                    '                            partial_head_route = opening_id\n                            expansion = "partial_asr_head_at_speech_edge"', 1)
code = code.replace('possible_ayah=future_ayah, reason=anchor, speech_edge=edge)',
    'possible_ayah=future_ayah, reason=anchor, speech_edge=edge,\n                                              competing_fragments=_asr_head_competitors(next_obs, groups, word_ends, threshold))', 1)
code = code.replace('reason=expansion, speech_edge=edge)',
    'reason=expansion, speech_edge=edge,\n                                  competing_fragments=_asr_head_competitors(obs, groups, word_ends, threshold))', 1)
needle = 'and phrase_quality >= threshold and not alias_conflict):'
assert needle in code
code = code.replace(needle, '''and phrase_quality >= threshold and not alias_conflict
                            and (partial_head_route is None or not _strong_continuation_conflict(
                                obs.units, obs.unit_quality, groups[ayah], len(phrase), threshold))):''', 1)
code = code.replace('                        if noisy_proof:\n                            proof.update(noisy_proof)',
    '                        if partial_head_route is not None:\n                            proof["partial_asr_head_cue_id"] = partial_head_route\n                        if noisy_proof:\n                            proof.update(noisy_proof)', 1)
needle = 'and not next_obs.truncated and quality >= threshold and not conflict):'
assert needle in code
code = code.replace(needle, '''and not next_obs.truncated and quality >= threshold and not conflict
                                    and (partial_head_route is None or not _strong_continuation_conflict(
                                        next_obs.units, next_obs.unit_quality, groups[future_ayah], len(next_phrase), threshold))):
                                if partial_head_route is not None:
                                    future_proof["partial_asr_head_cue_id"] = partial_head_route''', 1)
compile(code, str(file), 'exec')
file.write_text(code)
roots['B2'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(roots, indent=2) + '\n')
(P / 'B2-sha.json').write_text(json.dumps({'sha256': hashlib.sha256(code.encode()).hexdigest(),
    'changes': 'clipped/glued single word plans; symmetric cue competitors; stronger continuation check for new route',
    'control_results_seen': False}, indent=2) + '\n')
