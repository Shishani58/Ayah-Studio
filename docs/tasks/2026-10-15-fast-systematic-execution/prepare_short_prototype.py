"""Isolated H1/H4 prototype; never writes production sources."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
paths = json.loads((P / 'snapshot-paths.json').read_text())
base = Path(paths['baseline']) / 'surah_cutter/core/fast_executor.py'
dest = Path(paths['prototype']) / 'surah_cutter/core/fast_executor.py'
assert base.read_bytes() == dest.read_bytes()
source = base.read_text()
source = source.replace('class Policy:\n', 'class Policy:\n    ctc_short_speech_ms: int = 500\n', 1)
helper = '''def _short_context_cue(observation, phrase, groups, word_ends, ayah, threshold):
    """A distinct two-phone remainder permits one shorter read, never a proof.

    Limited to an entire minimal three-phone opening: the measured short crop
    can contain the whole word. Rivals get the same missing-first-phone rule.
    The new observation still must prove every original target phone normally.
    """
    if (len(phrase) != 3 or word_ends[ayah][0] != 3 or observation.truncated
            or _prefix(observation.units, phrase) == len(phrase)):
        return False
    remainder = phrase[1:]
    q = observation.unit_quality[:len(remainder)]
    if (tuple(observation.units[:len(remainder)]) != remainder or len(q) != len(remainder)
            or any(not math.isfinite(value) or value < threshold for value in q)):
        return False
    for owner, units in groups.items():
        for start in [0] + word_ends[owner][:-1]:
            if (owner, start) == (ayah, 0):
                continue
            if any(units[start + missing:start + missing + len(remainder)] == remainder
                   for missing in (0, 1)):
                return False
    return True


'''
source = source.replace('def _noisy_opening(', helper + 'def _noisy_opening(', 1)
old = '''                    opening_id = reads.last_read_id
                    beam_proof = {}'''
new = '''                    opening_id = reads.last_read_id
                    short_context_proof = {}
                    if (step == 0 and pause and adapter.metric == "ctc_greedy_probability"
                            and not plan["ambiguous"] and not plan.get("contextual_owner")
                            and _short_context_cue(obs, phrase, groups, word_ends, ayah, threshold)):
                        initial_opening_id = opening_id
                        short_end = min(end, speech_start + policy.ctc_short_speech_ms)
                        event("SHORT_CONTEXT_RECHECK", ayah, candidate=index,
                              initial_read_id=initial_opening_id, window_ms=[start, short_end],
                              reason="unique_short_opening_remainder")
                        shorter = reads.read(start, short_end, reason=f"ayah={ayah}:short_context_recheck")
                        if (shorter and not shorter.truncated and _prefix(shorter.units, phrase) == len(phrase)
                                and min(shorter.unit_quality[:len(phrase)], default=0.) >= threshold):
                            obs = shorter
                            opening_id = reads.last_read_id
                            short_context_proof = {"short_context_recheck": True,
                                                   "initial_read_id": initial_opening_id}
                    beam_proof = dict(short_context_proof)'''
assert old in source
source = source.replace(old, new, 1)
# Keep ordinary observations/expansions on a failed short attempt. Proof is the
# same full exact opening and same pause geometry; no new validator bypass.
source = source.replace('VERSION = ', 'VERSION = ', 1)
dest.write_text(source)
(P / 'short-prototype-sha.json').write_text(json.dumps({'initial': hashlib.sha256(base.read_bytes()).hexdigest(),
     'prototype': hashlib.sha256(dest.read_bytes()).hexdigest()}, indent=2))
print(dest)
