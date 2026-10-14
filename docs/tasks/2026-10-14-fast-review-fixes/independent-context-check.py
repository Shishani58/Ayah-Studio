"""Adversarial proof mutation for repeated ayah ownership; no acoustic calls."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from surah_cutter.core.fast_adapters import Observation, arabic_units
from surah_cutter.core.fast_executor import Boundary, plan_seam, validate_boundaries
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.data import quran_text

words = quran_text.surah_words(109)
groups, ends = {}, {}
for word, text in zip(words, inference_words(words, "whisper-tarteel-tiny")):
    groups.setdefault(word.ayah, []).extend(arabic_units(text))
    ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
groups = {a: tuple(u) for a, u in groups.items()}
seam = plan_seam(groups, 3, ends)
assert seam and not seam["ambiguous"]
phrase, tail = tuple(seam["units"]), tuple(seam["tail"])

def observed(units):
    return asdict(Observation(units, None, .99, unit_quality=(.99,) * len(units)))

proof = {"phrase": phrase, "previous_tail": tail, "contextual_owner": True,
         "previous_context_read_id": 1, "complete": True, "ambiguous": False,
         "competitors_checked": True, "offset": 0, "pause_ms": [1000., 1200.]}
boundary = Boundary(3, 1100, "pause", .99, read_id=0, proof=proof)
original = [{"outcome": "completed", "window_ms": [1000., 2500.], "observed": observed(phrase)},
            {"outcome": "completed", "window_ms": [200., 1000.], "observed": observed(tail)}]

def validate(reads):
    validate_boundaries([boundary], [3], 10000., reads,
                        groups=groups, word_ends=ends, threshold=.6)

validate(original)
checks = {}
for name in ("stale_left_tail", "truncated_left_tail", "weak_left_tail", "wrong_left_owner", "unbounded_right_preroll", "overstated_context_confidence"):
    mutated = deepcopy(original)
    if name == "stale_left_tail":
        mutated[1]["window_ms"][1] = 600.
    elif name == "truncated_left_tail":
        mutated[1]["observed"]["truncated"] = True
    elif name == "weak_left_tail":
        mutated[1]["observed"]["unit_quality"] = (.1,) * len(tail)
    elif name == "wrong_left_owner":
        mutated[1]["observed"]["units"] = ("عبدتم",)
    elif name == "unbounded_right_preroll":
        mutated[0]["window_ms"][0] = 0.
    elif name == "overstated_context_confidence":
        mutated[1]["observed"]["unit_quality"] = (.61,) * len(tail)
    try:
        validate(mutated)
    except ValueError as error:
        checks[name] = str(error)
    else:
        raise AssertionError(f"accepted forged contextual proof: {name}")
Path(__file__).with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
print("6 independent repeated-owner negative cases passed")
