"""Exhaustive prefix-probability check for prospective CTC scoring study."""
from itertools import product
import json
import math
from pathlib import Path
import runpy

import numpy as np
from surah_cutter.core.fast_ctc import prefix_scores

prefix_score = runpy.run_path(str(Path(__file__).with_name("likelihood_probe.py")))["prefix_score"]


def collapse(path):
    output, previous = [], None
    for token in path:
        if token and token != previous:
            output.append(token)
        previous = token
    return tuple(output)


rng = np.random.default_rng(912)
cases = []
for frames in (1, 2, 3, 4, 5):
    probabilities = rng.uniform(.05, 1, (frames, 3))
    probabilities /= probabilities.sum(-1, keepdims=True)
    logs = np.log(probabilities)
    targets = ((1,), (2,), (1, 1), (1, 2), (2, 1, 2), (1, 1, 2), (1, 2, 2, 1))
    production_scores = prefix_scores(logs, targets, 0)
    for target, score in zip(targets, production_scores):
        expected = 0.
        for path in product(range(3), repeat=frames):
            if collapse(path)[:len(target)] == target:
                expected += math.prod(probabilities[t, token] for t, token in enumerate(path))
        measured = math.exp(score)
        assert abs(measured - math.exp(prefix_score(logs, target, 0))) < 1e-12
        assert abs(measured - expected) < 1e-12, (frames, target, expected, measured)
        cases.append({"frames": frames, "target": target, "probability_error": abs(measured - expected)})
Path(__file__).with_suffix(".json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
print("35 exhaustive CTC prefix probability cases passed")
