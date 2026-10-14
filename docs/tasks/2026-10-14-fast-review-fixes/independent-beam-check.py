"""Compare local CTC decoder with exhaustive paths, without known Quran text."""
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from surah_cutter.core.fast_ctc import independent_beam


def collapse(path):
    output = []
    previous = None
    for token in path:
        if token and token != previous:
            output.append(token)
        previous = token
    return tuple(output)


rng = np.random.default_rng(731)
checks = []
for frames in (1, 2, 3, 4):
    for case in range(8):
        probabilities = rng.uniform(.01, 1, (frames, 3))
        probabilities /= probabilities.sum(-1, keepdims=True)
        logs = np.log(probabilities)
        exhaustive = {}
        for path in product(range(3), repeat=frames):
            probability = math.prod(probabilities[t, token] for t, token in enumerate(path))
            label = collapse(path)
            exhaustive[label] = exhaustive.get(label, 0.) + probability
        hypotheses, null = independent_beam(logs, 0, {0: "", 1: "a", 2: "b"}, 20., 0., width=64)
        assert all(math.isfinite(h["log_probability"]) for h in hypotheses), (frames, case, "nonfinite hypothesis")
        measured = {tuple(1 if c == "a" else 2 for c in h["units"]): math.exp(h["log_probability"])
                    for h in hypotheses}
        assert set(measured) == set(exhaustive)
        error = max(abs(measured[k] - expected) for k, expected in exhaustive.items())
        assert error < 1e-12, (frames, case, error, measured, exhaustive)
        assert abs(math.exp(null) - exhaustive[()]) < 1e-12
        assert all(len(h["units"]) == len(h["positions_ms"]) == len(h["unit_quality"])
                   for h in hypotheses)
        checks.append({"frames": frames, "case": case, "max_probability_error": error})
output = Path(__file__).with_suffix(".json")
output.write_text(json.dumps({"kind": "exhaustive_ctc_path_check", "cases": checks}, indent=2) + "\n")
print(f"{len(checks)} exhaustive local CTC cases passed")
