"""Mutation checks for posterior evidence; tiny synthetic normalized frames."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from surah_cutter.core.fast_adapters import Observation
from surah_cutter.core.fast_executor import Boundary, Policy, _competing_prefix, validate_boundaries

probabilities = np.full((5, 5), .0075)
probabilities[np.arange(5), [1, 0, 2, 0, 0]] = .97
posterior = np.log(probabilities)
sha = hashlib.sha256(memoryview(posterior)).hexdigest()
obs = Observation(("wrong",), (0.,), .99, unit_quality=(.99,), posterior=posterior,
                  posterior_sha256=sha, stride_ms=20., null_log_probability=float(posterior[:, 0].sum()))
read = {"outcome": "completed", "window_ms": [1000., 2500.],
        "observed": {k: v for k, v in asdict(obs).items() if k != "posterior"}}
groups, ends = {1: ("a", "b"), 2: ("c", "d")}, {1: [2], 2: [2]}
adapter = SimpleNamespace(vocab={"a": 1, "b": 2, "c": 3, "d": 4}, blank_id=0,
                          model_id="independent", lag_ms=0)
selected, posterior_proof, expansion = _competing_prefix(obs, groups[1], groups, 1, ends,
                                                        adapter, read, Policy(), .25)
assert posterior_proof and selected.units == groups[1]
proof = {"phrase": groups[1], "complete": True, "ambiguous": False,
         "competitors_checked": True, "offset": 0, "pause_ms": [1000., 1200.], **posterior_proof}
boundary = Boundary(1, 1100, "pause", min(selected.unit_quality), read_id=0, proof=proof)

def validate(record, candidate=boundary):
    validate_boundaries([candidate], [1], 10000., [record], groups=groups, word_ends=ends, threshold=.25)

validate(read)
checks = {}
for name in ("nan_expected", "nan_required_probability", "nan_required_margin", "mismatched_expected_score", "forged_temporal_support"):
    mutated = deepcopy(read)
    comparison = mutated["prefix_comparisons"][posterior_proof["prefix_comparison_id"]]
    candidate = boundary
    if name == "nan_expected":
        comparison["expected_log_probability"] = float("nan")
        comparison["residual_log_probability"] = None
    elif name == "nan_required_probability":
        comparison["required_probability"] = float("nan")
    elif name == "nan_required_margin":
        comparison["required_log_margin"] = float("nan")
    elif name == "mismatched_expected_score":
        comparison["alternatives"][0][2] = -100.
    elif name == "forged_temporal_support":
        candidate = Boundary(1, round(1000. + selected.positions_ms[0]), "first_onset",
                             boundary.confidence, read_id=0, proof=proof)
    try:
        validate(mutated, candidate)
    except ValueError as error:
        checks[name] = str(error)
    else:
        raise AssertionError(f"accepted forged posterior proof: {name}")
Path(__file__).with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
print("5 independent posterior-proof mutation cases passed")
