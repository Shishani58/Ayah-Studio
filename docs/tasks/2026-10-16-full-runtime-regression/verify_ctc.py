"""Differential checks against the frozen pre-change exact CTC implementation."""

import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from surah_cutter.core import _ctc  # noqa: E402

spec = importlib.util.spec_from_file_location("ctc_before", Path(__file__).with_name("ctc_before.py"))
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
rng = np.random.default_rng(916)
count = 0
errors = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for case in range(1200):
        frames, vocab = int(rng.integers(0, 40)), int(rng.integers(2, 10))
        length, blank = int(rng.integers(0, 15)), int(rng.integers(vocab))
        matrix = rng.integers(-7, 1, (1, frames, vocab)).astype(float)
        if case % 3 == 0:
            matrix = rng.normal(-4, 2, matrix.shape)
        if matrix.size and case % 7 == 0:
            matrix.flat[int(rng.integers(matrix.size))] = [np.nan, np.inf, -np.inf][case % 3]
        tokens = np.array([i for i in range(vocab) if i != blank])
        targets = rng.choice(tokens, (1, length))
        values = []
        for fn in (old.forced_align, _ctc.forced_align):
            try:
                values.append(fn(matrix, targets, blank=blank))
            except ValueError as exc:
                values.append((type(exc), str(exc)))
        if isinstance(values[0][0], type):
            assert values[0] == values[1], case
            errors += 1
        else:
            for a, b in zip(*values):
                np.testing.assert_array_equal(a, b, err_msg=f"case {case}")
        count += 1

result = {"cases": count, "identical_errors": errors, "exact_path_scores": True}
if "--benchmark" in sys.argv:
    matrix = rng.normal(-5, 2, (1, 13000, 38))
    targets = rng.integers(1, 38, (1, 1100))
    timings = {"before": [], "after": []}
    outputs = []
    for label, fn in [("before", old.forced_align), ("after", _ctc.forced_align),
                      ("after", _ctc.forced_align), ("before", old.forced_align)]:
        started = time.perf_counter()
        outputs.append(fn(matrix, targets))
        timings[label].append(time.perf_counter() - started)
    assert all(np.array_equal(a, b) for other in outputs[1:] for a, b in zip(outputs[0], other))
    result["benchmark_s"] = timings
Path(__file__).with_name("ctc-verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
