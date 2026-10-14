"""Sparse-anchor negatives without expected timing/ASR model shortcuts."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from surah_cutter.core.fast_adapters import Observation
from surah_cutter.core.fast_executor import Policy, align_surah_sequential
from surah_cutter.data import quran_text


class Acoustic:
    version = "independent-negative-v1"
    model_id = "independent-test"
    revision = "captured-fake"
    metric = "asr_token_probability"
    lag_ms = 0
    timing = False

    def __init__(self, future, repeated=False):
        self.words = quran_text.surah_words(112)
        self.rendered = tuple((f"ayah-{w.ayah}-word-{i}",) for i, w in enumerate(self.words))
        if repeated:
            rendered = list(self.rendered)
            indices = [i for i, w in enumerate(self.words) if w.ayah == future]
            other = [i for i, w in enumerate(self.words) if w.ayah == future + 1]
            for i, j in zip(indices, other):
                rendered[j] = rendered[i]
            self.rendered = tuple(rendered)
        self.future = future
        self.calls = []

    def represent(self, words):
        return self.rendered

    def opening(self, ayah):
        return tuple(u for w, units in zip(self.words, self.rendered) if w.ayah == ayah for u in units)[:1]

    def observe(self, pcm, sr, *, total_samples):
        start = round(float(pcm[0]))
        self.calls.append(start)
        units = self.opening(self.future) if start == 1000 else self.opening(1) if start == 10000 else ()
        return Observation(units, None, .99, unit_quality=(.99,) * len(units))


def run(fake, policy=Policy()):
    pcm = np.repeat(np.arange(40000, dtype=np.float32), 16)
    decisions = [SimpleNamespace(accepted=True, pause=SimpleNamespace(
        start_ms=float(s), end_ms=float(s + 200), center_ms=float(s + 100))) for s in (1000, 10000)]
    trace = {}
    with patch("surah_cutter.core.fast_executor.first_pause_hypotheses", return_value=[]):
        rows = align_surah_sequential(pcm, 16000, fake.words, adapter=fake, trace=trace,
                                      policy=policy, pause_decisions=decisions)
    return rows, trace


rows, trace = run(Acoustic(3))
assert rows[0][2] == rows[1][2] == 0 and rows[2][2] > 0
assert 1 in trace["review_required_ayahs"] and 2 in trace["review_required_ayahs"]
assert rows[0][1] < rows[1][1] < rows[2][1] < rows[3][1]
assert trace["boundaries"][2]["proof"]["sparse_anchor"]
assert not any(b["ayah"] == 1 and b["confidence"] > 0 for b in trace["boundaries"])

far_rows, far_trace = run(Acoustic(4))
if Policy().max_unresolved_ayahs is None:
    assert all(r[2] == 0 for r in far_rows[:3]) and far_rows[3][2] > 0
    capped_rows, _ = run(Acoustic(4), policy=Policy(max_unresolved_ayahs=2))
    assert all(conf == 0 for a, ms, conf in capped_rows)
else:
    assert all(conf == 0 for a, ms, conf in far_rows)
repeat_rows, repeat_trace = run(Acoustic(3, repeated=True))
assert repeat_rows[2][2] == 0
assert not any((b.get("proof") or {}).get("sparse_anchor") for b in repeat_trace["boundaries"])
Path(__file__).with_suffix(".json").write_text(json.dumps({
    "policy": asdict(Policy()), "gap_stays_review": True, "late_current_not_substituted": True,
    "long_gap_remains_review": True, "explicit_gap_cap_respected": True, "repeated_future_rejected": True,
}, indent=2) + "\n")
print("5 independent sparse-anchor cases passed")
