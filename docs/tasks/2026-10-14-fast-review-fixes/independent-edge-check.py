"""Measured speech at crop edge permits expansion, never confirms wrong text."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from surah_cutter.core.fast_adapters import Observation
from surah_cutter.core.fast_executor import _incomplete_speech_edge, align_surah_sequential
from surah_cutter.data import quran_text

pcm = np.full(16000 * 30, .05, dtype=np.float32)
obs = Observation(("x",), (0.,), .99, unit_quality=(.99,))
groups = {1: tuple("abc"), 2: tuple("def")}
ends = {1: (3,), 2: (3,)}
def edge(audio, observation):
    return _incomplete_speech_edge(audio, 16000, 2000., observation,
                                   groups[1], groups, ends, 1, .25)
assert edge(pcm, obs)
checks = {}
cases = {
    "silent_edge": (np.zeros_like(pcm), obs),
    "silence_after_edge": (np.r_[pcm[:32000], np.zeros_like(pcm[32000:])], obs),
    "weak_decode": (pcm, replace(obs, unit_quality=(.1,))),
    "truncated_decode": (pcm, replace(obs, truncated=True)),
    "empty_decode": (pcm, replace(obs, units=(), unit_quality=(), positions_ms=())),
    "complete_other_owner": (pcm, replace(obs, units=tuple("def"), positions_ms=(0.,)*3,
                                          unit_quality=(.99,)*3)),
}
for label, arguments in cases.items():
    assert edge(*arguments) is None, label
    checks[label] = True

class WrongSpeech:
    version = "independent-edge-negative-v1"
    model_id = "independent-test"
    revision = "fake"
    metric = "ctc_greedy_probability"
    lag_ms = 0
    timing = True

    def represent(self, words):
        return tuple(tuple(f"ayah{w.ayah}word{i}") for i, w in enumerate(words))

    def observe(self, local, sr, *, total_samples):
        return obs

trace = {}
pause = SimpleNamespace(start_ms=1000., end_ms=1200., center_ms=1100.)
with patch("surah_cutter.core.fast_executor.first_pause_hypotheses", return_value=[]):
    rows = align_surah_sequential(pcm, 16000, quran_text.surah_words(112),
                                 adapter=WrongSpeech(), trace=trace,
                                 pause_decisions=[SimpleNamespace(accepted=True, pause=pause)])
assert all(confidence == 0 for _, _, confidence in rows)
assert len(trace["acoustic_reads"]) <= 6
assert any(e["state"] == "EXPAND_LOCAL" for e in trace["transitions"])
checks["wrong_speech_never_confirms_after_finite_expansion"] = True
Path(__file__).with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
print("7 independent speech-edge negative cases passed")
