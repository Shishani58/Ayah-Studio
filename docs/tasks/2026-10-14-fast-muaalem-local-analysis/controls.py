"""Silence/shift/wrong-text controls of independently decoded local phones."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from adapter import MuaalemLocal
from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_ctc import prefix_scores
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

task = Path(__file__).parent
model_store.ACTIVE_ID = "muaalem"
bundle = align.default_bundle()
adapter = MuaalemLocal(bundle)
pcm = decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/001.mp3")
words = quran_text.surah_words(1)
units = adapter.represent(words)
target = next(u for w, u in zip(words, units) if w.ayah == 2)
wrong = next(u for w, u in zip(words, units) if w.ayah == 3)
left, end = 6420, 8520
start, stop = left * 16, end * 16
shifted = pcm.copy()
shifted[start:stop + 3200] = np.concatenate((np.zeros(3200, dtype=np.float32), pcm[start:stop]))
silence = pcm.copy()
silence[start:stop] = 0.
records = []
for name, signal, right in (("original", pcm, end), ("leading_silence_200ms", shifted, end + 200),
                            ("silence", silence, end)):
    ledger = LocalEvidenceReads(signal, 16000, adapter)
    with dispatch_spy(bundle, signal) as actual:
        observation = ledger.read(left, right, reason=f"muaalem_control:{name}")
    trace = {"total_ms": len(signal) / 16, **ledger.diagnostics()}
    verify_dispatch(actual, signal, trace)
    matches = [i for i in range(len(observation.units))
               if observation.units[i:i + len(target)] == target]
    first_position = observation.positions_ms[matches[0]] if len(matches) == 1 else None
    probabilities = [math.exp(score) for score in prefix_scores(
        observation.posterior, [[adapter.vocab[u] for u in phrase] for phrase in (target, wrong)],
        adapter.blank_id)]
    records.append({"control": name, "expected": target, "wrong_target": wrong,
                    "greedy": observation.units, "greedy_occurrences": matches,
                    "first_position_ms": first_position, "unit_quality": observation.unit_quality,
                    "prefix_probability_correct": probabilities[0], "prefix_probability_wrong": probabilities[1],
                    "ledger": trace, "actual_dispatches": actual})
    (task / "controls-partial.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
first, second, quiet = records
delta = (second["first_position_ms"] - first["first_position_ms"]
         if first["first_position_ms"] is not None and second["first_position_ms"] is not None else None)
report = {"records": records, "shift_expected_ms": 200, "shift_observed_ms": delta,
          "shift_error_ms": abs(delta - 200) if delta is not None else None,
          "silence_word_absent": not quiet["greedy_occurrences"],
          "adapter_version": adapter.version, "runtime_identity": adapter.runtime_identity,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "scope": "local temporal controls; not general word-onset calibration or native FAST activation"}
(task / "controls.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("records", "runtime_identity")}, indent=2))
release_runtimes()
