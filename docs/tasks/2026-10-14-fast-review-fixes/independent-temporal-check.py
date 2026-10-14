"""Independently reconstruct each bounded temporal diagnostic PCM input."""
import hashlib
import json
from pathlib import Path

import numpy as np

from surah_cutter.core import decode

root = Path(__file__).parent
data = json.loads((root / "temporal-preroll-probe.json").read_text())
assert data["source_sha256"] == hashlib.sha256((root / "temporal_preroll_probe.py").read_bytes()).hexdigest()
assert data["worker_sha256"] == hashlib.sha256(Path("surah_cutter/core/model_worker.py").read_bytes()).hexdigest()
original = decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/108.mp3")
cases = {"basmala_negative": original}
for start in (5720, 6320):
    for silence in (0, 200, 1200):
        cases[f"crop{start}_silence{silence}"] = np.r_[np.zeros(silence * 16, np.float32), original[start * 16:]]
calls = 0
table = []
for item in data["results"] + data["ctc"]:
    pcm = cases[item["case"]]
    trace = item["ledger"]
    reads = trace["acoustic_reads"]
    assert len(reads) == len(item["actual_dispatches"])
    input_ms = 0.
    for read, actual in zip(reads, item["actual_dispatches"]):
        a, b = [round(t * 16) for t in read["window_ms"]]
        local = np.ascontiguousarray(pcm[a:b], dtype=np.float32)
        assert 4800 <= len(local) <= 96000 and len(local) < len(pcm)
        assert actual["samples"] == len(local)
        assert actual["sha256"] == hashlib.sha256(memoryview(local)).hexdigest()
        input_ms += len(local) / 16
        calls += 1
    assert abs(input_ms - trace["audio_ms"]) < .1
    assert input_ms <= len(pcm) / 16 * 2
    assert trace["covered_audio_ms"] <= len(pcm) / 16 * .65
    if "timings" in item:
        for timing in item["timings"].values():
            index = timing["first_text_index"]
            assert timing["diagnostic_predicting_query"] == timing["query_times"][index]
            assert timing["existing_query_plus_one"][0][0] == timing["query_times"][index + 1]
        timing = item["timings"]["إِنَّا"]
        table.append({"model": item["model"], "case": item["case"],
                      "independent_identity": item["independently_proved_inna"],
                      "existing_ms": timing["existing_query_plus_one"][0][0],
                      "alternative_ms": timing["diagnostic_predicting_query"]})
assert calls == 49
assert all(not r["independent_identity"] for r in table if r["case"] == "basmala_negative")
(root / "independent-temporal-check.json").write_text(json.dumps({"verified_local_passes": calls,
    "timing_conventions": table, "no_word_onset_ground_truth": True}, indent=2) + "\n")
print("49 actual temporal diagnostic passes independently verified")
