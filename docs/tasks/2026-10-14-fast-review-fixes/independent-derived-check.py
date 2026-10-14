"""Reconstruct derived PCM and check shifted reference after frozen inference."""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from surah_cutter.core import decode

root = Path(__file__).parent
original = decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/108.mp3")
cases = {
    "crop_after_intro_candidate_6320ms": original[101120:],
    "leading_silence_12s": np.r_[np.zeros(192000, dtype=np.float32), original],
    "repeat_segment_6320_10460ms": np.r_[original[:167360], original[101120:167360], original[167360:]],
    "truncated_at_8s": original[:128000],
    "very_short_500ms": original[:8000],
}
items = json.loads((root / "edges-final.json").read_text())
assert len(items) == 15
with sqlite3.connect("file:data/references/abu_bakr_ash_shatri.db?mode=ro", uri=True) as connection:
    truth = dict(connection.execute("SELECT ayah,time FROM timings WHERE sura=108"))
verified = 0
errors = []
for item in items:
    name, pcm, trace = item["case"], cases[item["case"]], item["trace"]
    assert item["pcm_sha256"] == hashlib.sha256(memoryview(pcm)).hexdigest()
    assert [r[0] for r in item["rows"]] == [1, 2, 3]
    assert all(a[1] < b[1] for a, b in zip(item["rows"], item["rows"][1:]))
    assert all(0 <= r[1] < len(pcm) / 16 for r in item["rows"])
    reads = trace["acoustic_reads"]
    assert len(reads) == len(item["actual_dispatches"])
    inputs, intervals = 0., []
    for read, actual in zip(reads, item["actual_dispatches"]):
        a, b = [round(t * 16) for t in read["window_ms"]]
        local = np.ascontiguousarray(pcm[a:b], dtype=np.float32)
        assert 0 < len(local) <= 96000 and len(local) < len(pcm)
        assert actual["samples"] == len(local)
        assert actual["sha256"] == hashlib.sha256(memoryview(local)).hexdigest()
        inputs += len(local) / 16
        intervals.append((a, b))
        verified += 1
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    union = sum(b-a for a, b in merged) / 16
    assert abs(inputs - trace["audio_ms"]) < .1
    assert abs(union - trace["covered_audio_ms"]) < .1
    assert inputs <= len(pcm) / 16 * 2 and union <= len(pcm) / 16 * .65
    if name == "very_short_500ms":
        assert not reads and all(r[2] == 0 for r in item["rows"])
    # These two transformations preserve each original reference boundary.
    # Cropped/truncated clips have no separately labelled word-onset truth.
    if name in ("leading_silence_12s", "repeat_segment_6320_10460ms"):
        for ayah, ms, confidence in item["rows"]:
            if confidence > 0:
                shift = 12000 if name == "leading_silence_12s" else 4140 if truth[ayah] >= 10460 else 0
                error = abs(ms - truth[ayah] - shift)
                assert error <= 500
                errors.append([item["model"], name, ayah, error])
(root / "independent-derived-check.json").write_text(json.dumps({"cases": len(items),
    "verified_dispatches": verified, "shifted_positive_errors_ms": errors}, indent=2) + "\n")
print("15 derived inputs and their actual acoustic passes independently verified")
