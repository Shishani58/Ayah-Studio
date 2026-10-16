"""Measure PCM envelopes of saved ambiguous FULL cuts. No model calls or DB writes."""
import json
from pathlib import Path

import numpy as np

from surah_cutter.core import decode

TASK = Path(__file__).resolve().parent
PREVIOUS = TASK.parent / "2026-10-16-full-transition-refinement"


def main():
    baseline = json.loads((PREVIOUS / "verification.json").read_text())
    results = []
    arrays = {}
    for item in baseline:
        reads = [r for r in item["trace"].get("full_boundary_reads", []) if r.get("role") == "two-sided-cut"]
        if not reads:
            continue
        audio = next(p for p in item["protected_sha256"] if p.endswith(".mp3"))
        pcm = decode.decode(audio)
        checks = {c["ayah"]: c for c in item["before_checks"]}
        for read in reads:
            a = read["ayah"]
            tail = checks[a - 1]["full_ctc_end_ms"]
            first = checks[a]["full_ctc_first_ms"]
            lo = max(0, int((tail - 400) // 5) * 5)
            hi = min(len(pcm) / 16, int((first + 600) // 5) * 5)
            crop = pcm[round(lo * 16):round(hi * 16)]
            frames = crop[:len(crop) // 80 * 80].reshape(-1, 80)
            rms = np.sqrt(np.mean(frames * frames, axis=1))
            peak = np.max(np.abs(frames), axis=1)
            hf = np.sqrt(np.mean(np.diff(frames, axis=1) ** 2, axis=1))
            times = lo + np.arange(len(rms)) * 5
            start, end = read["pause_start_ms"], read["pause_end_ms"]
            guard = min(80, (end - start) * .22)
            quiet = (times >= start + guard) & (times + 5 <= end - guard)
            left = (times >= tail - 250) & (times < tail + 50)
            right = (times >= first - 200) & (times < first + 250)
            scale = max(float(np.percentile(rms[left], 75)), float(np.percentile(rms[right], 75)), 1e-10)
            key = f"{item['surah']}_{a}"
            arrays[key] = np.stack([times, rms / scale, peak / scale, hf / scale])
            result = dict(surah=item["surah"], ayah=a, raw_ms=checks[a]["raw_onset_ms"],
                          tail_ms=tail, first_ms=first, pause_ms=[start, end], scale=scale,
                          quiet_rms_percentiles=np.percentile(rms[quiet] / scale, [10, 50, 95, 100]).tolist(),
                          quiet_peak_percentiles=np.percentile(peak[quiet] / scale, [50, 95, 100]).tolist(),
                          quiet_hf_percentiles=np.percentile(hf[quiet] / scale, [50, 95, 100]).tolist())
            results.append(result)
            print(json.dumps(result), flush=True)
    (TASK / "measurements.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez(TASK / "envelopes.npz", **arrays)


if __name__ == "__main__":
    main()
