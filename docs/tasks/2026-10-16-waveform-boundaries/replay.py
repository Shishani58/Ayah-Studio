"""Signal-only replay of all stored pre-refinement FULL checks."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from surah_cutter.core import decode, full_signal
from surah_cutter.data import references

TASK = Path(__file__).resolve().parent
PREVIOUS = TASK.parent / "2026-10-16-full-transition-refinement"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    baseline = json.loads((PREVIOUS / "verification.json").read_text())
    protected = {p: sha(p) for item in baseline for p in item["protected_sha256"]}
    results = []
    for item in baseline:
        audio = next(p for p in item["protected_sha256"] if p.endswith(".mp3"))
        assert sha(audio) == item["protected_sha256"][audio]
        pcm = decode.decode(audio)
        checks = {c["ayah"]: deepcopy(c) for c in item["before_checks"]}
        before = deepcopy(checks)
        trace = {}
        start = time.perf_counter()
        full_signal.refine_late_boundaries(checks, pcm, 16000, trace=trace)
        elapsed = time.perf_counter() - start
        reference = references.load_reference(item["reference"])
        changes = [dict(ayah=a, before_ms=before[a]["onset_ms"], after_ms=c["onset_ms"],
                        reference_ms=reference[item["surah"], a]) for a, c in checks.items()
                   if c["onset_ms"] != before[a]["onset_ms"]]
        new_over500 = [a for a, c in checks.items() if abs(c["onset_ms"] - reference[item["surah"], a]) > 500
                       >= abs(before[a]["onset_ms"] - reference[item["surah"], a])]
        assert not new_over500
        assert all(a["onset_ms"] < b["onset_ms"] for a, b in zip(checks.values(), list(checks.values())[1:]))
        result = dict(reference=item["reference"], surah=item["surah"], n=item["n"],
                      elapsed_s=elapsed, changes=changes, new_over500=new_over500,
                      outcomes=dict(Counter(r["reason"] for r in trace["full_signal_checks"])), trace=trace)
        results.append(result)
        (TASK / "comparison.json").write_text(json.dumps(results, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "trace"}, ensure_ascii=False), flush=True)
    after = {p: sha(p) for p in protected}
    (TASK / "protected-sha256.json").write_text(json.dumps(dict(before=protected, after=after), indent=2) + "\n")
    assert after == protected


if __name__ == "__main__":
    main()
