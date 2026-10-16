"""Final implementation on captured pre-refinement states, with fresh local reads.

The expensive original FULL acoustic pass is reused verbatim. This is not a
second complete FULL run. Final real-audio pytest separately covers the full
pipeline, and verification.json includes the long bounded run.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import wave

import numpy as np

from surah_cutter.core import align, decode, emissions, full_transition
from surah_cutter.data import quran_text, references

TASK = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    items = json.loads((TASK / "verification.json").read_text())
    bundle = align._OnnxBundle(emissions.load_emitter(Path("models/wav2vec2-quran-phonetics.onnx")))
    results = []
    clips = TASK / "clips"
    clips.mkdir(exist_ok=True)
    for item in items:
        audio = next(Path(p) for p in item["protected_sha256"] if p.endswith(".mp3"))
        assert item["protected_sha256"] == {p: sha(Path(p)) for p in item["protected_sha256"]}
        pcm = decode.decode(audio)
        checks = {c["ayah"]: deepcopy(c) for c in item["before_checks"]}
        trace = {}
        start = time.monotonic()
        full_transition.refine_late_boundaries(bundle, pcm, 16000, quran_text.surah_words(item["surah"]),
                                               checks, trace=trace)
        elapsed = time.monotonic() - start
        reference = references.load_reference(item["reference"])
        changes = []
        for old in item["before_checks"]:
            new = checks[old["ayah"]]
            assert new["conf"] == old["conf"] and new["raw_onset_ms"] == old["raw_onset_ms"]
            if new["onset_ms"] != old["onset_ms"]:
                a = old["ayah"]
                changes.append(dict(ayah=a, before_ms=old["onset_ms"], after_ms=new["onset_ms"],
                                    reference_ms=reference[item["surah"], a]))
                for label, cut in (("before", old["onset_ms"]), ("after", new["onset_ms"])):
                    for side, lo, hi in (("left", old["onset_ms"] - 1600, cut),
                                         ("right", cut, old["onset_ms"] + 1600)):
                        path = clips / f"{Path(item['reference']).stem}-{item['surah']}-{a}-{side}-{label}.wav"
                        segment = pcm[max(0, round(lo * 16)):min(len(pcm), round(hi * 16))]
                        with wave.open(str(path), "wb") as output:
                            output.setnchannels(1)
                            output.setsampwidth(2)
                            output.setframerate(16000)
                            output.writeframes((np.clip(segment, -1, 1) * 32767).astype("<i2").tobytes())
        old_errors = {c["ayah"]: abs(c["onset_ms"] - reference[item["surah"], c["ayah"]])
                      for c in item["before_checks"]}
        new_errors = {a: abs(c["onset_ms"] - reference[item["surah"], a]) for a, c in checks.items()}
        assert all(a["onset_ms"] < b["onset_ms"] for a, b in zip(checks.values(), list(checks.values())[1:]))
        reads = trace.get("full_boundary_reads", [])
        for read in reads:
            if read["accepted"]:
                lo, hi = read["safe_interval_ms"]
                assert lo <= read["after_ms"] <= hi
                assert read["after_ms"] < read["before_ms"]
        result = dict(reference=item["reference"], surah=item["surah"], n=item["n"],
                      n_chunks=item["n_chunks"], refinement_s=elapsed, changes=changes,
                      old_over500=sum(e > 500 for e in old_errors.values()),
                      new_over500=sum(e > 500 for e in new_errors.values()),
                      new_over500_ayahs=[a for a, e in new_errors.items() if e > 500 >= old_errors[a]],
                      outcomes=dict(Counter(r.get("reason") for r in reads)), trace=trace)
        assert not result["new_over500_ayahs"]
        assert item["protected_sha256"] == {p: sha(Path(p)) for p in item["protected_sha256"]}
        results.append(result)
        (TASK / "final-verification.json").write_text(json.dumps(results, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "trace"}, ensure_ascii=False), flush=True)
    paths = [Path(p) for p in ("surah_cutter/core/full_transition.py", "surah_cutter/core/full_transition_verifier.py",
                               "surah_cutter/core/full_pause_center.py",
                               "surah_cutter/core/align.py", "surah_cutter/core/align_tiled.py",
                               "surah_cutter/core/full_windows.py", "tests/test_full_transition.py",
                               "tests/test_full_pause_center.py", "tests/test_alignment_audio.py",
                               "tests/test_full_resume.py", "tests/test_full_transition_verifier.py")]
    paths += [TASK / p for p in ("verify.py", "replay.py")]
    (TASK / "source-sha256.json").write_text(json.dumps({str(p): sha(p) for p in paths}, indent=2) + "\n")


if __name__ == "__main__":
    main()
