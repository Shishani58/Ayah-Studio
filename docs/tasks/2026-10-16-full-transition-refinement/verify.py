"""Fresh production FULL; capture the preceding policy's cuts before refinement.

References are opened only after every prediction for a recording completes.
No settings or project/reference DBs are written. --long adds bounded Baqara.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from surah_cutter.core import align, decode, emissions, full_transition
from surah_cutter.data import quran_text, references

TASK = Path(__file__).resolve().parent
ROOT = Path.home() / "Desktop/Abu_Bakr_Ash-Shatri"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--other-reciters", action="store_true")
    parser.add_argument("--suras", nargs="+", type=int, default=[1, 42, 67, 78, 80, 83, 89, 95, 98, 112, 114])
    parser.add_argument("--output", default="verification.json")
    args = parser.parse_args()
    bundle = align._OnnxBundle(emissions.load_emitter(Path("models/wav2vec2-quran-phonetics.onnx")))
    cases = [("abu_bakr_ash_shatri", ROOT, s) for s in args.suras + ([2] if args.long else [])]
    if args.other_reciters:
        recordings = json.loads((TASK.parent / "2026-10-14-full-other-reciters/data-manifest.json").read_text())["recordings"]
        cases += [(r["reference"], Path(r["audio_dir"]), s) for r in recordings for s in (80, 98, 112)]
    original = full_transition.refine_late_boundaries
    captured = {}

    def capture(bundle, pcm, sr, words, checks, **kwargs):
        captured["before"] = deepcopy(checks)
        start = time.monotonic()
        original(bundle, pcm, sr, words, checks, **kwargs)
        captured["refinement_s"] = time.monotonic() - start

    full_transition.refine_late_boundaries = capture
    results = []
    for reference_id, folder, surah in cases:
        audio = folder / f"{surah:03d}.mp3"
        protected = [p for p in folder.glob("*.db")] + [audio]
        hashes = {str(p): sha(p) for p in protected}
        captured.clear()
        pcm = decode.decode(audio)
        words = quran_text.surah_words(surah)
        trace = {}
        start = time.monotonic()
        rows = align.align_surah(pcm, 16000, words, surah=surah, model=bundle, trace=trace)
        elapsed = time.monotonic() - start
        assert [a for a, _, _ in rows] == list(range(1, quran_text.ayah_count(surah) + 1))
        assert all(a[1] < b[1] for a, b in zip(rows, rows[1:]))
        assert hashes == {str(p): sha(p) for p in protected}
        before = captured["before"]
        # No reference is passed to FULL, refinement, or candidate selection.
        reference = references.load_reference(reference_id)
        changes = [dict(ayah=a, before_ms=before[a]["onset_ms"], after_ms=ms,
                        reference_ms=reference[surah, a]) for a, ms, _ in rows
                   if before[a]["onset_ms"] != ms]
        old_errors = [abs(c["onset_ms"] - reference[surah, a]) for a, c in before.items()]
        new_errors = [abs(ms - reference[surah, a]) for a, ms, _ in rows]
        reads = [r for r in trace.get("full_boundary_reads", []) if r.get("role") == "two-sided-cut"]
        for read in reads:
            assert read["end_ms"] - read["start_ms"] <= 12000
            if read["accepted"]:
                lo, hi = read["safe_interval_ms"]
                assert read["pause_start_ms"] < lo <= read["after_ms"] <= hi < read["pause_end_ms"]
                assert read["after_ms"] < read["before_ms"]
        item = dict(reference=reference_id, surah=surah, elapsed_s=elapsed,
                    refinement_s=captured["refinement_s"], n=len(rows), n_chunks=trace.get("n_chunks"),
                    changes=changes, old_over500=sum(e > 500 for e in old_errors),
                    new_over500=sum(e > 500 for e in new_errors),
                    new_over500_ayahs=[a for a, ms, _ in rows
                        if abs(ms - reference[surah, a]) > 500 >= abs(before[a]["onset_ms"] - reference[surah, a])],
                    outcomes=dict(Counter(r.get("reason") for r in reads)), protected_sha256=hashes,
                    rows=rows, before_checks=list(before.values()), trace=trace)
        results.append(item)
        (TASK / args.output).write_text(json.dumps(results, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in item.items() if k not in
                          {"trace", "rows", "before_checks", "protected_sha256"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
