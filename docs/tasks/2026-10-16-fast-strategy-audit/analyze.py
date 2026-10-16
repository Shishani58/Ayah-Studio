"""Recompute strategy-level metrics from frozen reports; no acoustic inference."""

import hashlib
import json
from pathlib import Path

from surah_cutter.data.quran_text import surah_ayah_counts


def main():
    root = Path(__file__).resolve().parents[3]
    folder = Path(__file__).resolve().parent
    previous = root / "docs/tasks/2026-10-16-fast-addressed-recovery"
    inputs = [previous / name for name in (
        "SF-per-record.json", "SF-corpus-gate.json", "PF-legacy37-comparison.json",
    )]
    records = json.loads(inputs[0].read_text())
    gate = json.loads(inputs[1].read_text())
    legacy = json.loads(inputs[2].read_text())
    counts = surah_ayah_counts()
    wide = [r for r in records if r["reciter"] == "shatri"]
    assert len(wide) == 114
    assert {r["surah"] for r in wide} == set(counts)
    assert sum(counts.values()) == 6236
    fields = ("before_proofs", "after_proofs", "before_calls", "after_calls",
              "before_input_ms", "after_input_ms", "after_union_ms", "duration_ms",
              "after_bad", "after_all_over500", "after_blocked_ayahs")
    totals = {key: sum(r[key] for r in wide) for key in fields}
    for key, value in totals.items():
        assert abs(value - gate["quality"]["shatri"][key]) < 1e-5, key
    total_ayahs = sum(counts.values())
    totals.update(
        ayahs=total_ayahs,
        unconfirmed=total_ayahs - totals["after_proofs"],
        confirmed_fraction=totals["after_proofs"] / total_ayahs,
        reference_over500_fraction=totals["after_all_over500"] / total_ayahs,
        union_fraction=totals["after_union_ms"] / totals["duration_ms"],
        submitted_fraction=totals["after_input_ms"] / totals["duration_ms"],
    )
    buckets = {}
    for label, predicate in (
        ("at_least_100_ayahs", lambda n: n >= 100),
        ("20_to_99_ayahs", lambda n: 20 <= n < 100),
        ("under_20_ayahs", lambda n: n < 20),
    ):
        subset = [r for r in wide if predicate(counts[r["surah"]])]
        n_ayahs = sum(counts[r["surah"]] for r in subset)
        n_proofs = sum(r["after_proofs"] for r in subset)
        buckets[label] = dict(records=len(subset), ayahs=n_ayahs, proofs=n_proofs,
                              confirmed_fraction=n_proofs / n_ayahs)
    by_reciter = {}
    for reciter in sorted({r["reciter"] for r in records}):
        subset = [r for r in records if r["reciter"] == reciter]
        n_ayahs = sum(counts[r["surah"]] for r in subset)
        n_proofs = sum(r["after_proofs"] for r in subset)
        by_reciter[reciter] = dict(records=len(subset), ayahs=n_ayahs,
                                  proofs=n_proofs, confirmed_fraction=n_proofs / n_ayahs)
    tracked = [
        "surah_cutter/core/fast_executor.py", "surah_cutter/core/fast_adapters.py",
        "surah_cutter/core/fast_reads.py", "surah_cutter/core/pipeline.py",
        "surah_cutter/data/alignment_result.py", "surah_cutter/data/project_db.py",
        "surah_cutter/data/export_db.py", "data/quran_mushaf.db",
    ]
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    result = {
        "measurement_kind": "independent aggregation of frozen sparse-14 artifacts",
        "new_acoustic_calls": 0,
        "physical_accuracy_measured": False,
        "input_sha256": {str(p.relative_to(root)): sha(p) for p in inputs},
        "inspected_source_sha256": {p: sha(root / p) for p in tracked},
        "shatri": totals,
        "zero_proof_surahs": [r["surah"] for r in wide if not r["after_proofs"]],
        "no_gain_records": sum(r["before_proofs"] == r["after_proofs"] for r in wide),
        "length_buckets": buckets,
        "reciters": by_reciter,
        "legacy37": legacy["summary"],
        "examples": [dict(surah=r["surah"], ayahs=counts[r["surah"]],
                          **{k: r[k] for k in (
                              "after_proofs", "after_calls", "after_union_ms",
                              "duration_ms", "after_largest_review_block")})
                     for r in wide if r["surah"] in (2, 3, 5, 78, 80)],
    }
    (folder / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if not k.endswith("sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
