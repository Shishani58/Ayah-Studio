"""Recompute quality/coverage from final artifacts, without inference or DB access."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def metrics(items):
    values = [(i, a, confidence, error) for i in items
              for (a, _, confidence), error in zip(i["rows"], i["errors_ms"])]
    confirmed = [(i, a, confidence, error) for i, a, confidence, error in values
                 if confidence > 0 and a not in i["trace"]["review_required_ayahs"]]
    errors = [v[3] for v in values]
    return {"recordings": len(items), "ayahs": len(values), "confirmed": len(confirmed),
            "review": sum(a in i["trace"]["review_required_ayahs"] for i, a, _, _ in values),
            "confirmed_over500": [[i["surah"], a, error] for i, a, _, error in confirmed if error > 500],
            "all_over500": sum(error > 500 for error in errors),
            "median_ms": float(np.median(errors)), "p95_ms": float(np.percentile(errors, 95)),
            "max_ms": max(errors), "first_confirmed": sum(a == 1 for _, a, _, _ in confirmed),
            "model_reads": sum(len(i["actual_dispatches"]) for i in items),
            "local_model_seconds": sum(r["seconds"] for i in items for r in i["trace"]["acoustic_reads"])
            if all("seconds" in r for i in items for r in i["trace"]["acoustic_reads"]) else None,
            "boundary_kinds": dict(Counter(b["evidence_kind"] for i in items for b in i["trace"].get("boundaries", []))),
            "reason_counts": dict(Counter(b["review_reason"] for i in items for b in i["trace"].get("boundaries", [])
                                          if b["evidence_kind"] == "estimate")),
            "max_local_ms": max((r["samples"] / 16 for i in items for r in i["actual_dispatches"]), default=0),
            "max_union_ratio": max(i["trace"]["coverage"] for i in items),
            "max_sum_ratio": max(i["trace"]["audio_ms"] / i["duration_ms"] for i in items),
            "peak_accounted_cache_bytes": max(i["trace"].get("peak_cache_bytes", 0) for i in items),
            "sequence_violations": [v for i in items for v in i["sequence_violations"]]}


def main():
    new = json.loads((ROOT / "corpus-final.json").read_text())
    old = json.loads((ROOT.parent / "2026-10-14-fast-full-execution/corpus-legacy.json").read_text())
    previous = json.loads((ROOT / "corpus-before-hint-fix.json").read_text())
    models = {"quran-phonetics", "whisper-tarteel-tiny", "whisper-tarteel-base"}
    surahs = {1, 2, 9, 19, 20, 36, 50, 55, 77, *range(78, 115)}
    assert len(new) == 138 and {(i["model"], i["surah"]) for i in new} == {(m, s) for m in models for s in surahs}
    result = {}
    for model in sorted(models):
        current = [i for i in new if i["model"] == model]
        legacy = [i for i in old if i["model"] == model]
        paired = [i for i in current if i["surah"] in {j["surah"] for j in legacy}]
        for i in paired:
            j = next(j for j in legacy if j["surah"] == i["surah"])
            assert all(i[k] == j[k] for k in ("audio_sha256", "initial_pauses_sha256", "reference_sha256"))
        n, o = metrics(paired), metrics(legacy)
        previous_rows = {(i["surah"], a): (confidence > 0 and a not in i["trace"]["review_required_ayahs"], error)
                         for i in legacy for (a, _, confidence), error in zip(i["rows"], i["errors_ms"])}
        new_disagreements = [entry for entry in n["confirmed_over500"]
                             if not previous_rows[tuple(entry[:2])][0] or previous_rows[tuple(entry[:2])][1] <= 500]
        result[model] = {"all46": metrics(current), "previous_iteration46": metrics([i for i in previous if i["model"] == model]),
                         "paired37_new": n, "paired37_legacy": o,
                         "coverage_gate": n["confirmed"] >= o["confirmed"],
                         "false_confident_gate": len(n["confirmed_over500"]) <= len(o["confirmed_over500"]),
                         "new_unlistened_disagreements": new_disagreements,
                         "no_new_unlistened_disagreement_gate": not new_disagreements}
        assert result[model]["all46"]["ayahs"] == 1475
    (ROOT / "paired-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({m: {"all46": v["all46"]["confirmed"], "paired37": v["paired37_new"]["confirmed"],
                          "legacy": v["paired37_legacy"]["confirmed"], "over500": v["all46"]["confirmed_over500"],
                          "coverage_gate": v["coverage_gate"]} for m, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
