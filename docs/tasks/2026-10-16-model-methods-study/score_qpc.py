"""Evaluate saved completed predictions; this process performs no inference."""
import argparse
import json
from collections import defaultdict

import numpy as np

from common import TASK, write_json
from surah_cutter.data import references


def summary(errors):
    a = np.abs(errors)
    return {"n": len(a), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(np.mean(a)), "max_ms": float(np.max(a)), "over500": int(np.sum(a > 500))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", default=["strong20"])
    args = parser.parse_args()
    groups = defaultdict(list)
    failures = []
    details = []
    pause_cache = {}
    for folder in args.folders:
        for path in sorted((TASK / folder).glob("*.json")):
            item = json.loads(path.read_text())
            if "case" not in item:
                continue
            case = item["case"]
            key = (item["context"], item["variant"], case["split"])
            if "error" in item:
                failures.append({"record": case["id"], "method": list(key), "error": item["error"]})
                continue
            truth = references.load_reference(case["reciter"])
            errors = [ms - truth[case["surah"], a] for a, ms, _ in item["rows"]]
            if case["id"] not in pause_cache:
                base_path = TASK / "strong20" / f"{case['id']}-current.json"
                base = json.loads(base_path.read_text()) if base_path.exists() else item
                pause_cache[case["id"]] = [
                    (p["start_ms"], p["end_ms"])
                    for p in base.get("trace", {}).get("full_windows", {}).get("pause_decisions", [])
                    if p.get("accepted") is True]
            shared_pause = [a for (a, ms, _), e in zip(item["rows"], errors)
                            if abs(e) > 500 and any(lo <= ms <= hi and lo <= truth[case["surah"], a] <= hi
                                                   for lo, hi in pause_cache[case["id"]])]
            checks = {c["ayah"]: c for c in item["trace"].get("ayahs", [])}
            missed = sum(abs(e) > 500 and checks.get(a, {}).get("reliable", False)
                         for (a, *_), e in zip(item["rows"], errors))
            detail = {"record": case["id"], "reciter": case["reciter"], "split": case["split"],
                      "context": item["context"], "variant": item["variant"], "errors": errors,
                      "stats": summary(errors), "bad_without_review": missed,
                      "over500_in_same_dsp_pause": len(shared_pause), "same_pause_ayahs": shared_pause,
                      "baseline_dsp_intervals": len(pause_cache[case["id"]]),
                      "review": sum(not checks.get(a, {}).get("reliable", False) for a, *_ in item["rows"]),
                      "wall_s": item["wall_s"], "actual_calls": sum(not c["reused"] for c in item["calls"]),
                      "logical_calls": len(item["calls"]), "path": str(path)}
            details.append(detail)
            groups[key].append(detail)
    aggregated = []
    for key, records in sorted(groups.items()):
        aggregated.append({"context": key[0], "variant": key[1], "split": key[2],
                           "records": len(records), "stats": summary(sum([r["errors"] for r in records], [])),
                           "bad_without_review": sum(r["bad_without_review"] for r in records),
                           "over500_in_same_dsp_pause": sum(r["over500_in_same_dsp_pause"] for r in records),
                           "review": sum(r["review"] for r in records)})
    paired = []
    baseline = {r["record"]: r for r in details if r["context"] == "strong20" and r["variant"] == "current"}
    for row in details:
        base = baseline.get(row["record"])
        if not base:
            continue
        old, new = np.abs(base["errors"]), np.abs(row["errors"])
        paired.append({"record": row["record"], "context": row["context"], "variant": row["variant"],
                       "fixed_over500": [i + 1 for i in np.flatnonzero((old > 500) & (new <= 500)).tolist()],
                       "new_over500": [i + 1 for i in np.flatnonzero((old <= 500) & (new > 500)).tolist()],
                       "worse_over100": int(np.sum(new > old + 100)),
                       "better_over100": int(np.sum(new + 100 < old))})
    report = {"agreement_target": "packaged cut reference, not human onset GT",
              "summary": aggregated, "failures": failures, "records": details, "paired": paired}
    write_json(TASK / "qpc-scores.json", report)
    print(json.dumps(aggregated, ensure_ascii=False, indent=2))
    print("failures", len(failures))


if __name__ == "__main__":
    main()
