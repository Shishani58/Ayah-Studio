"""Rebuild the reciter audit from completed, immutable benchmark records."""

import hashlib
import json
from pathlib import Path
import statistics


TASK = Path(__file__).resolve().parent
manifest = json.loads((TASK / "data-manifest.json").read_text())
runtime = json.loads((TASK / "runtime-manifest.json").read_text())
runtime_path = Path(runtime["runtime"])
runtime_unchanged = all(hashlib.sha256((runtime_path / path).read_bytes()).hexdigest() == sha
                        for path, sha in runtime["files"].items())
assert runtime_unchanged
expected_cases = {"fixed:30:1:0"} | {
    f"strong:{seconds}:{workers}:0" for seconds in (10, 20, 30) for workers in (1, 3)
}
all_results = {}
all_parity = []
all_regressions = []
all_model_hashes = set()
aggregate = {name: [] for name in expected_cases}


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors_summary(values):
    import numpy as np

    absolute = np.abs(values)
    return {"n": len(values), "median_ms": float(np.median(absolute)),
            "p95_ms": float(np.percentile(absolute, 95)), "max_ms": int(max(absolute)),
            "mean_ms": statistics.mean(map(abs, values)),
            "over500": sum(abs(v) > 500 for v in values)}


def differences(reciter, base, trial, comparison):
    result = []
    for surah, item in trial.items():
        previous = base[surah]
        cuts = [w["end_ms"] for w in item["windows"].get("windows", [])[:-1]]
        for row, error, old in zip(item["rows"], item["errors_ms"], previous["errors_ms"]):
            if abs(error) > 500 and abs(old) <= 500:
                reference_ms = row[1] - error
                entry = {"reciter": reciter, "comparison": comparison, "surah": surah,
                         "ayah": row[0], "previous_error_ms": old, "error_ms": error,
                         "flagged": row[0] in item["review"],
                         "reference_distance_to_cut_ms": min(
                             (abs(reference_ms - cut) for cut in cuts), default=None),
                         "onset_check": next(c for c in item["onset_checks"]
                                             if c["ayah"] == row[0])}
                result.append(entry)
    return result


for rec in manifest["recordings"]:
    reciter = rec["name"]
    data = json.loads((TASK / (reciter + ".json")).read_text())
    cases = {c["case"]: c for c in data["cases"]}
    assert set(cases) == expected_cases
    assert all(not c["failures"] for c in cases.values())
    assert all(rec["all_reference_rows_match_packaged"] for rec in manifest["recordings"])
    assert all(digest_file(f["path"]) == f["sha256"] for f in rec["files"])
    assert all(c["reference_sha256"] == digest_file(rec["reference"]) for c in cases.values())
    all_model_hashes.update(c["model_sha256"] for c in cases.values())
    audio_hashes = {int(Path(f["path"]).stem): f["sha256"]
                    for f in rec["files"] if Path(f["path"]).suffix == ".mp3"}
    assert all(s["audio_sha256"] == audio_hashes[s["surah"]]
               for c in cases.values() for s in c["surahs"])
    assert all(len({c[key] for c in cases.values()}) == 1 for key in (
        "model_sha256", "vocab_sha256", "meta_sha256", "onnxruntime",
    ))
    ss = {name: {s["surah"]: s for s in c["surahs"]} for name, c in cases.items()}
    for name, by_surah in ss.items():
        assert set(by_surah) == {80, 98, 112}
        for surah, item in by_surah.items():
            count = {80: 42, 98: 8, 112: 4}[surah]
            assert [r[0] for r in item["rows"]] == list(range(1, count + 1))
            assert all(b[1] > a[1] for a, b in zip(item["rows"], item["rows"][1:]))
            aggregate[name].extend(item["errors_ms"])
            if name.startswith("strong"):
                assert item["windows"].get("fallback_cuts", 0) == 0
                accepted = [p for p in item["windows"].get("pause_decisions", [])
                            if p["accepted"]]
                assert all(w["cut_reason"] == "pause"
                           and any(p["start_ms"] <= w["end_ms"] <= p["end_ms"]
                                   for p in accepted)
                           for w in item["windows"]["windows"][:-1])
    parity = []
    for seconds in (10, 20, 30):
        sequential = ss[f"strong:{seconds}:1:0"]
        parallel = ss[f"strong:{seconds}:3:0"]
        for surah in sequential:
            a, b = sequential[surah], parallel[surah]
            checks = {"emissions_sha_equal": a["emissions_sha256"] == b["emissions_sha256"],
                      "rows_confidence_equal": a["rows"] == b["rows"],
                      "review_equal": a["review"] == b["review"],
                      "raw_review_equal": a["raw_review"] == b["raw_review"],
                      "onset_checks_equal": a["onset_checks"] == b["onset_checks"],
                      "windows_equal": a["windows"]["windows"] == b["windows"]["windows"]}
            assert all(checks.values()), (reciter, seconds, surah, checks)
            entry = {"reciter": reciter, "target_s": seconds, "surah": surah, **checks}
            entry["windows_count"] = len(a["windows"]["windows"])
            parity.append(entry)
            all_parity.append(entry)
    stats = {}
    for name, case in cases.items():
        values = [e for s in case["surahs"] for e in s["errors_ms"]]
        large = [{"surah": s["surah"], "ayah": r[0], "error_ms": e,
                  "flagged": r[0] in s["review"]}
                 for s in case["surahs"] for r, e in zip(s["rows"], s["errors_ms"])
                 if abs(e) > 500]
        stats[name] = {"full_s": case["full_s"], "peak_rss_mib": case["peak_rss_mib"],
                       "errors": errors_summary(values), "large_errors": large,
                       "unflagged_large": sum(not e["flagged"] for e in large)}
    regressions = []
    regressions.extend(differences(reciter, ss["fixed:30:1:0"], ss["strong:20:3:0"],
                                  "new20_vs_old_fixed30"))
    for seconds in (10, 30):
        regressions.extend(differences(reciter, ss["strong:20:3:0"], ss[f"strong:{seconds}:3:0"],
                                      f"{seconds}_vs20"))
    all_regressions.extend(regressions)
    all_results[reciter] = {"cases": stats, "parity": parity,
                           "regressions": regressions,
                           "20_workers3_time_reduction_vs1_percent": 100 * (
                               1 - cases["strong:20:3:0"]["full_s"] / cases["strong:20:1:0"]["full_s"])}

assert len(all_model_hashes) == 1
out = {"recordings": all_results,
       "aggregate_errors": {name: errors_summary(values) for name, values in aggregate.items()},
       "all_worker_parity_checks": all_parity, "parity_pairs": len(all_parity),
       "all_worker_parity_exact": True, "frozen_runtime_unchanged": runtime_unchanged,
       "multi_window_parity_pairs": sum(p["windows_count"] > 1 for p in all_parity),
       "at_least_three_window_pairs": sum(p["windows_count"] >= 3 for p in all_parity),
       "model_sha256": next(iter(all_model_hashes)),
       "all_new_large_errors": all_regressions, "unique_ayah_starts": 216,
       "model_predictions": 216 * 7, "reciters": 4, "surahs_each": [80, 98, 112]}
(TASK / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print("worker parity", len(all_parity), "pairs exact")
for name, result in all_results.items():
    print(name)
    for case in ("fixed:30:1:0", "strong:10:3:0", "strong:20:1:0",
                 "strong:20:3:0", "strong:30:3:0"):
        s = result["cases"][case]
        print(case, round(s["full_s"], 2), s["errors"], "unflagged", s["unflagged_large"])
print("regressions", [(e["reciter"], e["comparison"], e["surah"], e["ayah"],
                       e["previous_error_ms"], e["error_ms"], e["flagged"])
                      for e in all_regressions])
