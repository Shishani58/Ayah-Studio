"""Rebuild the read-only matrix report from recorded benchmark cases."""

import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent
data = json.loads((TASK / "surah42-matrix.json").read_text())
groups = {}
for case in data["cases"]:
    groups.setdefault(case["case"], []).append(case)
assert len(data["cases"]) == 33
assert len(groups) == 11
assert all(len(cases) == 3 for cases in groups.values())
assert all(not c["failures"] and len(c["surahs"]) == 1 for c in data["cases"])
baseline = groups["strong:20:3:0"][0]["surahs"][0]
current_data = json.loads(
    (TASK.parent / "2026-10-14-full-pause-windows/surah42-three-options.json").read_text()
)
current = next(c for c in current_data["cases"] if c["case"] == "fixed:30:1:0")["surahs"][0]
assert current["audio_sha256"] == baseline["audio_sha256"]
summary = {}
all_sources_ok = all(
    hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in data["sources"].items()
)
all_inputs_equal = all(len({c[key] for c in data["cases"]}) == 1 for key in (
    "model_sha256", "vocab_sha256", "meta_sha256", "reference_sha256",
)) and len({c["surahs"][0]["audio_sha256"] for c in data["cases"]}) == 1
assert all_sources_ok and all_inputs_equal
for name, cases in groups.items():
    ss = [c["surahs"][0] for c in cases]
    first = ss[0]
    assert all([r[0] for r in s["rows"]] == list(range(1, 54)) for s in ss)
    assert all(all(b[1] > a[1] for a, b in zip(s["rows"], s["rows"][1:])) for s in ss)
    for s in ss:
        accepted = [p for p in s["windows"]["pause_decisions"] if p["accepted"]]
        assert s["windows"]["fallback_cuts"] == 0
        for w in s["windows"]["windows"][:-1]:
            assert w["cut_reason"] == "pause"
            assert any(p["start_ms"] <= w["end_ms"] <= p["end_ms"] for p in accepted)
    times = [s["full_s"] for s in ss]
    rss = [c["peak_rss_mib"] for c in cases]
    new_large = [
        {"ayah": i, "error_ms": e, "baseline_ms": old, "flagged": i in first["review"]}
        for i, (e, old) in enumerate(zip(first["errors_ms"], baseline["errors_ms"]), 1)
        if abs(e) > 500 and abs(old) <= 500
    ]
    new_large_current = [
        {"ayah": i, "error_ms": e, "current_ms": old, "flagged": i in first["review"]}
        for i, (e, old) in enumerate(zip(first["errors_ms"], current["errors_ms"]), 1)
        if abs(e) > 500 and abs(old) <= 500
    ]
    summary[name] = {
        "full_times_s": times, "median_full_s": statistics.median(times),
        "range_full_s": [min(times), max(times)],
        "median_peak_rss_mib": statistics.median(rss), "max_peak_rss_mib": max(rss),
        "peak_rss_mib": rss, "errors": first["stats"],
        "mean_abs_error_ms": statistics.mean(map(abs, first["errors_ms"])),
        "new_large_errors_vs_strong20": new_large,
        "new_large_errors_vs_current": new_large_current,
        "changed_ayahs_vs_strong20": [
            {"ayah": i, "baseline_error_ms": old, "error_ms": e, "delta_ms": e - old}
            for i, (e, old) in enumerate(zip(first["errors_ms"], baseline["errors_ms"]), 1)
            if e != old
        ],
        "max_abs_error_worsening_vs_strong20_ms": max(
            abs(e) - abs(old) for e, old in zip(first["errors_ms"], baseline["errors_ms"])
        ),
        "unflagged_large_errors": [i for i, e in enumerate(first["errors_ms"], 1)
                                   if abs(e) > 500 and i not in first["review"]],
        "windows": len(first["windows"]["windows"]),
        "max_window_s": max((w["end_ms"] - w["start_ms"]) / 1000
                            for w in first["windows"]["windows"]),
        "emissions_repeat_exact": len({s["emissions_sha256"] for s in ss}) == 1,
        "rows_repeat_exact": all(s["rows"] == first["rows"] for s in ss),
        "review_repeat_exact": all(s["review"] == first["review"] for s in ss),
        "windows_repeat_exact": all(s["windows"]["windows"] == first["windows"]["windows"]
                                    for s in ss),
    }
for seconds in (25, 30):
    ss = [c["surahs"][0] for name, cs in groups.items()
          if name.startswith(f"strong:{seconds}:") for c in cs]
    assert len({s["emissions_sha256"] for s in ss}) == 1
    assert all(s["rows"] == ss[0]["rows"] and s["review"] == ss[0]["review"] for s in ss)
control_time = summary["strong:20:3:0"]["median_full_s"]
for entry in summary.values():
    entry["time_reduction_vs_strong20_percent"] = 100 * (1 - entry["median_full_s"] / control_time)
out = {"groups": summary, "all_sources_unchanged": all_sources_ok,
       "all_inputs_equal": all_inputs_equal, "all_cuts_in_accepted_pauses": True,
       "same_duration_identical_across_workers": True,
       "current_quality_reference_only": "surah42-three-options.json fixed:30:1:0"}
(TASK / "surah42-analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print("case | median s | range s | median/max peak MiB | median/p95/max error ms | new >500")
for name in sorted(summary, key=lambda n: tuple(map(int, n.split(":")[1:]))):
    g = summary[name]
    print(f"{name} | {g['median_full_s']:.2f} | "
          f"{g['range_full_s'][0]:.2f}–{g['range_full_s'][1]:.2f} | "
          f"{g['median_peak_rss_mib']:.0f}/{g['max_peak_rss_mib']:.0f} | "
          f"{g['errors']['median_ms']}/{g['errors']['p95_ms']:.1f}/{g['errors']['max_ms']} | "
          f"{g['new_large_errors_vs_strong20']}")
