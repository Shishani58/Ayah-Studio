"""Paired proof preservation and direct-vs-cascaded failures after inference."""
import json
from pathlib import Path
import sqlite3

task = Path(__file__).parent.resolve()
root = task.parents[2]
with sqlite3.connect(f"file:{root / 'data/references/abu_bakr_ash_shatri.db'}?mode=ro", uri=True) as db:
    truth = {(s, a): t for s, a, t in db.execute("SELECT sura,ayah,time FROM timings")}
control = {i["surah"]: i for i in json.loads((task / "control.json").read_text())}
report = {}
for name in ("short_initial", "short_expansion", "short_both"):
    path = task / (name + ".json")
    if not path.exists():
        continue
    items = json.loads(path.read_text())
    cases = []
    for item in items:
        surah = item["surah"]
        old = control[surah]
        before = {a: (ms, c) for a, ms, c in old["rows"]}
        after = {a: (ms, c) for a, ms, c in item["rows"]}
        lost = [a for a in before if before[a][1] > 0 and after[a][1] == 0]
        added = [a for a in before if before[a][1] == 0 and after[a][1] > 0]
        violations = [[a, ms, truth[surah, a], c] for a, (ms, c) in after.items()
                      if c > 0 and abs(ms - truth[surah, a]) > 500]
        details = []
        for ayah in sorted(set(lost + added)):
            pair = []
            for label, case in (("control", old), (name, item)):
                trace = case["trace"]
                local = [r for r in trace["acoustic_reads"] if r["reason"].startswith(f"ayah={ayah}:")]
                pair.append({"variant": label, "row": next(row for row in case["rows"] if row[0] == ayah),
                             "read_attempted": bool(local),
                             "reads": [{"window_ms": r["window_ms"], "reason": r["reason"],
                                        "text": r["observed"]["text"],
                                        "unit_quality": r["observed"]["unit_quality"]} for r in local],
                             "events": [e for e in trace["transitions"] if e.get("ayah") == ayah
                                        and e["state"] in ("REVIEW_REQUIRED", "EXPAND_LOCAL", "EVALUATE")]})
            details.append({"ayah": ayah, "change": "lost" if ayah in lost else "added",
                            "reference_ms": truth[surah, ayah], "paired": pair})
        cases.append({"surah": surah, "proofs_before": sum(c > 0 for _, c in before.values()),
                      "proofs_after": sum(c > 0 for _, c in after.values()), "lost": lost, "added": added,
                      "confident_outliers": violations, "seconds_before": old["seconds"],
                      "seconds_after": item["seconds"],
                      "calls_before": len(old["actual_dispatches"]), "calls_after": len(item["actual_dispatches"]),
                      "same_initial_pauses": old["initial_pauses_sha256"] == item["initial_pauses_sha256"],
                      "same_audio": old["audio_sha256"] == item["audio_sha256"], "changes": details})
    report[name] = {"cases": cases, "summary": {
        "cases": len(cases), "proofs_before": sum(c["proofs_before"] for c in cases),
        "proofs_after": sum(c["proofs_after"] for c in cases),
        "lost": sum(len(c["lost"]) for c in cases), "added": sum(len(c["added"]) for c in cases),
        "confident_outliers": sum(len(c["confident_outliers"]) for c in cases),
        "seconds_before": sum(c["seconds_before"] for c in cases),
        "seconds_after": sum(c["seconds_after"] for c in cases),
        "same_inputs": all(c["same_initial_pauses"] and c["same_audio"] for c in cases)}}
(task / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({name: value["summary"] for name, value in report.items()}, indent=2))
