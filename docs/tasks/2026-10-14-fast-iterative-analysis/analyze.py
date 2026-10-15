"""Post-inference diagnosis; reference never influences executor decisions."""
import json
from pathlib import Path
import sqlite3
from collections import Counter

root = Path(__file__).parent
with sqlite3.connect("file:data/references/abu_bakr_ash_shatri.db?mode=ro", uri=True) as db:
    truth = {(s, a): t for s, a, t in db.execute("select sura,ayah,time from timings")}
report = {}
for path in sorted(root.glob("*.json")):
    if path.stem in ("analysis",):
        continue
    items = json.loads(path.read_text())
    if not isinstance(items, list) or not items or "trace" not in items[0]:
        continue
    cases = []
    for item in items:
        trace, s = item["trace"], item["surah"]
        evaluations = [e for e in trace["transitions"] if e["state"] == "EVALUATE"]
        failures = [e for e in trace["transitions"] if e["state"] == "REVIEW_REQUIRED"]
        nearest_reads = []
        for failure in failures:
            ayah = failure["ayah"]
            target = truth[s, ayah]
            local = [r for r in trace["acoustic_reads"] if r["reason"].startswith(f"ayah={ayah}:")]
            for read in sorted(local, key=lambda r: abs(r["window_ms"][0] - target))[:3]:
                nearest_reads.append({"ayah": ayah, "reference_ms": target,
                    "window_ms": read["window_ms"], "text": read["observed"].get("text"),
                    "truncated": read["observed"].get("truncated"),
                    "unit_quality": read["observed"].get("unit_quality"),
                    "reason": read["reason"]})
        confirmed = [b for b in trace["boundaries"] if b["confidence"] > 0]
        errors = [abs(t-truth[s,a]) for a,t,c in item["rows"]]
        cases.append({"model": item["model"], "surah": s, "confirmed": len(confirmed),
            "total": len(item["rows"]), "calls": len(trace["acoustic_reads"]),
            "seconds": item["seconds"],
            "inference_seconds": sum(r["seconds"] for r in trace["acoustic_reads"]),
            "cache_hits": trace["cache_hits"], "coverage": trace["coverage"],
            "over500": sum(e > 500 for e in errors),
            "false_confident": [[b["ayah"],b["start_ms"]-truth[s,b["ayah"]]] for b in confirmed if abs(b["start_ms"]-truth[s,b["ayah"]])>500],
            "failures": failures, "nearest_reads": nearest_reads,
            "expansion_reasons": dict(Counter(e.get("acoustic_anchor") for e in trace["transitions"] if e["state"]=="EXPAND_LOCAL")),
            "fully_heard_evaluations": sum("needed_units" in e and e.get("complete_units")==e["needed_units"] for e in evaluations)})
    report[path.name] = cases
(root / "analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
for name,cases in report.items():
    print(name)
    for model in sorted({c["model"] for c in cases}):
        subset=[c for c in cases if c["model"]==model]
        print(model,"cases",len(subset),"proofs",sum(c["confirmed"] for c in subset),"calls",sum(c["calls"] for c in subset),"errors>500",sum(c["over500"] for c in subset),"confident errors",[(c["surah"],c["false_confident"]) for c in subset if c["false_confident"]])
