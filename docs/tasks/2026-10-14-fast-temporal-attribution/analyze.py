"""Retrospective temporal attribution; never change timings or dispatch a model."""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np

task = Path(__file__).parent.resolve()
root = task.parents[2]
source = task.parent / "2026-10-14-fast-late-future-anchor/late-corpus.json"
cases = json.loads(source.read_text())
reference = root / "data/references/abu_bakr_ash_shatri.db"
reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
with sqlite3.connect(f"file:{reference}?mode=ro", uri=True) as connection:
    truth = {(s, a): t for s, a, t in connection.execute("SELECT sura,ayah,time FROM timings")}
rows = []
for c in cases:
    assert c["reference_sha256"] == reference_hash
    for b in c["trace"]["boundaries"]:
        if b["confidence"] <= 0:
            continue
        p = b["proof"]
        r = c["trace"]["acoustic_reads"][b["read_id"]]
        observed = r["observed"]
        h = p.get("decoder_hypothesis")
        view = observed if h is None else observed["hypotheses"][h]
        offset = p["offset"]
        reference_ms = truth[c["surah"], b["ayah"]]
        phrase = p["phrase"]
        # A position is comparable only for the independently selected exact phrase.
        exact = view["units"][offset:offset + len(phrase)] == phrase
        native = round(r["window_ms"][0] + view["positions_ms"][offset]) if exact else None
        row = {"surah": c["surah"], "ayah": b["ayah"], "kind": b["evidence_kind"],
               "confidence": b["confidence"], "boundary_ms": b["start_ms"],
               "reference_ms": reference_ms, "signed_error_ms": b["start_ms"] - reference_ms,
               "native_observed_ms": native, "native_signed_error_ms": None if native is None else native - reference_ms,
               "read_id": b["read_id"], "read_start_ms": r["window_ms"][0], "exact_selected_phrase": exact}
        if "pause_id" in p:
            g = c["trace"]["pause_geometry"][p["pause_id"]]
            left, right = p["pause_ms"]
            row.update(pause_start_ms=left, pause_end_ms=right, pause_width_ms=right-left,
                       gt_in_pause=left <= reference_ms <= right,
                       gt_fraction=(reference_ms-left)/(right-left),
                       stationary=g.get("quiet_stationarity"), geometry_accepted=g["accepted"],
                       geometry_reason=g["reason"])
        rows.append(row)

def stats(items, key="signed_error_ms"):
    values = [r[key] for r in items if r.get(key) is not None]
    return {"count": len(values), "median_abs_ms": float(np.median(np.abs(values))),
            "p95_abs_ms": float(np.percentile(np.abs(values),95)),
            "median_signed_ms": float(np.median(values)), "outliers_gt500": sum(abs(v)>500 for v in values)} if values else {"count":0}
summary = {"scope": "Retained independent evidence, reference read-only; no inference; no automatic calibration",
           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "reference_sha256": reference_hash,
           "all": stats(rows), "by_kind": {}, "pause_partitions": {}}
for kind in sorted({r["kind"] for r in rows}):
    selected = [r for r in rows if r["kind"] == kind]
    summary["by_kind"][kind] = {"boundary":stats(selected), "observed_native_counterfactual":stats(selected,"native_signed_error_ms")}
pauses = [r for r in rows if r["kind"] == "pause"]
for name, selected in [("gt_inside",[r for r in pauses if r["gt_in_pause"]]),
                       ("gt_outside",[r for r in pauses if not r["gt_in_pause"]]),
                       ("stationary",[r for r in pauses if r["stationary"]]),
                       ("nonstationary",[r for r in pauses if not r["stationary"]]),
                       ("width_gt1000",[r for r in pauses if r["pause_width_ms"]>1000]),
                       ("width_le1000",[r for r in pauses if r["pause_width_ms"]<=1000])]:
    summary["pause_partitions"][name] = stats(selected)
summary["outliers"] = [r for r in rows if abs(r["signed_error_ms"])>500]
summary["outside_pause"] = [r for r in pauses if not r["gt_in_pause"]]
(task / "attribution.json").write_text(json.dumps({"summary":summary,"rows":rows},ensure_ascii=False,indent=2)+"\n")
keys = sorted({k for r in rows for k in r})
with (task / "attribution.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k not in ["outliers","outside_pause"]},ensure_ascii=False,indent=2))
print("outliers",[(r["surah"],r["ayah"],r["kind"],r["signed_error_ms"],r.get("gt_fraction")) for r in summary["outliers"]])
print("outside_pause",[(r["surah"],r["ayah"],r["signed_error_ms"],r["gt_fraction"]) for r in summary["outside_pause"]])
