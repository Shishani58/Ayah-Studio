"""Paired comparisons and failure/coverage accounting from saved predictions."""
import csv
import json
from collections import defaultdict

import numpy as np

from common import TASK, corpus, write_json
from score_qpc import summary


def main():
    scores = json.loads((TASK / "qpc-scores.json").read_text())
    rows = scores["records"]
    baseline = {r["record"]: r for r in rows
                if r["context"] == "strong20" and r["variant"] == "current"}
    groups = defaultdict(list)
    for row in rows:
        if row["record"] in baseline:
            groups[row["context"], row["variant"]].append(row)
    output = []
    for (context, variant), items in sorted(groups.items()):
        for split in ("all", "control", "development"):
            selected = [r for r in items if split == "all" or r["split"] == split]
            if not selected:
                continue
            old = np.concatenate([np.abs(baseline[r["record"]]["errors"]) for r in selected])
            new = np.concatenate([np.abs(r["errors"]) for r in selected])
            per_reciter = {}
            for reciter in sorted({r["reciter"] for r in selected}):
                subset = [r for r in selected if r["reciter"] == reciter]
                b = np.concatenate([np.abs(baseline[r["record"]]["errors"]) for r in subset])
                a = np.concatenate([np.abs(r["errors"]) for r in subset])
                per_reciter[reciter] = dict(records=len(subset), baseline=summary(b), method=summary(a))
            rng = np.random.default_rng(20261016)
            blocks = [(np.abs(baseline[r["record"]]["errors"]), np.abs(r["errors"])) for r in selected]
            improvements = []
            for _ in range(2000):
                sample = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
                n = sum(len(a) for a, _ in sample)
                improvements.append(sum(int(np.sum(a > 500))-int(np.sum(b > 500))
                                        for a, b in sample)/n)
            output.append(dict(context=context, variant=variant, split=split, records=len(selected),
                               record_ids=[r["record"] for r in selected], baseline=summary(old),
                               method=summary(new), fixed500=int(np.sum((old > 500) & (new <= 500))),
                               regressed500=int(np.sum((old <= 500) & (new > 500))),
                               better100=int(np.sum(new+100 < old)), worse100=int(np.sum(old+100 < new)),
                               bad_without_review=sum(r["bad_without_review"] for r in selected),
                               review=sum(r["review"] for r in selected), per_reciter=per_reciter,
                               over500_in_same_dsp_pause=sum(r["over500_in_same_dsp_pause"] for r in selected),
                               bootstrap_record_improvement500_rate95=np.percentile(improvements, [2.5, 97.5]).tolist()))
    control_results = []
    for path in sorted((TASK / "qpc-controls/strong20").glob("*.json")):
        item = json.loads(path.read_text())
        if "case" not in item:
            continue
        checks = item.get("trace", {}).get("ayahs", [])
        result = dict(record=item["case"]["id"], variant=item["variant"],
                      failed="error" in item, rows=len(item.get("rows", [])),
                      reliable=sum(bool(c.get("reliable")) for c in checks), checks=len(checks))
        if item["case"]["control"] == "shift" and "rows" in item:
            original = TASK / "strong20" / f"shatri-112-{item['variant']}.json"
            if original.exists():
                old = json.loads(original.read_text())
                if "rows" in old:
                    result["shift_residual_ms"] = [a[1]-b[1]-600 for a, b in zip(item["rows"], old["rows"])]
        control_results.append(result)
    expected = [c["id"] for c in corpus()]
    completeness = {}
    for (context, variant), items in groups.items():
        ids = {r["record"] for r in items}
        completeness[f"{context}/{variant}"] = dict(successes=len(ids),
                                                    absent_from_full44=sorted(set(expected)-ids))
    report = dict(paired=output, failures=scores["failures"], completeness=completeness,
                  controls=control_results, note="Paired metrics exclude failed records; failures are listed separately.")
    write_json(TASK / "paired-analysis.json", report)
    with (TASK / "paired-metrics.csv").open("w") as stream:
        fields = ["context", "variant", "split", "records", "n", "median_ms", "p95_ms", "mean_ms", "max_ms", "over500", "fixed500", "regressed500", "bad_without_review", "review"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in output:
            writer.writerow({**{k:row[k] for k in fields if k in row}, **row["method"]})
    for row in output:
        if row["split"] == "all":
            print(row["context"], row["variant"], row["records"], row["method"],
                  "fixed/regressed", row["fixed500"], row["regressed500"])


if __name__ == "__main__":
    main()
