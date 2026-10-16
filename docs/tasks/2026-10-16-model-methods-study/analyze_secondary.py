"""Compare models on common records and common timestamp coverage, offline."""
import json
from collections import defaultdict

from common import TASK, write_json
from score_secondary import stats


def main():
    saved = json.loads((TASK / "secondary-scores.json").read_text())
    cases = [c for c in json.loads((TASK / "secondary-inputs.json").read_text()) if c["surah"]]
    by_method = defaultdict(dict)
    for row in saved["records"]:
        by_method[row["method"]][row["record"]] = row
    qpc = json.loads((TASK / "qpc-scores.json").read_text())
    ids = {c["id"] for c in cases}
    for row in qpc["records"]:
        if row["record"] in ids and row["context"] == "strong20" and row["variant"] in ("current", "phones"):
            by_method["qpc_"+row["variant"]][row["record"]] = dict(
                record=row["record"], expected_ayahs=len(row["errors"]), covered_ayahs=len(row["errors"]),
                errors_by_ayah={str(i+1): e for i, e in enumerate(row["errors"])})
    baseline = by_method["qpc_current"]
    comparisons = []
    for method, records in sorted(by_method.items()):
        if method in ("greedy", "greedy_sdpa", "greedy_overlap", "beam5", "beam5_wrapper", "nbest5"):
            continue
        pairs = sorted(set(baseline) & set(records))
        old, new = [], []
        expected = matched = 0
        for key in pairs:
            a, b = baseline[key]["errors_by_ayah"], records[key]["errors_by_ayah"]
            common = set(a) & set(b)
            expected += baseline[key]["expected_ayahs"]
            matched += len(common)
            for ayah in common:
                old.append(a[ayah])
                new.append(b[ayah])
        comparisons.append(dict(method=method, records=len(pairs), expected_ayahs=expected,
                                paired_ayahs=matched, missing_ayahs=expected-matched,
                                qpc_on_same_ayahs=stats(old), method_on_same_ayahs=stats(new),
                                nonincreasing_pairs=sum(records[k].get("nonincreasing_pairs",0) for k in pairs)))
    asr_pairs = []
    for method in ("rnnt", "beam5", "token_timestamps", "greedy_sdpa", "greedy_overlap", "nbest5"):
        base_method = "ctc" if method == "rnnt" else "greedy"
        pairs = sorted(set(by_method[method]) & set(by_method[base_method]))
        if pairs:
            a = [by_method[base_method][k] for k in pairs]
            b = [by_method[method][k] for k in pairs]
            n = sum(r["expected_words"] for r in a)
            asr_pairs.append(dict(baseline=base_method, method=method, records=len(pairs), words=n,
                                  baseline_wer=sum(r["edits"] for r in a)/n,
                                  method_wer=sum(r["edits"] for r in b)/n,
                                  improved_records=sum(y["edits"] < x["edits"] for x,y in zip(a,b)),
                                  regressed_records=sum(y["edits"] > x["edits"] for x,y in zip(a,b))))
    write_json(TASK / "secondary-paired.json", dict(timing=comparisons, recognition=asr_pairs,
               failures=saved["failures"], negative=saved["negative"]))
    print(json.dumps(dict(timing=comparisons, recognition=asr_pairs),indent=2))


if __name__ == "__main__":
    main()
