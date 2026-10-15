"""Independent arithmetic on historical evidence, no new inference/replay."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).parent
new_path = ROOT.parent / "2026-10-14-fast-review-fixes/corpus-final.json"
old_path = ROOT.parent / "2026-10-14-fast-full-execution/corpus-legacy.json"
new, old = (json.loads(path.read_text()) for path in (new_path, old_path))
reference = Path("data/references/abu_bakr_ash_shatri.db")
with sqlite3.connect(f"{reference.resolve().as_uri()}?mode=ro", uri=True) as conn:
    truth = {(s, a): t for s, a, t in conn.execute("SELECT sura,ayah,time FROM timings")}
ref_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
for item in new + old:
    assert item["reference_sha256"] == ref_sha
    assert [abs(t - truth[item["surah"], a]) for a, t, _ in item["rows"]] == item["errors_ms"]
for item in new:
    actual, trace = item["actual_dispatches"], item["trace"]
    assert len(actual) == len(trace["acoustic_reads"])
    total = 0
    windows = []
    for dispatch, read in zip(actual, trace["acoustic_reads"]):
        left, right = read["window_ms"]
        assert 300 <= right - left <= 6000
        assert abs(dispatch["samples"] / 16 - (right - left)) < .001
        total += right - left
        windows.append((left, right))
    union = []
    for left, right in sorted(windows):
        if union and left <= union[-1][1]:
            union[-1][1] = max(right, union[-1][1])
        else:
            union.append([left, right])
    covered = sum(right - left for left, right in union)
    assert abs(total - trace["audio_ms"]) < .001
    assert abs(covered - trace["covered_audio_ms"]) < .001
    assert covered <= .65 * item["duration_ms"] and total <= 2 * item["duration_ms"]
summary = {}
for model in sorted({i["model"] for i in new}):
    def stats(items):
        rows = [(i["surah"], a, c, e, a in i["trace"]["review_required_ayahs"])
                for i in items for (a, _, c), e in zip(i["rows"], i["errors_ms"])]
        return {"ayahs": len(rows), "confirmed": sum(c > 0 and not review for _, _, c, _, review in rows),
                "all_over500": sum(e > 500 for _, _, _, e, _ in rows),
                "confirmed_over500": [[s, a, e] for s, a, c, e, review in rows if c > 0 and not review and e > 500]}
    latest = [i for i in new if i["model"] == model]
    legacy = [i for i in old if i["model"] == model]
    paired = [i for i in latest if i["surah"] in {j["surah"] for j in legacy}]
    for i in paired:
        j = next(j for j in legacy if j["surah"] == i["surah"])
        assert all(i[k] == j[k] for k in ("audio_sha256", "initial_pauses_sha256", "reference_sha256"))
    summary[model] = {"historical_sparse4_full46": stats(latest),
                      "historical_sparse4_paired37": stats(paired), "historical_legacy37": stats(legacy),
                      "estimate_reasons": dict(Counter(b["review_reason"] for i in latest
                                                        for b in i["trace"]["boundaries"] if b["evidence_kind"] == "estimate"))}
output = {"kind": "historical evidence arithmetic, not fresh inference or replay",
          "source_artifacts_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (new_path, old_path)},
          "reference_sha256": ref_sha, "records_checked": len(new), "models": summary}
(ROOT / "historical-check.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False))
