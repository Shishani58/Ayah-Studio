"""Exact before/after corpus comparison; no models or tests."""
import hashlib
import json
from pathlib import Path

task = Path(__file__).parent.resolve()
old = {c["surah"]:c for c in json.loads((task.parent / "2026-10-14-fast-late-future-anchor/late-corpus.json").read_text())}
new = json.loads((task / "integrated-corpus.json").read_text())
assert len(new)==12 and set(c["surah"] for c in new)==set(old)
source = Path((task / "snapshot-path.txt").read_text().strip())
manifest = json.loads((task / "source-manifest.json").read_text())
assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
out = []
for b in new:
    a = old[b["surah"]]
    for k in ["rows","actual_dispatches","audio_sha256","reference_sha256","initial_pauses_sha256","errors_ms"]:
        assert a[k]==b[k],(b["surah"],k)
    for k in ["boundaries","pause_geometry","transitions"]:
        assert a["trace"][k]==b["trace"][k],(b["surah"],k)
    trace = b["trace"]
    max_read = max(d["samples"] / 16 for d in b["actual_dispatches"])
    ratio = trace["audio_ms"] / b["duration_ms"]
    assert max_read<=6000.001 and trace["coverage"]<=.650001 and ratio<=2.000001
    assert not b["sequence_violations"]
    out.append({"surah":b["surah"],"rows":len(b["rows"]),"proofs":sum(r[2]>0 for r in b["rows"]),
                "calls":len(b["actual_dispatches"]),"before_search_seconds":a["search_seconds"],
                "after_search_seconds":b["search_seconds"],"before_dsp_seconds":a["dsp_seconds"],
                "after_dsp_seconds":b["dsp_seconds"],"before_inference_seconds":a["trace"]["inference_seconds"],
                "after_inference_seconds":trace["inference_seconds"],"max_read_ms":max_read,
                "coverage":trace["coverage"],"input_ratio":ratio})
result = {"all_rows_dispatches_proofs_exact":True,"source_guard_files":len(manifest),"cases":out,
          "totals":{k:sum(c[k] for c in out) for k in ["rows","proofs","calls","before_search_seconds",
                                                        "after_search_seconds","before_dsp_seconds","after_dsp_seconds",
                                                        "before_inference_seconds","after_inference_seconds"]}}
(task / "comparison.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result["totals"],indent=2))
