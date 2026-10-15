"""Compare completed model pairs only; do not read unscored checkpoints."""
import json
from pathlib import Path

task = Path(__file__).parent.resolve()
result = []
for model in ["quran-phonetics","quran-phonetics-fp32","whisper-tarteel-tiny","whisper-tarteel-base"]:
    if not (task / f"{model}-after.summary.json").exists():
        continue
    before = {c["surah"]:c for c in json.loads((task / f"{model}-before.json").read_text())}
    after = json.loads((task / f"{model}-after.json").read_text())
    assert len(after)==12 and set(c["surah"] for c in after)==set(before)
    summary = {"model":model,"lost":[],"added":[],"changed_prior":[],"new_outliers":[],"cases":[],"before_proofs":0,"after_proofs":0}
    for b in after:
        a = before[b["surah"]]
        for k in ["audio_sha256","reference_sha256","initial_pauses_sha256"]:
            assert a[k]==b[k],(model,b["surah"],k)
        ar = {r[0]:r for r in a["rows"] if r[2]>0}
        br = {r[0]:r for r in b["rows"] if r[2]>0}
        previous_outliers = {r[0] for r,e in zip(a["rows"],a["errors_ms"]) if r[2]>0 and e>500}
        summary["before_proofs"]+=len(ar)
        summary["after_proofs"]+=len(br)
        summary["lost"].extend([[b["surah"],ar[k]] for k in sorted(ar.keys()-br.keys())])
        summary["added"].extend([[b["surah"],br[k]] for k in sorted(br.keys()-ar.keys())])
        summary["changed_prior"].extend([[b["surah"],ar[k],br[k]] for k in sorted(ar.keys()&br.keys()) if ar[k]!=br[k]])
        summary["new_outliers"].extend([[b["surah"],r,e] for r,e in zip(b["rows"],b["errors_ms"]) if r[2]>0 and e>500 and r[0] not in previous_outliers])
        max_read = max((d["samples"]/16 for d in b["actual_dispatches"]),default=0)
        ratio = b["trace"]["audio_ms"]/b["duration_ms"]
        assert max_read<=6000.001 and b["trace"]["coverage"]<=.650001 and ratio<=2.000001
        assert not b["sequence_violations"] and len(b["actual_dispatches"])==b["trace"]["model_passes"]
        summary["cases"].append({"surah":b["surah"],"before":len(ar),"after":len(br),"rows_equal":a["rows"]==b["rows"],"dispatches_equal":a["actual_dispatches"]==b["actual_dispatches"],"max_read_ms":max_read,"coverage":b["trace"]["coverage"],"input_ratio":ratio})
    result.append(summary)
(task / "comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print([{k:v for k,v in r.items() if k not in ["cases","added"]} for r in result])
