"""Preservation audit of fresh matched model/surah runs, after inference."""
import json
from pathlib import Path

task = Path(__file__).parent.resolve()
controls = []
for name in ("control", "other_controls", "base_long_control"):
    controls.extend(json.loads((task / (name + ".json")).read_text()))
before = {(i["model"], i["surah"]): i for i in controls}
after = json.loads((task / "next_pause.json").read_text())
cases = []
for item in after:
    old = before[item["model"], item["surah"]]
    a = {ayah: (ms, c) for ayah, ms, c in old["rows"]}
    b = {ayah: (ms, c) for ayah, ms, c in item["rows"]}
    lost = [ayah for ayah in a if a[ayah][1] > 0 and b[ayah][1] == 0]
    changed = [ayah for ayah in a if a[ayah][1] > 0 and b[ayah][1] > 0 and a[ayah] != b[ayah]]
    added = [ayah for ayah in a if a[ayah][1] == 0 and b[ayah][1] > 0]
    outliers = [[ayah, error, confidence] for (ayah, _, confidence), error
                in zip(item["rows"], item.get("errors_ms", ())) if confidence > 0 and error > 500]
    probes = [e for e in item["trace"]["transitions"]
              if e.get("reason") == "empty_tempo_region_next_dsp_pause"]
    for event in probes:
        pause = item["trace"]["phase1_pauses"][event["candidate"]]
        left, right = event["lookahead_region_ms"]
        assert left < pause["start_ms"] <= right
    cases.append({"model": item["model"], "surah": item["surah"], "lost_proofs": lost,
                  "changed_proofs": changed, "added_proofs": added, "confident_outliers": outliers,
                  "proofs_before": sum(c > 0 for _, c in a.values()),
                  "proofs_after": sum(c > 0 for _, c in b.values()),
                  "over500_before": sum(e > 500 for e in old["errors_ms"]),
                  "over500_after": sum(e > 500 for e in item.get("errors_ms", ())),
                  "seconds_before": old["seconds"], "seconds_after": item["seconds"],
                  "calls_before": len(old["actual_dispatches"]), "calls_after": len(item["actual_dispatches"]),
                  "input_identity_equal": old["audio_sha256"] == item["audio_sha256"]
                      and old["initial_pauses_sha256"] == item["initial_pauses_sha256"],
                  "probes": probes})
summary = {}
for model in sorted({c["model"] for c in cases}):
    subset = [c for c in cases if c["model"] == model]
    summary[model] = {"cases": len(subset), "proofs_before": sum(c["proofs_before"] for c in subset),
                      "proofs_after": sum(c["proofs_after"] for c in subset),
                      "lost_proofs": sum(len(c["lost_proofs"]) for c in subset),
                      "changed_proofs": sum(len(c["changed_proofs"]) for c in subset),
                      "new_confident_outliers": sum(len(c["confident_outliers"]) for c in subset),
                      "over500_before": sum(c["over500_before"] for c in subset),
                      "over500_after": sum(c["over500_after"] for c in subset),
                      "extra_probes": sum(len(c["probes"]) for c in subset),
                      "seconds_before": sum(c["seconds_before"] for c in subset),
                      "seconds_after": sum(c["seconds_after"] for c in subset)}
report = {"cases": cases, "summary": summary, "scope": "fresh shared PCM/DSP; all local dispatches verified independently"}
(task / "next-pause-comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, indent=2))
