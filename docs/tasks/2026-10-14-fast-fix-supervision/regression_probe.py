"""Independent baseline/current recovery comparison using synthetic acoustics."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import types

sys.path.insert(0, "tests")
helpers = runpy.run_path("tests/test_fast_algorithm_gaps.py")
run = helpers["run"]
current_executor = run.__globals__["align_surah_sequential"]
baseline = types.ModuleType("surah_cutter.core._review_baseline_executor")
sys.modules[baseline.__name__] = baseline
source = subprocess.check_output(["git", "show", "37f3309:surah_cutter/core/fast_executor.py"], text=True)
exec(compile(source, "baseline37f3309", "exec"), baseline.__dict__)
baseline.first_pause_hypotheses = lambda *args: []
results = []
for kind in ("weak", "wrong", "truncated", "contradictory_future"):
    fake = helpers["contextual_fake"]("correct" if kind == "contradictory_future" else kind)
    original = fake.responder
    if kind == "contradictory_future":
        fake.responder = lambda f, start, duration: (
            helpers["observation"](f.phrase(3), positions=False)
            if start == 6000 else original(f, start, duration)
        )
    else:
        fake.responder = lambda f, start, duration: (
            helpers["observation"](f.phrase(2), positions=False)
            if start == 6000 and duration > 2000 else original(f, start, duration)
        )
    pauses = [helpers["pause"](1000), helpers["pause"](6000)]
    item = {"tail": kind}
    for label, executor in (("baseline", baseline.align_surah_sequential), ("current", current_executor)):
        run.__globals__["align_surah_sequential"] = executor
        fake.calls.clear()
        rows, trace = run(fake, pauses)
        item[label] = {"rows": rows, "trace": trace}
    item["regression"] = item["baseline"]["rows"][1] != item["current"]["rows"][1]
    results.append(item)
run.__globals__["align_surah_sequential"] = current_executor
Path(__file__).with_name("regression-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps([{ "tail": i["tail"], "baseline_ayah2": i["baseline"]["rows"][1],
                   "current_ayah2": i["current"]["rows"][1], "regression": i["regression"]}
                  for i in results], ensure_ascii=False))
assert not any(i["regression"] for i in results)
