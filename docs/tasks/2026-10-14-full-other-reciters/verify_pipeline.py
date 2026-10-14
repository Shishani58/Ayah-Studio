"""Read-only actual FULL entrypoint check against the factorial experiment."""

import json
from pathlib import Path
import sys
import time


TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
runtime = Path(json.loads((TASK / "runtime-manifest.json").read_text())["runtime"])
sys.path.insert(0, str(runtime))

import numpy as np  # noqa: E402
from surah_cutter.core import align, emissions, model_store, pipeline  # noqa: E402

model_store.ACTIVE_ID = "quran-phonetics"
model = ROOT / "models/wav2vec2-quran-phonetics.onnx"
bundle = align._OnnxBundle(emissions.load_emitter(model))
align._get_onnx_bundle = lambda: bundle
bundle.emitter.emissions(np.zeros(16_000, np.float32))
manifest = json.loads((TASK / "data-manifest.json").read_text())
results = []
for rec in manifest["recordings"]:
    data = json.loads((TASK / (rec["name"] + ".json")).read_text())
    expected = next(s for c in data["cases"] if c["case"] == "strong:20:3:0"
                    for s in c["surahs"] if s["surah"] == 98)
    trace = {}
    started = time.perf_counter()
    rows = pipeline.process_surah(str(Path(rec["audio_dir"]) / "098.mp3"), 98,
                                  mode="full", trace=trace)
    elapsed = time.perf_counter() - started
    checks = {c["ayah"]: c for c in trace["ayahs"]}
    actual = [[a, ms, checks[a]["conf"]] for _, a, ms in rows]
    review = [c["ayah"] for c in trace["ayahs"] if not c["reliable"]]
    assert actual == expected["rows"]
    assert review == expected["review"]
    assert trace["full_windows"]["workers"] == 3
    assert trace["full_windows"]["window_s"] == 20
    results.append({"reciter": rec["name"], "surah": 98, "pipeline_s": elapsed,
                    "rows": rows, "trace": trace, "matches_experiment": True})
    (TASK / "actual-pipeline.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
    )
    print(rec["name"], "actual pipeline matches", round(elapsed, 2), flush=True)
