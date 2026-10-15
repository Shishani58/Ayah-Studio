"""Independent local ASR without Quran prompts, no application state writes."""

import json, sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK.parents[2]))
from surah_cutter.core import align, decode, model_store
from surah_cutter.core.model_runtime import release_runtimes

model_store.ACTIVE_ID = "whisper-tarteel-base"
bundle = align._get_onnx_bundle()
results = []
run = json.loads(
    Path(
        "/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/Abu_Bakr_Ash-Shatri.logs/sura_002_full.json"
    ).read_text()
)
total_samples = round(run["trace"]["total_ms"] * 16)
for lo, hi in [(4837000, 4843000), (4843000, 4846000), (4845900, 4848900)]:
    pcm = decode.decode_range(
        "/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3", start_ms=lo, end_ms=hi
    )
    pcm = pcm[: (hi - lo) * 16]
    r = bundle.emitter.recognize_local(pcm, total_samples=total_samples)
    results.append({"crop_ms": [lo, hi], "result": r})
    print(lo, hi, r, flush=True)
    (TASK / "asr.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
release_runtimes()
