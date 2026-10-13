"""Read-only Shatri local FAST audit; reference is loaded only for evaluation."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from surah_cutter.core import align, align_fast, decode, emissions  # noqa: E402
from surah_cutter.data import quran_text, references  # noqa: E402
from tools.bench_fast import pause_contract_violations  # noqa: E402

bundle = align._OnnxBundle(emissions.load_emitter(ROOT / "models/wav2vec2-quran-phonetics.onnx"))
items = []
output = Path(__file__).with_name("benchmark.json")
for surah in (list(map(int, sys.argv[1:])) or range(78, 115)):
    pcm = decode.decode(str(Path.home() / "Desktop/Abu_Bakr_Ash-Shatri" / f"{surah:03d}.mp3"))
    words = quran_text.surah_words(surah)
    trace = {}
    started = time.monotonic()
    rows = align_fast.align_surah_fast(pcm, 16000, words, surah=surah, bundle=bundle, trace=trace)
    elapsed = time.monotonic() - started
    ref = references.load_reference("abu_bakr_ash_shatri")
    errors = [ms - ref[(surah, a)] for a, ms, _ in rows]
    item = {"surah": surah, "seconds": elapsed, "rows": rows, "errors_ms": errors,
            "max_ms": max(map(abs, errors)), "p95_ms": float(np.percentile(np.abs(errors), 95)),
            "pause_violations": pause_contract_violations(rows, trace), "trace": trace}
    items.append(item)
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    print(surah, round(elapsed, 2), item["max_ms"], round(trace["coverage"], 3),
          trace["emission_passes"], len(trace.get("review_required_ayahs", [])), flush=True)

errors = np.array([abs(e) for item in items for e in item["errors_ms"]])
summary = {
    "suras": len(items), "ayahs": len(errors), "median_ms": float(np.median(errors)),
    "p95_ms": float(np.percentile(errors, 95)), "max_ms": int(errors.max()),
    "over500": int((errors > 500).sum()),
    "review_ayahs": sum(len(i["trace"].get("review_required_ayahs", [])) for i in items),
    "seconds": sum(i["seconds"] for i in items),
    "audio_ms": sum(i["trace"]["audio_ms"] for i in items),
    "covered_audio_ms": sum(i["trace"]["covered_audio_ms"] for i in items),
    "total_ms": sum(i["trace"]["total_ms"] for i in items),
    "emission_passes": sum(i["trace"]["emission_passes"] for i in items),
    "pause_violations": sum(len(i["pause_violations"]) for i in items),
    "positive_conf_over500": [
        (i["surah"], a, e, c) for i in items
        for (a, ms, c), e in zip(i["rows"], i["errors_ms"]) if abs(e) > 500 and c > 0
    ],
}
model = ROOT / "models/wav2vec2-quran-phonetics.onnx"
metadata = {
    "settings": {
        "sr": 16000, "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "provider": "CPUExecutionProvider", "normalize": True, "batch_size": 1,
        "max_read_ms": 6000, "max_coverage": .65, "max_audio_ratio": 2,
        "initial_pause_speech_ms": 1300, "lag_ms": 200,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("SC_FAST_")},
    },
    "source_sha256": {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in ["surah_cutter/core/align_fast.py", "surah_cutter/core/fast_reads.py",
                         "surah_cutter/core/emissions.py"]
    },
    "audio_sha256": {
        str(i["surah"]): hashlib.sha256(
            (Path.home() / "Desktop/Abu_Bakr_Ash-Shatri" / f"{i['surah']:03d}.mp3").read_bytes()
        ).hexdigest() for i in items
    },
    "summary": summary,
}
output.with_name("summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
assert all(i["trace"]["coverage"] <= .65 for i in items)
assert all(0 < r["window_ms"][1] - r["window_ms"][0] <= 6000
           for i in items for r in i["trace"]["acoustic_reads"])
assert all(i["trace"]["emission_passes"] == len(i["trace"]["acoustic_reads"]) for i in items)
assert summary["pause_violations"] == 0
