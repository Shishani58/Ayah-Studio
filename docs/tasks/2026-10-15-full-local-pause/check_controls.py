"""Read-only controls with frozen main emissions; reference used after prediction."""
import json
from collections import Counter
from pathlib import Path

from surah_cutter.core import align, decode, emissions
from surah_cutter.data import quran_text, references
import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
CACHE = Path.home() / ".cache/surah-cutter/full-accuracy"
bundle = align._OnnxBundle(emissions.load_emitter(ROOT / "models/wav2vec2-quran-phonetics.onnx"))
old = json.loads((TASK.parent / "2026-10-15-full-pause-center/audio-check.json").read_text())
results = []
for baseline in old:
    surah = baseline["surah"]
    stem = f"abu_bakr_ash_shatri-{surah}"
    matrix = np.load(CACHE / f"{stem}.npy")
    windows = json.loads((CACHE / f"{stem}-windows.json").read_text())

    def cached(bundle, pcm, np, *, trace=None):
        if trace is not None:
            trace["full_windows"] = dict(windows)
        return matrix

    align._full_emissions = cached
    pcm = decode.decode(str(Path.home() / f"Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3"))
    trace = {}
    rows = align.align_surah(pcm, 16000, quran_text.surah_words(surah), model=bundle, trace=trace)
    assert len(rows) == quran_text.ayah_count(surah)
    assert all(left[1] < right[1] for left, right in zip(rows, rows[1:]))
    reference = references.load_reference("abu_bakr_ash_shatri")
    old_rows = {r["ayah"]: r for r in baseline["comparison"]}
    comparison = [dict(ayah=a, before_ms=old_rows[a]["after_ms"], after_ms=ms,
                       reference_ms=reference[surah, a]) for a, ms, _ in rows]
    result = dict(surah=surah, count=len(rows),
                  reasons=dict(Counter(c["pause_center_reason"] for c in trace["ayahs"])),
                  comparison=comparison)
    for key in ("before_ms", "after_ms"):
        errors = np.array([abs(r[key] - r["reference_ms"]) for r in comparison])
        result[key] = dict(median_ms=float(np.median(errors)), mean_ms=float(np.mean(errors)),
                           max_ms=int(max(errors)), over500=int(sum(errors > 500)))
    results.append(result)
    print(json.dumps({k: v for k, v in result.items() if k != "comparison"}), flush=True)
    (TASK / "controls.json").write_text(json.dumps(results, indent=2) + "\n")
