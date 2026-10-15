"""Read-only QPC FULL check on local audio; reuse frozen main-pass emissions."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from surah_cutter.core import align, decode, emissions
from surah_cutter.data import quran_text, references

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path.home() / ".cache/surah-cutter/full-accuracy"
bundle = align._OnnxBundle(emissions.load_emitter(ROOT / "models/wav2vec2-quran-phonetics.onnx"))
results = []
for surah in [1, 83, 95, 112, 114, 42]:
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
    moves = []
    for c in trace["ayahs"]:
        if c.get("pause_center_reason") != "centered":
            assert c["onset_ms"] == c["pre_center_onset_ms"]
            continue
        assert c["onset_ms"] == round((c["center_pause_start_ms"] + c["center_pause_end_ms"]) / 2)
        moves.append({k: c[k] for k in ("ayah", "pre_center_onset_ms", "onset_ms",
                                       "center_pause_start_ms", "center_pause_end_ms")})
    result = dict(surah=surah, seconds=len(pcm) / 16000, mode=trace["mode"],
                  count=len(rows), reasons=dict(Counter(c["pause_center_reason"] for c in trace["ayahs"])),
                  moves=moves)
    # Load the reference only after predictions; never feed it into alignment.
    reference = references.load_reference("abu_bakr_ash_shatri")
    result["comparison"] = [dict(ayah=c["ayah"], reference_ms=reference[surah, c["ayah"]],
                                 before_ms=c["pre_center_onset_ms"], after_ms=c["onset_ms"])
                            for c in trace["ayahs"]]
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    Path(__file__).with_name("audio-check.json").write_text(json.dumps(results, indent=2) + "\n")
