"""Inspect existing QPC outputs on bounded PCM; no expected text or DB access."""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from surah_cutter.core.decode import decode_range  # noqa: E402
from surah_cutter.core.emissions import _normalize, load_emitter  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    sources = [
        "surah_cutter/core/emissions.py", "surah_cutter/core/align.py",
        "surah_cutter/core/full_windows.py", "surah_cutter/core/fast_adapters.py",
        "surah_cutter/core/model_worker.py", "surah_cutter/core/model_catalog.json",
        "surah_cutter/tools/export_onnx.py",
    ]
    before = {p: sha((ROOT / p).read_bytes()) for p in sources}
    inputs = []
    # Fixed clock intervals, selected before any decode; no reference/owners.
    for surah in (1, 80):
        path = Path.home() / f"Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3"
        pcm = decode_range(str(path), start_ms=1500, end_ms=7500)
        assert len(pcm) == 96000
        np.save(OUT / f"pcm-{surah:03d}.npy", pcm, allow_pickle=False)
        inputs.append((str(path), pcm, {
            "start_ms": 1500, "end_ms": 7500, "audio_sha256": sha(path.read_bytes()),
            "pcm_sha256": sha(pcm.tobytes()),
            "normalized_sha256": sha(_normalize(pcm).tobytes()),
        }))
    inputs.append(("synthetic-zero", np.zeros(96000, dtype=np.float32), {}))
    results = []
    ledger_path = OUT / "dispatch.jsonl"
    if ledger_path.exists():
        ledger_path.rename(OUT / f"dispatch-previous-{time.time_ns()}.jsonl")
    with ledger_path.open("w") as ledger:
        for filename in ("wav2vec2-quran-phonetics.onnx", "wav2vec2-quran-phonetics.fp32.onnx"):
            emitter = load_emitter(ROOT / "models" / filename,
                                   vocab_path=ROOT / "models/wav2vec2-quran-phonetics.onnx.vocab.json")
            inv = {v: k for k, v in emitter.vocab.items()}
            for source, pcm, manifest in inputs:
                event = {"model": filename, "source": source, "samples": len(pcm), **manifest}
                ledger.write(json.dumps({"state": "dispatch", **event}) + "\n")
                ledger.flush()
                start = time.perf_counter()
                matrix = emitter.emissions(pcm)
                raw = matrix.argmax(1)
                collapsed = [int(v) for i, v in enumerate(raw)
                             if v != emitter.blank_id and (i == 0 or v != raw[i - 1])]
                special = {}
                for token in ("[PAD]", " ", "-", "|", "[UNK]"):
                    idx = emitter.vocab[token]
                    prob = np.exp(matrix[:, idx])
                    special[token] = {"max_probability": float(prob.max()),
                                      "mean_probability": float(prob.mean()),
                                      "argmax_frames": int(np.sum(raw == idx))}
                event.update(seconds=time.perf_counter() - start, frames=len(matrix),
                             vocab_columns=matrix.shape[1], blank_id=emitter.blank_id,
                             greedy="".join(inv[v] for v in collapsed), special=special,
                             runtime=emitter.runtime_identity)
                results.append(event)
                ledger.write(json.dumps({"state": "complete", **event}, ensure_ascii=False) + "\n")
                ledger.flush()
                print(filename, source, event["greedy"], special["|"], flush=True)
            del emitter
            gc.collect()
    after = {p: sha((ROOT / p).read_bytes()) for p in sources}
    report = {"purpose": "output contract, not alignment accuracy or FAST executor",
              "inference_calls": len(results), "real_recording_reads": 4, "zero_controls": 2,
              "source_sha_before": before, "source_sha_after": after,
              "production_unchanged": before == after, "results": results}
    (OUT / "qpc-output.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert before == after


if __name__ == "__main__":
    main()
