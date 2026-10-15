"""Isolate INT8 vs FP32 and local vs global DSP normalization on one crop."""
import hashlib
import json
from pathlib import Path

import numpy as np

from surah_cutter.core import align, decode, emissions, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_ctc import prefix_scores
from surah_cutter.core.fast_reads import LocalEvidenceReads
from tools.bench_fast_models import dispatch_spy, verify_dispatch

ROOT = Path(__file__).parent
audio = Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/001.mp3")
pcm = decode.decode(str(audio))
# Whole-signal arithmetic is DSP only. Models get exclusively one local crop.
normalized = emissions._normalize(pcm)
items = []
for model in ("quran-phonetics", "quran-phonetics-fp32"):
    model_store.ACTIVE_ID = model
    bundle = align.default_bundle()
    for norm in ("local", "global-dsp"):
        if norm == "local":
            selected, selected_pcm = bundle, pcm
        else:
            emitter = emissions.OnnxEmitter(bundle.emitter.session, bundle.vocab,
                blank_id=bundle.blank_id, do_normalize=False,
                weights_sha256=bundle.emitter.runtime_identity["weights_sha256"])
            selected, selected_pcm = align._OnnxBundle(emitter), normalized
        adapter = CTCAdapter(selected)
        reads = LocalEvidenceReads(selected_pcm, 16000, adapter)
        with dispatch_spy(selected, selected_pcm) as actual:
            obs = reads.read(6420, 7920, reason=f"runtime-ab:{model}:{norm}")
        trace = {"total_ms": len(pcm) / 16, **reads.diagnostics()}
        verify_dispatch(actual, selected_pcm, trace)
        phrase = tuple("alḥamdu")
        scores = prefix_scores(obs.posterior, [[adapter.vocab[u] for u in phrase],
                                               [adapter.vocab[u] for u in phrase[1:]]], adapter.blank_id)
        l_index = obs.units.index("l")
        frame = round((obs.positions_ms[l_index] + adapter.lag_ms) / obs.stride_ms)
        before = obs.posterior[:min(frame + 1, len(obs.posterior))]
        items.append({"model": model, "normalization": norm, "greedy": obs.text,
                      "canonical_prefix_probability": float(np.exp(scores[0])),
                      "max_initial_a_probability": float(np.exp(before[:, adapter.vocab["a"]].max())),
                      "missing_a_prefix_probability": float(np.exp(scores[1])),
                      "runtime_identity": adapter.runtime_identity, "trace": trace,
                      "actual_dispatches": actual})
    align._ONNX_CACHE.clear()
output = {"audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
          "global_normalization_pcm_sha256": hashlib.sha256(memoryview(normalized)).hexdigest(),
          "global_dsp_mean": float(pcm.mean()), "global_dsp_var": float(pcm.var()),
          "results": items}
(ROOT / "runtime-ab.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps([{k: i[k] for k in ("model", "normalization", "greedy",
                   "canonical_prefix_probability", "max_initial_a_probability",
                   "missing_a_prefix_probability")} for i in items], ensure_ascii=False))
