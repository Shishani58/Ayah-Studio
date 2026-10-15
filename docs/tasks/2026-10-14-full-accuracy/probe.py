"""Isolate FP32 and context on selected recordings; cached baseline is immutable."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
RUNTIME = Path.home() / ".cache/surah-cutter/full-reciter-audit/runtime-20261014"
sys.path.insert(0, str(RUNTIME))
from surah_cutter.core import align, decode, emissions, full_windows  # noqa: E402

mode = sys.argv[1]
model = ROOT / "models" / ("wav2vec2-quran-phonetics.fp32.onnx" if mode == "fp32" else "wav2vec2-quran-phonetics.onnx")
import onnxruntime as ort  # noqa: E402
vocab, _ = emissions._load_sidecar(ROOT / "models/wav2vec2-quran-phonetics.onnx", None)
emitter = emissions.OnnxEmitter(ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]),
                              vocab, blank_id=emissions._resolve_blank_id(vocab))
from surah_cutter.data import quran_text  # noqa: E402
sources = json.loads((TASK / "diagnostics.json").read_text())
results = []
for s in sources:
    if (s["reciter"], s["surah"]) not in {( "abdul_basit_mujawwad",98), ("saad_al_ghamidi",80),
             ("abu_bakr_ash_shatri",42), ("abu_bakr_ash_shatri",83), ("abu_bakr_ash_shatri",95)}:
        continue
    pcm = decode.decode(s["audio"])
    for context in ([2] if mode == "fp32" else [1, 4]):
        windows = {}
        matrix = full_windows.generate_pause_emissions(emitter, pcm, window_s=20, workers=3,
                    context_s=context, strong_pauses=True, max_window_s=60, trace=windows)
        bundle = align._OnnxBundle(SimpleNamespace(vocab=vocab, blank_id=emitter.blank_id,
                                                 emissions=lambda _:matrix))
        trace = {}
        rows = align.align_surah(pcm,16000,quran_text.surah_words(s["surah"]),model=bundle,
                                surah=s["surah"],trace=trace)
        align._review_pause_full_onsets(trace["ayahs"], windows)
        # Evaluate against already recorded GT only after model prediction.
        errors = [r[1]-(old[1]-e) for r,old,e in zip(rows,s["rows"],s["errors_ms"])]
        results.append({"reciter":s["reciter"],"surah":s["surah"],"context_s":context,
                        "mode":mode,"model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
                        "rows":rows,"checks":trace["ayahs"],"errors_ms":errors,"windows":windows})
        (TASK/(mode+"-probe.json")).write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
        print(mode,s["reciter"],s["surah"],context,"max",max(map(abs,errors)),flush=True)
