"""Controlled FULL target/topology/context experiments; never a FAST executor."""
import argparse
import gc
import json
import resource
import threading
import time
import traceback

import numpy as np
import onnxruntime as ort

from common import SOURCE, TASK, corpus, digest, prediction_guard, write_json
from surah_cutter.core import align, decode, emissions, full_boundary, full_windows
from surah_cutter.data import quran_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", choices=["strong20", "fixed30", "strong6"], default="strong20")
    parser.add_argument("--variants", nargs="+", default=["current", "no_delimiter", "phones", "free_edges"])
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    parser.add_argument("--normalization", choices=["record", "call"], default="record")
    args = parser.parse_args()
    prediction_guard()
    context_label = args.context + ("-fp32" if args.precision == "fp32" else "")
    if args.normalization == "call":
        context_label += "-localnorm"
    output = TASK / (context_label + ("-fresh" if args.fresh else ""))
    output.mkdir(exist_ok=True)
    cache = TASK / "qpc-logits"
    cache.mkdir(exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    path = SOURCE / "models" / ("wav2vec2-quran-phonetics.fp32.onnx" if args.precision == "fp32"
                                else "wav2vec2-quran-phonetics.onnx")
    weight_sha = digest(path.read_bytes())
    vocab, normalize = emissions._load_sidecar(path, SOURCE / "models/wav2vec2-quran-phonetics.onnx.vocab.json")
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    emitter = emissions.OnnxEmitter(session, dict(vocab), blank_id=37,
                                    do_normalize=normalize and args.normalization == "record",
                                    weights_sha256=weight_sha)
    original_run = emissions._run
    original_softmax = emissions._log_softmax
    original_targets = align._build_targets
    original_full = align._full_emissions
    state = {"variant": "current", "record": "", "events": []}
    lock = threading.Lock()
    ledger = (output / "calls.jsonl").open("a")

    def run(actual_session, audio, input_name, output_name):
        if args.normalization == "call":
            audio = np.stack([emissions._normalize(row) for row in audio])
        key = digest((weight_sha + repr(audio.shape)).encode() + audio.tobytes())
        destination = cache / (key + ".npy")
        event = {"record": state["record"], "variant": state["variant"], "context": args.context,
                 "pcm_normalized_sha": digest(audio.tobytes()), "shape": list(audio.shape), "key": key}
        start = time.perf_counter()
        reused = destination.exists() and not args.fresh
        if reused:
            raw = np.load(destination, allow_pickle=False)
            model_s = json.loads(destination.with_suffix(".json").read_text())["seconds"]
        else:
            with lock:
                ledger.write(json.dumps({"state": "dispatch", **event}) + "\n")
                ledger.flush()
            raw = original_run(actual_session, audio, input_name, output_name)
            model_s = time.perf_counter() - start
            if not destination.exists():
                np.save(destination, raw, allow_pickle=False)
                write_json(destination.with_suffix(".json"), {"seconds": model_s, "key": key,
                                                               "sha256": digest(raw.tobytes())})
        event.update(reused=reused, model_seconds=model_s, wall_seconds=time.perf_counter() - start,
                     frames=int(raw.shape[1]), state="reused" if reused else "complete")
        with lock:
            state["events"].append(event)
            ledger.write(json.dumps(event) + "\n")
            ledger.flush()
        return raw

    def softmax(raw):
        if state["variant"] == "blank_minus1":
            raw = raw.copy()
            raw[..., 37] -= 1.0
        if state["variant"] in ("phones", "free_edges"):
            raw = raw.copy()
            raw[..., 37] = np.logaddexp.reduce(raw[..., [37, 0, 2, 35]], axis=-1)
            raw[..., [0, 2, 35]] = -np.inf
        result = original_softmax(raw)
        if state["variant"] == "free_edges":
            # Wildcard is an alignment score, not a trained probability class.
            result = np.concatenate((result, np.zeros((*result.shape[:-1], 1), dtype=result.dtype)), axis=-1)
        return result

    def targets(words, trs, bundle, basmala_trs=None):
        variant = state["variant"]
        if variant in ("phones", "free_edges"):
            trs = ["".join(c for c in tr if c not in " -|") for tr in trs]
            if basmala_trs is not None:
                basmala_trs = ["".join(c for c in tr if c not in " -|") for tr in basmala_trs]
        if variant == "free_edges":
            ids, spans = original_targets(words, trs, bundle, None)
            return [40, *ids, 40], [(lo + 1, hi + 1) for lo, hi in spans]
        return original_targets(words, trs, bundle, basmala_trs)

    def full(bundle, pcm, np_module, *, trace=None):
        if args.context == "strong20":
            return original_full(bundle, pcm, np_module, trace=trace)
        window_trace = {}
        if args.context == "fixed30":
            matrix = emitter.emissions(pcm, window_s=30, context_s=2)
            window_trace.update(algorithm="research-fixed30-context2", windows=[],
                                review_policy="research-no-pause-seam-proof")
        else:
            matrix = full_windows.generate_pause_emissions(emitter, pcm, window_s=6, workers=3,
                                                           strong_pauses=True, max_window_s=60,
                                                           trace=window_trace)
            window_trace.update(algorithm=full_windows.FULL_POLICY,
                                review_policy=full_windows.FULL_REVIEW_POLICY)
        if trace is not None:
            trace["full_windows"] = window_trace
        return matrix

    emissions._run = full_windows._run = run
    emissions._log_softmax = full_windows._log_softmax = softmax
    align._build_targets = targets
    align._full_emissions = full

    def no_other_model(*_args, **kwargs):
        trace = kwargs.get("trace")
        if trace is not None:
            trace["research_auxiliary_model_disabled"] = True

    full_boundary._recover_long_split = no_other_model
    selected = [item for item in corpus() if not args.ids or item["id"] in args.ids]
    selected.sort(key=lambda item: item["surah"] == 2)
    for case in selected:
        audio = case["audio"]
        pcm = decode.decode(audio)
        words = quran_text.surah_words(case["surah"])
        for variant in args.variants:
            destination = output / f"{case['id']}-{variant}.json"
            if destination.exists():
                continue
            state.update(record=case["id"], variant=variant, events=[])
            emitter.vocab = dict(vocab, **({"*": 40} if variant == "free_edges" else {}))
            bundle = align._OnnxBundle(emitter)
            if variant in ("no_delimiter", "phones", "free_edges"):
                bundle.delim_id = None
            trace = {}
            start = time.perf_counter()
            result = {"case": case, "context": context_label, "variant": variant,
                      "weight_sha": weight_sha, "ort": ort.__version__,
                      "PCM_sha": digest(pcm.tobytes()), "duration_ms": len(pcm) / 16,
                      "reference_prior": "disabled-text-only", "auxiliary_models": "disabled",
                      "fresh": args.fresh, "canonical_word_count": len(words)}
            result["normalization"] = args.normalization
            try:
                rows = align.align_surah(pcm, 16000, words, model=bundle, surah=case["surah"], trace=trace)
                assert [r[0] for r in rows] == list(range(1, quran_text.ayah_count(case["surah"]) + 1))
                assert all(b[1] > a[1] for a, b in zip(rows, rows[1:]))
                result["rows"] = rows
            except Exception:
                result["error"] = traceback.format_exc()
            result.update(trace=trace, wall_s=time.perf_counter() - start, calls=state["events"],
                          process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write_json(destination, result)
            print(args.context, case["id"], variant, "ERROR" if "error" in result else "OK",
                  round(result["wall_s"], 2), "calls", len(state["events"]), flush=True)
            gc.collect()
    ledger.close()


if __name__ == "__main__":
    main()
