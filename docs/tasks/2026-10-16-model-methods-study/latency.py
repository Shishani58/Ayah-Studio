"""Serial alternating INT8/FP32 encoder benchmark, separate from cached replay."""
import json
import os
import platform
import time

import numpy as np
import onnxruntime as ort

from common import SOURCE, TASK, digest, prediction_guard, write_json
from surah_cutter.core import decode, emissions


def main():
    prediction_guard()
    baseline = json.loads((TASK / "strong20/shatri-080-current.json").read_text())
    pcm = decode.decode(baseline["case"]["audio"])
    assert digest(pcm.tobytes()) == baseline["PCM_sha"]
    window = baseline["trace"]["full_windows"]["windows"][0]
    start, end = round(window["read_start_ms"]*16), round(window["read_end_ms"]*16)
    left = round((window["read_start_ms"]-window["start_ms"]+2000)*16)
    right = window["input_samples"]-left-(end-start)
    long_input = np.pad(emissions._normalize(pcm)[start:end], (left,right))[None]
    assert any(c["pcm_normalized_sha"] == digest(long_input.tobytes()) for c in baseline["calls"])
    inputs = {"real6s": emissions._normalize(pcm[24000:120000])[None], "full_window": long_input}
    sessions, loads, identities = {}, {}, {}
    for precision, name in [("int8","wav2vec2-quran-phonetics.onnx"),
                            ("fp32","wav2vec2-quran-phonetics.fp32.onnx")]:
        path = SOURCE / "models" / name
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        start = time.perf_counter()
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        loads[precision] = time.perf_counter()-start
        sessions[precision] = session
        identities[precision] = digest(path.read_bytes())
    results, logits = [], {}
    for name, batch in inputs.items():
        for session in sessions.values():
            session.run(None,{session.get_inputs()[0].name:batch})
        for repeat in range(3):
            order = ("int8","fp32") if repeat%2 == 0 else ("fp32","int8")
            for precision in order:
                session = sessions[precision]
                load = os.getloadavg()
                start = time.perf_counter()
                output = session.run(None,{session.get_inputs()[0].name:batch})[0]
                seconds = time.perf_counter()-start
                results.append(dict(input=name,precision=precision,repeat=repeat,seconds=seconds,
                                    input_sha=digest(batch.tobytes()), samples=batch.shape[1],loadavg=load))
                if repeat == 0:
                    logits[name,precision] = output
    comparisons = {}
    for name in inputs:
        a,b = logits[name,"int8"],logits[name,"fp32"]
        comparisons[name] = dict(argmax_agreement=float(np.mean(a.argmax(-1)==b.argmax(-1))),
                                 max_abs_logit_difference=float(np.abs(a-b).max()))
    write_json(TASK / "latency.json",dict(ort=ort.__version__,hardware=platform.machine(),cpus=os.cpu_count(),
               intra_op_threads=2,weight_sha=identities,load_s=loads,results=results,comparisons=comparisons,
               caveat="Encoder only; one warmup per input/model, 3 alternating measurements. External host load is uncontrolled."))
    for name in inputs:
        for precision in sessions:
            times = [r["seconds"] for r in results if r["input"]==name and r["precision"]==precision]
            print(name,precision,"median",float(np.median(times)),"range",min(times),max(times))


if __name__ == "__main__":
    main()
