"""Same PCM: original pinned Torch weights versus FP32/INT8 ONNX export."""
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

from common import SOURCE, TASK, digest, write_json


def main():
    torch.set_num_threads(2)
    folder = (Path.home() / ".cache/huggingface/hub/models--TBOGamer22--wav2vec2-quran-phonetics"
              / "snapshots/65c3ab93446f2f1041a18082433a911e5e154a9d")
    model = Wav2Vec2ForCTC.from_pretrained(folder, local_files_only=True, attn_implementation="eager").eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(folder, local_files_only=True)
    inputs = []
    for surah in (1, 80):
        path = TASK.parent / "2026-10-16-model-capability-audit" / f"pcm-{surah:03d}.npy"
        inputs.append((str(surah), np.load(path, allow_pickle=False)))
    inputs.append(("zero", np.zeros(96000, dtype=np.float32)))
    original = {}
    for name, pcm in inputs:
        values = extractor(pcm, sampling_rate=16000, return_tensors="pt").input_values
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(values).logits.numpy()
        original[name] = (values.numpy(), logits)
        print("torch", name, logits.shape, round(time.perf_counter() - start, 2), flush=True)
    result = {"original_weights_sha": digest((folder / "model.safetensors").read_bytes()),
              "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "onnxruntime"]},
              "cases": []}
    for filename in ["wav2vec2-quran-phonetics.fp32.onnx", "wav2vec2-quran-phonetics.onnx"]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        path = SOURCE / "models" / filename
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        for name, (values, logits) in original.items():
            actual = session.run(None, {session.get_inputs()[0].name: values})[0]
            diff = np.abs(actual - logits)
            result["cases"].append({"model": filename, "case": name, "weight_sha": digest(path.read_bytes()),
                                    "input_sha": digest(values.tobytes()), "shape": list(actual.shape),
                                    "max_abs_logit_difference": float(diff.max()),
                                    "mean_abs_logit_difference": float(diff.mean()),
                                    "argmax_agreement": float(np.mean(actual.argmax(-1) == logits.argmax(-1))),
                                    "torch_argmax": logits.argmax(-1).tolist(), "onnx_argmax": actual.argmax(-1).tolist()})
        del session
    write_json(TASK / "parity.json", result)
    print(json.dumps([{k: v for k, v in c.items() if not k.endswith("argmax")}
                      for c in result["cases"]], indent=2))


if __name__ == "__main__":
    main()
