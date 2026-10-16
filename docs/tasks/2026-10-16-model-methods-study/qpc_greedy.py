"""Reconstruct FULL phonetic decode strictly from exact cached acoustic outputs."""
import json
from types import SimpleNamespace

import numpy as np

from common import SOURCE, TASK, corpus, digest, prediction_guard, write_json
from surah_cutter.core import decode, emissions, full_windows
from surah_cutter.data import transliteration


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


def main():
    prediction_guard()
    identity = json.loads((TASK / "strong20/shatri-001-current.json").read_text())
    weight_sha = identity["weight_sha"]
    vocab, normalize = emissions._load_sidecar(SOURCE / "models/wav2vec2-quran-phonetics.onnx", None)
    inverse = {v: k for k, v in vocab.items()}
    state = {"localnorm": False, "weight_sha": weight_sha}
    fp32_identity = TASK / "strong20-fp32/shatri-001-current.json"
    fp32_sha = json.loads(fp32_identity.read_text())["weight_sha"] if fp32_identity.exists() else None

    class Cache:
        def get_inputs(self):
            return [SimpleNamespace(name="input_values")]

        def get_outputs(self):
            return [SimpleNamespace(name="logits")]

        def run(self, outputs, inputs):
            audio = inputs["input_values"]
            if state["localnorm"]:
                audio = np.stack([emissions._normalize(row) for row in audio])
            key = digest((state["weight_sha"] + repr(audio.shape)).encode() + audio.tobytes())
            path = TASK / "qpc-logits" / (key + ".npy")
            if not path.exists():
                raise ValueError("Missing exact acoustic output: " + key)
            return [np.load(path, allow_pickle=False)]

    emitter = emissions.OnnxEmitter(Cache(), vocab, blank_id=37, do_normalize=normalize,
                                    weights_sha256=weight_sha)
    results = []
    for case in corpus():
        if case["surah"] not in (1, 80, 112):
            continue
        pcm = decode.decode(case["audio"])
        target = "".join(transliteration.surah_word_tr(case["surah"]))
        if case["surah"] not in (1, 9):
            target = "".join(transliteration.surah_word_tr(1)[:4]) + target
        target = "".join(c for c in target if c not in " -|")
        for method in ("strong20", "fixed30", "strong6", "strong20-fp32", "strong20-localnorm"):
            if not (TASK / method / f"{case['id']}-current.json").exists():
                continue
            state["localnorm"] = method.endswith("localnorm")
            state["weight_sha"] = fp32_sha if method == "strong20-fp32" else weight_sha
            emitter._do_normalize = normalize and not state["localnorm"]
            if method == "fixed30":
                matrix = emitter.emissions(pcm, window_s=30, context_s=2)
            else:
                matrix = full_windows.generate_pause_emissions(emitter, pcm, window_s=6 if method == "strong6" else 20,
                                                               workers=1, strong_pauses=True, max_window_s=60)
            assert not (set(target)-set(vocab)), "Expected representation contains unknown labels"
            for decoder in ("greedy", "blank_minus1"):
                scores = matrix.copy() if decoder == "blank_minus1" else matrix
                if decoder == "blank_minus1":
                    scores[:, 37] -= 1
                raw = scores.argmax(1)
                text = "".join(inverse[int(v)] for i, v in enumerate(raw)
                               if v != 37 and (i == 0 or v != raw[i - 1]))
                text = "".join(c for c in text if c not in " -|")
                errors = distance(target, text)
                results.append({"record": case["id"], "method": method, "decoder": decoder, "target": target,
                                "greedy": text, "edits": errors, "target_characters": len(target),
                                "phonetic_error_rate": errors / len(target), "frames": len(raw),
                                "blank_argmax_frames": int(np.sum(raw == 37)),
                                "delimiter_argmax_frames": int(np.sum(raw == 35)),
                                "delimiter_max_probability": float(np.exp(matrix[:, 35]).max())})
                print(case["id"], method, decoder, errors, len(target), flush=True)
    write_json(TASK / "phonetic-decode.json", results)


if __name__ == "__main__":
    main()
