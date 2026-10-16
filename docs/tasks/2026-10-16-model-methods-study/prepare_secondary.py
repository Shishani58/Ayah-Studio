"""Prepare FULL audio inputs, keeping evaluation truth in a separate file."""
import numpy as np
import soundfile as sf

from common import TASK, corpus, digest, prediction_guard, write_json
from surah_cutter.core import decode, model_runtime
from surah_cutter.data import quran_text


def main():
    prediction_guard()
    selected = [c for c in corpus() if c["surah"] in (98, 112)
                or c["id"] in ("shatri-001", "shatri-114")]
    out = TASK / "secondary-inputs"
    out.mkdir(exist_ok=True)
    inputs, truth = [], []
    intro_words = [w for w in quran_text.surah_words(1) if w.ayah == 1]
    for case in selected:
        pcm = decode.decode(case["audio"])
        np.save(out / f"{case['id']}.npy", pcm, allow_pickle=False)
        sf.write(out / f"{case['id']}.wav", pcm, 16000, subtype="FLOAT")
        inputs.append({**case, "duration_ms": len(pcm) / 16, "pcm_sha": digest(pcm.tobytes()),
                       "input": str(out / f"{case['id']}.npy")})
        words = quran_text.surah_words(case["surah"])
        renderings = model_runtime.inference_words(words, "fastconformer")
        intro = model_runtime.inference_words(intro_words, "fastconformer") if case["surah"] != 1 else []
        truth.append({"id": case["id"], "surah": case["surah"], "reciter": case["reciter"],
                      "words": renderings, "owners": [w.ayah for w in words], "intro": intro,
                      "canonical_words": [{"index": w.index, "ayah": w.ayah, "text": w.text} for w in words]})
    for name, pcm in [("zero4", np.zeros(64000, dtype=np.float32))]:
        np.save(out / f"{name}.npy", pcm, allow_pickle=False)
        sf.write(out / f"{name}.wav", pcm, 16000, subtype="FLOAT")
        inputs.append({"id": name, "surah": None, "reciter": None, "duration_ms": len(pcm) / 16,
                       "input": str(out / f"{name}.npy"), "pcm_sha": digest(pcm.tobytes())})
    write_json(TASK / "secondary-inputs.json", inputs)
    write_json(TASK / "secondary-truth.json", truth)
    print("prepared", len(inputs), "FULL inputs plus separate canonical truth")


if __name__ == "__main__":
    main()
