"""Read-only check of the frozen runtime, independent of user projects."""

import json
import subprocess
from pathlib import Path


def main() -> int:
    import imageio_ffmpeg
    import numpy as np
    import onnxruntime as ort

    from surah_cutter.core.emissions import load_emitter
    from surah_cutter.data.quran_text import surah_ayah_counts

    root = Path(__file__).resolve().parents[1]
    counts = surah_ayah_counts()
    if len(counts) != 114 or sum(counts.values()) != 6236:
        raise RuntimeError("Quran database is incomplete")
    for relative in ("assets/fonts/qpc_hafs.ttf", "surah_cutter/webgui/static/index.html",
                     "surah_cutter/webgui/static/app.js", "surah_cutter/webgui/static/style.css",
                     "surah_cutter/webgui/static/mark.svg", "data/word_tr_cache.json"):
        if not (root / relative).is_file():
            raise FileNotFoundError(relative)
    emitter = load_emitter(root / "models/wav2vec2-quran-phonetics.onnx")
    session = emitter.session
    output = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 16000), np.float32)})
    if not output or not np.isfinite(output[0]).all():
        raise RuntimeError("Acoustic model inference failed")
    ort.InferenceSession(str(root / "models/silero_vad.onnx"), providers=["CPUExecutionProvider"])
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], check=True, capture_output=True)
    print(json.dumps({"app": "Ayah Studio", "suras": len(counts), "ayahs": sum(counts.values()),
                      "models": "ok", "ffmpeg": "ok", "assets": "ok"}))
    return 0
