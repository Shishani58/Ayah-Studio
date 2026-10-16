"""Read-only exact-range check on all seven saved Fatiha ayahs."""
from io import BytesIO
import json
from pathlib import Path
import time
import wave

import numpy as np

from surah_cutter.core import decode
from surah_cutter.webgui.playback import audio_clip

task = Path(__file__).resolve().parent
audio = Path.home() / "Desktop/Abu_Bakr_Ash-Shatri/001.mp3"
full = decode.decode(audio)
rows = json.loads((task / "user-full.json").read_text())["trace"]["ayahs"]
result = []
for i, row in enumerate(rows):
    start = row["onset_ms"]
    end = rows[i + 1]["onset_ms"] if i + 1 < len(rows) else round(len(full) / 16)
    begin = time.perf_counter()
    data = audio_clip(audio, start, end)
    elapsed = time.perf_counter() - begin
    with wave.open(BytesIO(data)) as wav:
        actual = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    expected = (np.clip(full[start * 16:end * 16], -1, 1) * 32767).astype("<i2")
    assert len(actual) == len(expected)
    # FFmpeg's resampler may differ at a seek's leading filter edge.
    a, b = actual[1600:].astype(float), expected[1600:].astype(float)
    error = np.sqrt(np.mean((a - b) ** 2))
    scale = np.sqrt(np.mean(b ** 2))
    correlation = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert correlation > .998  # Fractional resampling phase can differ at a seek.
    result.append(dict(ayah=row["ayah"], start_ms=start, end_ms=end,
                       n_samples=len(actual), correlation=correlation,
                       relative_rms_error=float(error / scale), elapsed_s=elapsed))
(task / "finite-audio.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
