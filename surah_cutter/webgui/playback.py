"""Finite PCM previews use the same decoder and timebase as alignment."""
from io import BytesIO
import wave

import numpy as np

from surah_cutter.core import decode


def audio_clip(path, start_ms: int, end_ms: int) -> bytes:
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("Invalid audio range")
    pcm = decode.decode_range(str(path), start_ms=start_ms, end_ms=end_ms)
    pcm = pcm[:(end_ms - start_ms) * decode.DEFAULT_SR // 1000]
    if not len(pcm):
        raise ValueError("Empty audio range")
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(decode.DEFAULT_SR)
        wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return output.getvalue()
