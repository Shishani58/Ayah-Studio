from io import BytesIO
import wave

import numpy as np
import pytest

from surah_cutter.core import decode
from surah_cutter.webgui.playback import audio_clip


def test_clip_is_a_finite_sample_range_without_a_following_sound(monkeypatch):
    pcm = np.zeros(16001, dtype=np.float32)
    pcm[-1] = 1  # Decoder overrun / next sound must never enter the WAV.
    monkeypatch.setattr(decode, "decode_range", lambda *a, **kw: pcm)
    with wave.open(BytesIO(audio_clip("file.mp3", 1000, 2000))) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.getnframes() == 16000
        assert not np.frombuffer(wav.readframes(20000), dtype="<i2").any()


@pytest.mark.parametrize("start,end", [(-1, 100), (10, 10), (100, 10)])
def test_bad_range_is_rejected_before_decoding(monkeypatch, start, end):
    monkeypatch.setattr(decode, "decode_range", lambda *a, **kw: pytest.fail("decoded bad range"))
    with pytest.raises(ValueError):
        audio_clip("file.mp3", start, end)


def test_range_past_eof_is_rejected(monkeypatch):
    monkeypatch.setattr(decode, "decode_range", lambda *a, **kw: np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="Empty"):
        audio_clip("file.mp3", 1000, 2000)
