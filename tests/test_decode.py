import threading
import wave

import numpy as np
import pytest

from surah_cutter.core import decode
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope


def read(path, ranged):
    if ranged:
        return decode.decode_range(str(path), start_ms=100, end_ms=200)
    return decode.decode(str(path))


@pytest.mark.parametrize("ranged", [False, True])
def test_ffmpeg_decodes_exact_writable_pcm(tmp_path, ranged):
    samples = np.arange(4800, dtype=np.int16)
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())

    pcm = read(path, ranged)
    expected = samples[1600:3200] if ranged else samples
    np.testing.assert_array_equal(pcm, expected.astype(np.float32) / 32768)
    assert pcm.flags.writeable


@pytest.mark.parametrize("ranged", [False, True])
def test_ffmpeg_error_retains_source_path(tmp_path, ranged):
    path = tmp_path / "missing.wav"
    with pytest.raises(decode.DecodeError, match="missing.wav"):
        read(path, ranged)


@pytest.mark.parametrize("ranged", [False, True])
def test_ffmpeg_cancellation_is_not_a_decode_error(tmp_path, ranged):
    event = threading.Event()
    event.set()
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        read(tmp_path / "missing.wav", ranged)
