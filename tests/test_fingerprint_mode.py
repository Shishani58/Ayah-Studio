from pathlib import Path
from types import SimpleNamespace

import numpy as np

from surah_cutter.core import decode, fingerprint


def test_fingerprint_mode_matches_same_recording_clips(tmp_path, monkeypatch):
    hop = decode.DEFAULT_SR // 20
    pcm = np.repeat(np.random.default_rng(42).uniform(.02, 1, 80).astype(np.float32), hop)
    clips = {"112001.wav": pcm[10 * hop:22 * hop],
             "112002.wav": pcm[50 * hop:62 * hop]}
    for name in clips:
        (tmp_path / name).touch()
    monkeypatch.setattr(decode, "decode", lambda path: pcm if path == "full.wav"
                        else clips[Path(path).name])
    words = [SimpleNamespace(ayah=1), SimpleNamespace(ayah=2)]

    assert fingerprint.align_surah_by_fingerprint(
        "full.wav", 112, words, ayah_audio_dir=tmp_path,
    ) == [(112, 1, 500), (112, 2, 2500)]
