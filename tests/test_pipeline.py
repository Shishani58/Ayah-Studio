from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from surah_cutter.core import align_fast, decode, pipeline


def test_fast_align_keeps_fast_rows_on_drift(monkeypatch):
    logs: list[str] = []
    monkeypatch.setattr(decode, "DEFAULT_SR", 1000)
    monkeypatch.setattr(decode, "decode", lambda _path: np.zeros(2000, dtype=np.float32))

    def fake_fast(_pcm, _sr, _words, *, surah, trace, log_event):
        trace.update({
            "drift_detected": True,
            "mean_conf": 0.2,
            "verify_sim": 0.1,
            "ayahs": [{"ayah": 1, "conf": 0.2}],
        })
        return [(1, 111, 0.2)]

    monkeypatch.setattr(align_fast, "align_surah_fast", fake_fast)

    trace: dict = {}
    rows = pipeline._fast_align(
        "dummy.mp3",
        11,
        [SimpleNamespace(ayah=1)],
        trace=trace,
        log_event=logs.append,
    )

    assert rows == [(11, 1, 111)]
    assert trace["drift_detected"] is True
    assert "fallback_from" not in trace
    assert not any("tiled" in line for line in logs)


def test_fast_align_ignores_legacy_drift_fallback_env(monkeypatch):
    monkeypatch.setenv("SC_FAST_DRIFT_FALLBACK", "tiled")
    monkeypatch.setattr(decode, "DEFAULT_SR", 1000)
    monkeypatch.setattr(decode, "decode", lambda _path: np.zeros(2000, dtype=np.float32))

    def fake_fast(_pcm, _sr, _words, *, surah, trace, log_event):
        trace.update({"drift_detected": True, "mean_conf": 0.2, "verify_sim": 0.1})
        return [(1, 111, 0.2)]

    monkeypatch.setattr(align_fast, "align_surah_fast", fake_fast)

    trace: dict = {}
    rows = pipeline._fast_align(
        "dummy.mp3",
        11,
        [SimpleNamespace(ayah=1)],
        trace=trace,
    )

    assert rows == [(11, 1, 111)]
    assert trace["drift_detected"] is True


def test_fast_align_forwards_optional_fingerprint_clips(monkeypatch):
    monkeypatch.setattr(decode, "DEFAULT_SR", 1000)
    monkeypatch.setattr(decode, "decode", lambda _path: np.zeros(2000, dtype=np.float32))
    received: dict[str, object] = {}

    def fake_fast(_pcm, _sr, _words, *, surah, trace, log_event, ayah_audio_dir):
        received.update(surah=surah, ayah_audio_dir=ayah_audio_dir)
        return [(1, 111, 0.2)]

    monkeypatch.setattr(align_fast, "align_surah_fast", fake_fast)

    rows = pipeline._fast_align(
        "dummy.mp3",
        11,
        [SimpleNamespace(ayah=1)],
        ayah_audio_dir="/clips",
    )

    assert rows == [(11, 1, 111)]
    assert received == {"surah": 11, "ayah_audio_dir": "/clips"}
