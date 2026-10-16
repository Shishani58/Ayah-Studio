from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from surah_cutter.core import align, align_fast, decode, pipeline


def test_default_is_full_and_modes_never_use_reciter_reference(monkeypatch):
    from surah_cutter.data import quran_text

    monkeypatch.setattr(quran_text, "surah_words", lambda surah: [])
    monkeypatch.setattr(pipeline, "_full_align", lambda *args, **kw: [(114, 1, 100)])
    monkeypatch.setattr(pipeline, "_fast_align", lambda *args, **kw: [(114, 1, 200)])
    monkeypatch.setattr(pipeline, "_tiled_align", lambda *args, **kw: [(114, 1, 300)])
    assert pipeline.process_surah("audio.mp3", 114) == [(114, 1, 100)]
    for mode, time in [("full", 100), ("fast", 200), ("tiled", 300)]:
        trace = {}
        assert pipeline.process_surah("audio.mp3", 114, mode=mode, trace=trace) == [(114, 1, time)]
        assert not trace["model_source"].startswith("reference:")


def test_removed_auto_mode_is_rejected_before_processing():
    import pytest

    with pytest.raises(ValueError, match="unknown mode"):
        pipeline.process_surah("missing.mp3", 114, mode="auto")


def test_fast_align_keeps_fast_rows_on_drift(monkeypatch):
    logs: list[str] = []
    monkeypatch.setattr(decode, "DEFAULT_SR", 1000)
    monkeypatch.setattr(decode, "decode", lambda _path: np.zeros(2000, dtype=np.float32))
    monkeypatch.setattr(align, "default_bundle", lambda: object())

    def fake_fast(_pcm, _sr, _words, *, surah, trace, log_event, bundle):
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
    monkeypatch.setattr(align, "default_bundle", lambda: object())

    def fake_fast(_pcm, _sr, _words, *, surah, trace, log_event, bundle):
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


def test_fast_align_rejects_fingerprint_clips_before_model_or_decode(monkeypatch):
    import pytest

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported clips loaded a model or audio")
    monkeypatch.setattr(align, "default_bundle", forbidden)
    monkeypatch.setattr(decode, "decode", forbidden)
    with pytest.raises(ValueError, match="fingerprint"):
        pipeline._fast_align("dummy.mp3", 11, [SimpleNamespace(ayah=1)], ayah_audio_dir="/clips")


def test_full_alignment_forwards_confidence_trace_and_logs(monkeypatch):
    from surah_cutter.core import align

    monkeypatch.setattr(decode, "decode", lambda _: np.zeros(16_000, np.float32))

    def fake_full(_pcm, _sr, _words, *, surah, trace, log_event):
        trace["ayahs"] = [{"ayah": 1, "conf": 0.01, "reliable": False}]
        log_event("weak-onset")
        return [(1, 6000, 0.01)]

    monkeypatch.setattr(align, "align_surah", fake_full)
    trace, logs = {}, []
    rows = pipeline._full_align("dummy.mp3", 114, [], trace=trace, log_event=logs.append)

    assert rows == [(114, 1, 6000)]
    assert trace["ayahs"][0]["conf"] == 0.01
    assert logs[-1] == "weak-onset"
