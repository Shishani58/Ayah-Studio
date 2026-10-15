"""Regression guard for repeated whole-audio RMS work in pause metrics."""
import numpy as np
from surah_cutter.core import candidates


def test_pipeline_reduces_whole_recording_once_per_run(monkeypatch):
    pcm = np.linspace(-.2, .2, 48_000, dtype=np.float32)
    pauses = [candidates.Pause(300, 600, 300), candidates.Pause(1000, 1400, 400),
              candidates.Pause(2000, 2300, 300)]
    monkeypatch.setattr(candidates, "_pause_recall_candidates",
                        lambda *args, **kwargs: [(p, False) for p in pauses])
    original_mean = np.mean
    whole_reductions = []

    def mean(array, *args, **kwargs):
        if np.asarray(array).size == pcm.size:
            whole_reductions.append(True)
        return original_mean(array, *args, **kwargs)

    monkeypatch.setattr(np, "mean", mean)
    first = candidates.pause_pipeline_decisions(pcm, 16_000, surah=1)
    assert len(first) == len(pauses)
    assert len(whole_reductions) == 1
    second = candidates.pause_pipeline_decisions(pcm.copy(), 16_000, surah=1)
    assert second == first
    assert len(whole_reductions) == 2
    for decision, pause in zip(first, pauses):
        standalone = candidates._dsp_pause_metrics(pcm, 16_000, pause)
        assert decision.rms_ratio == standalone["rms_ratio"]
        assert decision.flatness == standalone["flatness"]
        assert decision.zcr == standalone["zcr"]
