from __future__ import annotations

import numpy as np

from surah_cutter.core import candidates


def _speech_noise(duration_ms: float, sr: int, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_ms * sr / 1000.0))
    return (amp * rng.normal(0.0, 1.0, n)).astype(np.float32)


def _silence(duration_ms: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(duration_ms * sr / 1000.0)), dtype=np.float32)


def test_combined_pauses_recovers_short_weak_pause():
    sr = 16_000
    pcm = np.concatenate([
        _speech_noise(1000.0, sr, 0.30, 1),
        _silence(260.0, sr),
        _speech_noise(1000.0, sr, 0.30, 2),
    ])

    strict = candidates.detect_energy_pauses(pcm, sr)
    decisions = candidates.pause_pipeline_decisions(pcm, sr, surah=1)

    assert strict == []
    assert len(decisions) == 1
    assert decisions[0].accepted
    assert decisions[0].reason == "weak-dsp"
    assert 990.0 <= decisions[0].pause.start_ms <= 1010.0
    assert 1250.0 <= decisions[0].pause.end_ms <= 1270.0


def test_energy_pause_edges_use_raw_rms_not_smoothed_shelf():
    sr = 16_000
    pcm = np.concatenate([
        _speech_noise(1000.0, sr, 0.30, 11),
        _silence(520.0, sr),
        _speech_noise(1000.0, sr, 0.30, 12),
    ])

    pauses = candidates.detect_energy_pauses(pcm, sr)

    assert len(pauses) == 1
    assert 990.0 <= pauses[0].start_ms <= 1010.0
    assert 1510.0 <= pauses[0].end_ms <= 1530.0
    assert 510.0 <= pauses[0].depth_ms <= 530.0


def test_dsp_pause_metrics_handles_non_power_of_two_segment():
    sr = 16_000
    pcm = _speech_noise(1000.0, sr, 0.02, 13)
    pause = candidates.Pause(100.0, 212.7, 112.7)

    metrics = candidates._dsp_pause_metrics(pcm, sr, pause)

    assert set(metrics) == {"flatness", "zcr", "rms_ratio"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_combined_pauses_does_not_duplicate_strict_pause():
    sr = 16_000
    pcm = np.concatenate([
        _speech_noise(1000.0, sr, 0.30, 3),
        _silence(520.0, sr),
        _speech_noise(1000.0, sr, 0.30, 4),
    ])

    decisions = candidates.pause_pipeline_decisions(pcm, sr, surah=1)

    assert len(decisions) == 1
    assert decisions[0].accepted
    assert decisions[0].reason == "dsp"


def test_combined_pauses_respects_explicit_higher_min_pause():
    sr = 16_000
    pcm = np.concatenate([
        _speech_noise(1000.0, sr, 0.30, 5),
        _silence(260.0, sr),
        _speech_noise(1000.0, sr, 0.30, 6),
    ])

    decisions = candidates.pause_pipeline_decisions(pcm, sr, surah=1, min_pause_ms=500.0)

    assert decisions == []


def test_intro_vad_cannot_expand_pause_into_sounding_audio(monkeypatch):
    energy_pause = candidates.Pause(5700, 6300, 600)
    broad_vad = candidates.Pause(4600, 6310, 1710)
    monkeypatch.setattr(candidates, "_pause_recall_candidates", lambda *a, **k:
                        [(energy_pause, False)])
    monkeypatch.setattr(candidates, "_dsp_pause_metrics", lambda *a:
                        {"flatness": 0.1, "zcr": 0.1, "rms_ratio": 0.05})
    monkeypatch.setattr(candidates, "_dsp_accepts_pause", lambda *a: True)
    monkeypatch.setattr(candidates, "_local_vad_pauses", lambda *a, **k: ([], [broad_vad]))
    monkeypatch.setattr(candidates, "_vad_confirms_pause", lambda *a: True)

    decision = candidates.pause_pipeline_decisions(np.zeros(16_000), 16_000, surah=114)[0]

    assert decision.accepted
    assert decision.pause == energy_pause
    assert decision.reason == "intro-vad"
