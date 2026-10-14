from types import SimpleNamespace

import numpy as np

from surah_cutter.core.fast_adapters import Observation
from surah_cutter.core.fast_reads import LocalEvidenceReads


def reader(monkeypatch, seconds=20):
    calls = []

    def infer(pcm, sr, *, total_samples):
        calls.append(len(pcm))
        return Observation(("test",), None, .95, unit_quality=(.95,), frames=len(pcm) // 20)

    adapter = SimpleNamespace(observe=infer, metric="ctc_greedy_probability")
    return LocalEvidenceReads(np.zeros(seconds * 1000), 1000, adapter), calls


def test_whole_sura_and_large_window_never_reach_model(monkeypatch):
    reads, calls = reader(monkeypatch)
    assert reads.read(0, 20_000, reason="test") is None
    assert reads.read(3000, 10_000, reason="test") is None
    assert calls == []
    assert reads.diagnostics()["emission_passes"] == 0


def test_local_cache_accounts_actual_inference_and_union(monkeypatch):
    reads, calls = reader(monkeypatch)
    assert reads.read(2000, 4000, reason="test") is not None
    assert reads.read(2000, 4000, reason="cached") is not None
    assert reads.read(2500, 3500, reason="new") is not None
    assert reads.read(3000, 5000, reason="new") is not None
    assert calls == [2000, 1000, 2000]
    diag = reads.diagnostics()
    assert diag["emission_passes"] == 3
    assert diag["audio_ms"] == 5000
    assert diag["covered_audio_ms"] == 3000
    assert diag["coverage"] == 0.15
    assert diag["acoustic_intervals_ms"] == [[2000, 5000]]


def test_sequential_tiles_cannot_cover_recording(monkeypatch):
    reads, calls = reader(monkeypatch)
    for start in range(0, 20_000, 2000):
        reads.read(start, start + 2000, reason="test")
    assert sum(calls) <= 13_000
    assert reads.covered_ms <= 13_000
    assert reads.denied[-1]["reason"] == "coverage_budget"


def test_repeated_expansion_cannot_exhaust_inference_budget(monkeypatch):
    reads, calls = reader(monkeypatch)
    for start in range(50):
        reads.read(start, 4000 + start, reason="test")
    assert sum(calls) <= 40_000
    assert reads.denied[-1]["reason"] == "inference_budget"


def test_short_recording_cannot_be_read_whole(monkeypatch):
    reads, calls = reader(monkeypatch, seconds=2)
    assert reads.read(0, 2000, reason="test") is None
    assert calls == []
