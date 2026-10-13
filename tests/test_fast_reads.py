from types import SimpleNamespace

import numpy as np

from surah_cutter.core import align
from surah_cutter.core.fast_reads import LocalAcousticReads


def reader(monkeypatch, seconds=20):
    calls = []

    def infer(bundle, pcm, np_module):
        calls.append(len(pcm))
        return np.zeros((len(pcm) // 20, 3), dtype=np.float32)

    monkeypatch.setattr(align, "_emissions", infer)
    return LocalAcousticReads(np.zeros(seconds * 1000), 1000, SimpleNamespace()), calls


def test_whole_sura_and_large_window_never_reach_model(monkeypatch):
    reads, calls = reader(monkeypatch)
    assert reads.read(0, 20_000) is None
    assert reads.read(3000, 10_000) is None
    assert calls == []
    assert reads.diagnostics()["emission_passes"] == 0


def test_local_cache_accounts_actual_inference_and_union(monkeypatch):
    reads, calls = reader(monkeypatch)
    assert reads.read(2000, 4000) is not None
    assert reads.read(2500, 3500) is not None
    assert reads.read(3000, 5000) is not None
    assert calls == [2000, 2000]
    diag = reads.diagnostics()
    assert diag["emission_passes"] == 2
    assert diag["audio_ms"] == 4000
    assert diag["covered_audio_ms"] == 3000
    assert diag["coverage"] == 0.15
    assert diag["acoustic_intervals_ms"] == [[2000, 5000]]


def test_sequential_tiles_cannot_cover_recording(monkeypatch):
    reads, calls = reader(monkeypatch)
    for start in range(0, 20_000, 2000):
        reads.read(start, start + 2000)
    assert sum(calls) <= 13_000
    assert reads.covered_ms <= 13_000
    assert reads.denied[-1]["reason"] == "coverage_budget"


def test_repeated_expansion_cannot_exhaust_inference_budget(monkeypatch):
    reads, calls = reader(monkeypatch)
    for start in range(50):
        reads.read(start, 4000 + start)
    assert sum(calls) <= 40_000
    assert reads.denied[-1]["reason"] == "inference_budget"


def test_short_recording_cannot_be_read_whole(monkeypatch):
    reads, calls = reader(monkeypatch, seconds=2)
    assert reads.read(0, 2000) is None
    assert calls == []


def test_fast_insufficient_audio_returns_review_without_escalation(monkeypatch):
    from surah_cutter.core import align_fast, align_tiled, candidates
    from surah_cutter.data import quran_text, transliteration

    def forbidden(*args, **kwargs):
        raise AssertionError("whole-sura inference/escalation")

    monkeypatch.setattr(align, "_emissions", forbidden)
    monkeypatch.setattr(align, "align_surah", forbidden)
    monkeypatch.setattr(align_tiled, "align_surah_tiled", forbidden)
    monkeypatch.setattr(candidates, "pause_pipeline_decisions", lambda *args, **kwargs: [])
    chars = sorted(set("".join(transliteration.surah_word_tr(112))))
    bundle = SimpleNamespace(vocab={c: i + 1 for i, c in enumerate(chars)}, blank_id=0, delim_id=len(chars) + 1)
    trace = {}
    rows = align_fast.align_surah_fast(np.zeros(200), 1000, quran_text.surah_words(112),
                                       surah=112, bundle=bundle, trace=trace)
    assert [a for a, _, _ in rows] == [1, 2, 3, 4]
    assert all(conf == 0 for _, _, conf in rows)
    assert trace["review_required_ayahs"] == [1, 2, 3, 4]
    assert trace["emission_passes"] == 0
    assert trace["bounded_ctc_fallback"] is None
