import threading
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope
from surah_cutter.core.fast_reads import LocalAcousticReads, LocalWhisperReads


@pytest.fixture(params=["ctc", "whisper"])
def local_reader(request, monkeypatch):
    calls = []
    behavior = SimpleNamespace(callback=None)

    def infer(pcm):
        calls.append(len(pcm))
        if behavior.callback:
            behavior.callback()
        if request.param == "ctc":
            return np.zeros((len(pcm) // 20, 3), dtype=np.float32)
        return {"text": "test", "token_confidence": 0.95, "truncated": False}

    pcm = np.zeros(20_000)
    if request.param == "ctc":
        monkeypatch.setattr(align, "_emissions", lambda bundle, pcm, np_module: infer(pcm))
        reads = LocalAcousticReads(pcm, 1000, SimpleNamespace())
        read = reads.read
    else:
        emitter = SimpleNamespace(recognize_local=lambda pcm, **kwargs: infer(pcm))
        reads = LocalWhisperReads(pcm, 1000, emitter)

        def read(start, end):
            return reads.read(start, end, reason="test")
    return reads, read, calls, behavior


def test_failed_retries_are_charged_and_eventually_denied(local_reader):
    reads, read, calls, behavior = local_reader
    failure = RuntimeError("model failed")

    def fail():
        # The attempt must be accounted before entering the model.
        assert reads.audio_ms == len(calls) * 4000
        assert len(reads.calls) == len(calls)
        raise failure

    behavior.callback = fail
    for _ in range(10):
        with pytest.raises(RuntimeError) as caught:
            read(2000, 6000)
        assert caught.value is failure
    assert read(2000, 6000) is None
    assert calls == [4000] * 10
    assert reads.audio_ms == 40_000
    assert reads.covered_ms == 4000
    assert reads.diagnostics()["acoustic_intervals_ms"] == [[2000, 6000]]
    assert reads.denied[-1]["reason"] == "inference_budget"
    assert all(call["outcome"] == "failed" for call in reads.calls)
    assert reads.cache == []
    assert not getattr(reads, "results", {})


def test_failed_distinct_reads_still_limit_coverage(local_reader):
    reads, read, calls, behavior = local_reader

    def fail():
        raise RuntimeError("model failed")

    behavior.callback = fail
    for start in range(0, 12_000, 2000):
        with pytest.raises(RuntimeError):
            read(start, start + 2000)
    assert read(12_000, 14_000) is None
    assert calls == [2000] * 6
    assert reads.covered_ms == reads.audio_ms == 12_000
    assert reads.denied[-1]["reason"] == "coverage_budget"


@pytest.mark.parametrize("raises", [False, True])
def test_cancelled_inference_is_charged_but_not_cached(local_reader, raises):
    reads, read, calls, behavior = local_reader
    event = threading.Event()

    def cancel():
        event.set()
        if raises:
            raise GenerationCancelled()

    behavior.callback = cancel
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        read(2000, 4000)
    assert reads.audio_ms == reads.covered_ms == 2000
    assert reads.calls[0]["outcome"] == "cancelled"
    assert reads.cache == []
    assert not getattr(reads, "results", {})

    event.clear()
    behavior.callback = None
    with cancellation_scope(event):
        assert read(2000, 4000) is not None
    assert calls == [2000, 2000]
    assert reads.audio_ms == 4000
    assert reads.covered_ms == 2000
    assert reads.calls[1]["outcome"] == "completed"


def test_cancellation_precedes_cached_evidence(local_reader):
    reads, read, calls, _ = local_reader
    assert read(2000, 4000) is not None
    event = threading.Event()
    event.set()
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        read(2000, 4000)
    assert calls == [2000]
    assert reads.audio_ms == 2000
    assert len(reads.calls) == 1
    assert reads.calls[0]["outcome"] == "completed"


def test_cancellation_before_inference_does_not_charge(local_reader, monkeypatch):
    reads, read, calls, _ = local_reader
    event = threading.Event()
    permit = reads._permit

    def cancel_after_permit(*args):
        allowed = permit(*args)
        event.set()
        return allowed

    monkeypatch.setattr(reads, "_permit", cancel_after_permit)
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        read(2000, 4000)
    assert calls == reads.calls == []
    assert reads.audio_ms == reads.covered_ms == 0


@pytest.mark.parametrize("start,end", [
    (float("nan"), 4000), (2000, float("nan")),
    (float("inf"), 4000), (2000, float("-inf")),
    (4000, 2000), (2000, 2000), (None, 4000), (2000, "bad"),
])
def test_invalid_windows_never_infer_or_change_accounting(local_reader, start, end):
    reads, read, calls, _ = local_reader
    assert read(start, end) is None
    assert calls == reads.calls == []
    assert reads.audio_ms == reads.covered_ms == 0
