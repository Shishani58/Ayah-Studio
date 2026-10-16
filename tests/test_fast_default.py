"""Production entrypoints share FAST proofs without legacy or postprocessing."""
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, align_fast, align_tiled, decode, fast_executor, lag, pipeline, repair
from surah_cutter.data import quran_text


def test_legacy_fast_entrypoint_is_absent():
    assert not hasattr(align_fast, "align_surah_fast_legacy")


@pytest.mark.parametrize("model", ["quran-phonetics", "whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_public_fast_uses_shared_executor_with_captured_inputs(model, monkeypatch):
    bundle = SimpleNamespace(emitter=SimpleNamespace(model_id=model))
    pcm = np.zeros(160000, np.float32)
    words = quran_text.surah_words(112)
    trace, decisions, progress, logs = {}, [], [], []
    expected = [(1, 200, .9)]

    def forbidden(*args, **kwargs):
        pytest.fail("public FAST selected legacy or full")
    def execute(local, sr, canonical, **kwargs):
        assert local is pcm and sr == 16000 and canonical is words
        assert kwargs["bundle"] is bundle and kwargs["trace"] is trace
        assert kwargs["pause_decisions"] is decisions
        kwargs["ayah_progress"](1)
        kwargs["log_event"]("shared")
        return expected

    monkeypatch.setattr(align_fast, "align_surah_fast_legacy", forbidden, raising=False)
    monkeypatch.setattr(align, "align_surah", forbidden)
    monkeypatch.setattr(align_tiled, "align_surah_tiled", forbidden)
    monkeypatch.setattr(fast_executor, "align_surah_sequential", execute)
    assert align_fast.align_surah_fast(pcm, 16000, words, bundle=bundle, trace=trace,
        pause_decisions=decisions, ayah_progress=progress.append, log_event=logs.append) is expected
    assert progress == [1] and logs == ["shared"]


def test_explicit_fast_keeps_proven_rows_without_lag_or_full_repair(monkeypatch):
    bundle = object()
    trace = {}
    monkeypatch.setattr(align, "default_bundle", lambda: bundle)
    monkeypatch.setattr(decode, "decode", lambda _: np.zeros(120 * 16000, np.float32))

    def forbidden(*args, **kwargs):
        pytest.fail("FAST changed proof times or escalated")
    def execute(pcm, sr, words, **kwargs):
        assert kwargs["bundle"] is bundle
        return [(1, 1234, .9), (2, 3456, 0.)]

    monkeypatch.setattr(fast_executor, "align_surah_sequential", execute)
    monkeypatch.setattr(align_fast, "align_surah_fast_legacy", forbidden, raising=False)
    monkeypatch.setattr(align, "align_surah", forbidden)
    monkeypatch.setattr(align_tiled, "align_surah_tiled", forbidden)
    monkeypatch.setattr(lag, "calibrate_rows_to_pauses", forbidden)
    monkeypatch.setattr(repair, "repair_surah_timings", forbidden)
    assert pipeline._fast_align("fake", 112, quran_text.surah_words(112), trace=trace) == [
        (112, 1, 1234), (112, 2, 3456)]


def test_fast_debug_log_reports_shared_transitions_and_review_estimates():
    from surah_cutter.webgui.server import _format_fast_log
    trace = {"fast_version": fast_executor.VERSION,
             "transitions": [{"state": "REVIEW_REQUIRED", "ayah": 1, "reason": "no_local_evidence"}],
             "model_reads": [{"window_ms": [100, 1400], "observed": {"text": "بسم الله"}}],
             "pause_matches": [{"ayah": 1, "boundary_evidence": "estimate"}]}
    log = _format_fast_log(112, [], trace, [(1, 100, 0.)])
    assert fast_executor.VERSION in log and "REVIEW_REQUIRED" in log
    assert "no_local_evidence" in log and "بسم الله" in log and "prior/review" in log
