"""Work advances before ayah rows, without changing alignment results."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import emissions, model_runtime, progress
from surah_cutter.webgui import server
from surah_cutter.webgui.generation import GenerationEvents


def test_stage_rate_throttling_and_unknown_total(monkeypatch):
    now = [100.]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    seen = []
    with progress.progress_scope(seen.append):
        progress.report("model")
        assert seen[-1]["stage_total"] is None
        assert seen[-1]["stage_eta_seconds"] is None
        progress.report("acoustic", 0, 10)
        now[0] += 4
        progress.advance("acoustic", 2, 10)
        assert seen[-1]["stage_eta_seconds"] == 16
        progress.advance("acoustic", 3, 10)
        assert seen[-1]["stage_done"] == 2  # frame spam is throttled
        progress.advance("acoustic", 10, 10)
        assert seen[-1]["stage_done"] == 10
        assert seen[-1]["stage_eta_seconds"] is None
        progress.report("verify")
        progress.advance("acoustic", 1, 2)  # local recheck cannot reset the stage
        assert seen[-1]["stage"] == "verify"
    assert progress.current_stage() is None


def test_nested_scope_restores_outer_stage_even_on_failure(monkeypatch):
    now = [10.]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    seen = []
    with progress.progress_scope(seen.append):
        progress.report("acoustic", 0, 4)
        with pytest.raises(ValueError), progress.progress_scope(lambda _: None):
            progress.report("local", 0, 2)
            raise ValueError()
        now[0] = 14.
        progress.advance("acoustic", 1, 4)
    assert seen[-1]["stage_eta_seconds"] == 12


def test_window_progress_precedes_final_matrix_and_preserves_it(monkeypatch):
    clock = iter(range(100))
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(clock))
    seen, calls = [], []

    def run(session, batch, *args):
        calls.append(len(batch))
        if progress.current_stage() == "acoustic":
            assert seen[-1]["stage_done"] == len(calls) - 1
        return np.zeros((len(batch), batch.shape[1] // 320 - 1, 3), np.float32)

    monkeypatch.setattr(emissions, "_run", run)
    pcm = np.ones(5 * 16000, np.float32)
    with progress.progress_scope(seen.append):
        progress.report("acoustic")
        actual = emissions.generate_emissions(None, pcm, window_s=1, context_s=1)
    assert [e["stage_done"] for e in seen] == [None, 0, 1, 2, 3, 4, 5]
    expected = emissions.generate_emissions(None, pcm, window_s=1, context_s=1)
    np.testing.assert_array_equal(actual, expected)


def test_runtime_consumes_progress_envelopes_before_result():
    runtime = model_runtime.RuntimeEmitter.__new__(model_runtime.RuntimeEmitter)
    runtime.process = SimpleNamespace(stdout=io.StringIO(
        json.dumps({"progress": {"done": 0, "total": 3}}) + "\n" +
        json.dumps({"progress": {"done": 3, "total": 3}}) + "\n" +
        json.dumps({"rows": [[100, .9]]}) + "\n"))
    seen = []
    with progress.progress_scope(seen.append):
        progress.report("acoustic")
        result = runtime._read_response()
    assert result == {"rows": [[100, .9]]}
    assert seen[-1]["stage_done"] == 3


def test_job_reports_work_before_any_ayah_and_restores_stage(tmp_path, monkeypatch):
    job = server._GenJob(tmp_path / "progress.db", {114: tmp_path / "114.mp3"})

    def process(audio, surah, **kwargs):
        progress.report("acoustic", 0, 3)
        progress.report("acoustic", 3, 3)
        assert not any(e["type"] == "ayah_progress" for e in job.events.snapshot())
        assert job.status()["stage"]["stage_done"] == 3
        kwargs["ayah_progress"](1)
        return [(114, a, a * 1000) for a in range(1, 7)]

    monkeypatch.setattr("surah_cutter.core.pipeline.process_surah", process)
    job._run([114], "full")
    assert job.status()["terminal"]["type"] == "done"
    assert progress.current_stage() is None


def test_cancel_from_work_callback_never_finishes_stage_or_saves(tmp_path, monkeypatch):
    job = server._GenJob(tmp_path / "cancel.db", {114: tmp_path / "114.mp3"})

    def process(*args, **kwargs):
        progress.report("acoustic", 0, 10)
        job.request_cancel()
        progress.report("verify")
        raise AssertionError("cancel callback must interrupt")

    monkeypatch.setattr("surah_cutter.core.pipeline.process_surah", process)
    job._run([114], "full")
    assert job.status()["terminal"]["type"] == "cancelled"
    assert not any(e["type"] == "sura" for e in job.events.snapshot())


def test_stage_survives_eviction_separately_from_ayah_count():
    history = GenerationEvents(capacity=2)
    history.emit({"type": "ayah_progress", "ayahs_done": 1, "total_ayahs": 6})
    history.emit({"type": "stage_progress", "stage": "acoustic", "stage_done": 2})
    for _ in range(5):
        history.emit({"type": "sura_log", "line": "work"})
    assert history.status()["stage"]["stage_done"] == 2
    assert history.status()["ayahs_done"] == 1
    history.emit({"type": "sura_start"})
    assert history.status()["stage"] is None
