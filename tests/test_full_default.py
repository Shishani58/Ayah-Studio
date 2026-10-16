from types import SimpleNamespace
import threading

import numpy as np
import pytest

from surah_cutter.core import align, align_tiled, full_windows
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope
from surah_cutter.core.emissions import OnnxEmitter
from surah_cutter.data import transliteration
from surah_cutter.data.alignment_result import result_source, write_alignment_result
from surah_cutter.data.project_db import ProjectDB


class Session:
    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input_values")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, outputs, inputs):
        self.calls += 1
        frames = inputs["input_values"].shape[1] // 320 - 1
        return [np.tile([-10., 0., -10.], (1, frames, 1)).astype(np.float32)]


def bundle():
    return align._OnnxBundle(OnnxEmitter(Session(), {"[PAD]": 0, "a": 1, "|": 2},
                                        blank_id=0, do_normalize=False))


def targets(monkeypatch):
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["a"])
    monkeypatch.setattr(align, "_build_targets", lambda *args: ([1], [(0, 1)]))
    return [SimpleNamespace(surah=1, ayah=1, text="ا")]


def matrix():
    return np.tile([-10., 0., -10.], (50, 1)).astype(np.float32)


def test_short_exact_full_uses_pause_policy_and_records_actual_windows(monkeypatch):
    words = targets(monkeypatch)
    received = {}

    def generate(emitter, pcm, **kwargs):
        received.update(kwargs)
        kwargs["trace"].update(window_s=20, workers=3, context_s=2,
                               pause_policy="strong-dsp-v1", max_window_s=60,
                               windows=[{"start_ms": 0, "end_ms": 1000}])
        return matrix()

    monkeypatch.setattr(full_windows, "generate_pause_emissions", generate)
    monkeypatch.setattr(align, "_full_ctc_requires_tiling", lambda *args: False)
    trace = {}
    rows = align.align_surah(np.ones(16_000), 16_000, words, model=bundle(), trace=trace)
    assert [(a, ms) for a, ms, _ in rows] == [(1, 0)]
    assert received["window_s"] == 20 and received["workers"] == 3
    assert received["strong_pauses"] is True and received["max_window_s"] == 60
    assert trace["full_windows"]["algorithm"] == full_windows.FULL_POLICY
    assert trace["full_windows"]["windows"] == [{"start_ms": 0, "end_ms": 1000}]


def test_large_full_passes_pause_policy_to_bounded_ctc(monkeypatch):
    words = targets(monkeypatch)
    monkeypatch.setattr(align, "_full_ctc_requires_tiling", lambda *args: True)
    received = {}

    def tiled(pcm, sr, words, **kwargs):
        received.update(kwargs)
        kwargs["trace"].update(ayahs=[], full_windows={"pause_decisions": []})
        return [(1, 100, .5)]

    monkeypatch.setattr(align_tiled, "align_surah_tiled", tiled)
    assert align.align_surah(np.ones(16_000), 16_000, words, model=bundle()) == [(1, 100, .5)]
    assert received["parallel_full"] is True
    assert received["check_onsets"] is True


@pytest.mark.parametrize("parallel_full", [False, True])
def test_tiled_uses_new_emissions_only_when_called_from_full(monkeypatch, parallel_full):
    words = targets(monkeypatch)
    calls = []

    def old(*args):
        calls.append("old")
        return matrix()

    def new(*args, **kwargs):
        calls.append("full")
        return matrix()

    monkeypatch.setattr(align, "_emissions", old)
    monkeypatch.setattr(align, "_full_emissions", new)
    rows = align_tiled.align_surah_tiled(
        np.ones(16_000), 16_000, words, bundle=bundle(), parallel_full=parallel_full,
    )
    assert [(a, ms) for a, ms, _ in rows] == [(1, 0)]
    assert calls == ["full" if parallel_full else "old"]


def test_general_emitter_and_non_qpc_full_do_not_use_pause_policy(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local/FAST and other models must not call QPC full windowing")

    monkeypatch.setattr(full_windows, "generate_pause_emissions", forbidden)
    model = bundle()
    actual = align._emissions(model, np.ones(16_000), np)
    assert actual.shape == (49, 3)
    model.text_kind = "arabic"
    trace = {}
    np.testing.assert_array_equal(align._full_emissions(model, np.ones(16_000), np, trace=trace),
                                  actual)
    assert "full_windows" not in trace


def test_production_rejects_long_gap_before_any_onnx_call(monkeypatch):
    monkeypatch.setattr(full_windows, "detect_strong_pauses", lambda *a, **kw: [])
    model = bundle()
    with pytest.raises(ValueError, match="между подтверждёнными паузами"):
        align._full_emissions(model, np.ones(70 * 16_000), np)
    assert model.emitter.session.calls == 0


def test_cancelled_full_cannot_start_inference():
    model = bundle()
    event = threading.Event()
    event.set()
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        align._full_emissions(model, np.ones(16_000), np)
    assert model.emitter.session.calls == 0


def test_full_policy_is_saved_in_provenance_and_trace(tmp_path):
    trace = {"model_source": "onnx-ctc:test", "mode": "full", "backend": "onnx",
             "align_lag_ms": 200, "ayahs": [{"ayah": 1, "conf": .7, "reliable": False}],
             "full_windows": {"algorithm": full_windows.FULL_POLICY, "window_s": 20,
                              "workers": 3, "context_s": 2, "max_window_s": 60,
                              "pause_policy": "strong-dsp-v1", "windows": []}}
    path = tmp_path / "project.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        saved = write_alignment_result(db, [(1, 1, 100)], trace=trace, requested_mode="full",
                                       db_path=path)
        review = db.get_review(1, 1)
    assert saved.written == 1
    assert ":full_window_s=20:full_workers=3:full_context_s=2:" in review["source"]
    assert f":full_policy={full_windows.FULL_POLICY}:" in review["source"]
    assert review["source"] == result_source(trace, "full")


@pytest.mark.parametrize("tokens,onset,reason", [
    ([.9, .99, .001], 1000, "weak-word-token"),
    ([.62, .99, .99], 1315, "late-onset-in-pause"),
])
def test_full_review_warns_for_unsupported_token_or_late_silence_mark(tokens, onset, reason):
    check = {"ayah": 7, "onset_ms": onset, "conf": .94, "reliable": True,
             "token_confs": tokens}
    windows = {"review_policy": full_windows.FULL_REVIEW_POLICY, "pause_decisions": [
        {"accepted": True, "duration_ms": 940, "start_ms": 480, "end_ms": 1420,
         "center_ms": 950},
    ]}
    align._review_pause_full_onsets([check], windows)
    assert check["reliable"] is False and check["reason"] == reason
    assert check["onset_ms"] == onset and check["conf"] == .94


def test_full_review_keeps_supported_onset_near_pause_center():
    check = {"ayah": 7, "onset_ms": 1050, "conf": .94, "reliable": True,
             "token_confs": [.62, .99, .99]}
    windows = {"pause_decisions": [
        {"accepted": True, "duration_ms": 940, "start_ms": 480, "end_ms": 1420,
         "center_ms": 950},
    ]}
    align._review_pause_full_onsets([check], windows)
    assert check["reliable"] is True
