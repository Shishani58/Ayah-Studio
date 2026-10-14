from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from surah_cutter.core import full_windows
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope, current_event
from surah_cutter.core.candidates import Pause
from surah_cutter.core.emissions import OnnxEmitter


class SampleSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.maximum = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input_values")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, _outputs, inputs):
        batch = inputs["input_values"]
        assert batch.shape[0] == 1
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            # Different finish order; every output is tied to its input samples.
            time.sleep(.005 if batch[0, 0] else .02)
            frames = batch.shape[1] // 320 - 1
            values = batch[0, :frames * 320:320]
            return [np.stack([values, -values, np.zeros(frames)], axis=-1)[None]]
        finally:
            with self.lock:
                self.active -= 1


def emitter(session=None):
    return OnnxEmitter(session or SampleSession(), {"[PAD]": 0, "a": 1, "|": 2},
                       blank_id=0, do_normalize=False)


def test_pause_plan_covers_every_sample_and_keeps_bounded_spans():
    pauses = [Pause(980, 1060, 80), Pause(2180, 2260, 80), Pause(3040, 3120, 80)]
    spans = full_windows.plan_pause_windows(67_321, pauses, window_s=1)
    assert spans[0][0] == 0
    assert spans[-1][1] >= 67_321
    assert spans[-1][1] - 67_321 < 320
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(0 < right - left <= 24_000 for left, right, _ in spans)
    assert all(left % 320 == right % 320 == 0 for left, right, _ in spans)
    assert all(reason == "pause" for _, _, reason in spans[:-1])
    for _, end, _ in spans[:-1]:
        assert any(p.start_ms <= end / 16 <= p.end_ms for p in pauses)


def test_no_pause_cuts_keep_fixed_stride_and_small_tail_is_merged():
    spans = full_windows.plan_pause_windows(54_401, [], window_s=1)
    assert spans == [(0, 16_000, "no-pause"), (16_000, 32_000, "no-pause"),
                     (32_000, 54_720, "end")]


def test_concurrent_windows_preserve_order_frames_and_single_window_scale(monkeypatch):
    monkeypatch.setattr("surah_cutter.core.candidates.detect_energy_pauses", lambda *a, **kw: [])
    pcm = np.arange(96_123, dtype=np.float32) / 96_123
    sequential = full_windows.generate_pause_emissions(emitter(), pcm, window_s=1, context_s=.1)
    parallel_emitter = emitter()
    trace = {}
    parallel = full_windows.generate_pause_emissions(
        parallel_emitter, pcm, window_s=1, context_s=.1, workers=3, trace=trace,
    )
    np.testing.assert_array_equal(parallel, sequential)
    assert parallel_emitter.session.maximum == 3
    assert parallel.shape == (301, 3)
    # Core values should be exactly the original contiguous samples, including
    # the partial final frame, with no duplicate or skipped frame at any seam.
    values = pcm[::320]
    expected = full_windows._log_softmax(
        np.stack([values, -values, np.zeros(len(values))], axis=-1),
    )
    np.testing.assert_allclose(parallel, expected)
    assert trace["fallback_cuts"] > 0
    assert trace["windows"][-1]["end_ms"] == len(pcm) / 16


def test_short_clip_matches_original_exactly_without_dsp(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A short clip should preserve the existing single pass")

    monkeypatch.setattr("surah_cutter.core.candidates.detect_energy_pauses", forbidden)
    model = emitter()
    pcm = np.linspace(-.5, .5, 16_000, dtype=np.float32)
    expected = model.emissions(pcm, window_s=1)
    actual = full_windows.generate_pause_emissions(model, pcm, window_s=1, workers=4)
    np.testing.assert_array_equal(actual, expected)


def test_normalization_is_global_and_seams_do_not_change_statistics(monkeypatch):
    monkeypatch.setattr("surah_cutter.core.candidates.detect_energy_pauses", lambda *a, **kw: [])
    model = OnnxEmitter(SampleSession(), {"[PAD]": 0, "a": 1, "|": 2}, blank_id=0)
    pcm = np.linspace(-2, 5, 96_123, dtype=np.float32)
    actual = full_windows.generate_pause_emissions(model, pcm, window_s=1, context_s=.1)
    normalized = full_windows._normalize(pcm)[::320]
    expected = full_windows._log_softmax(
        np.stack([normalized, -normalized, np.zeros(len(normalized))], axis=-1),
    )
    np.testing.assert_allclose(actual, expected)


def test_frame_grid_cannot_turn_a_tiny_pause_into_a_cut_in_speech():
    pauses = [Pause(1009, 1011, 2)]  # Center rounds to 1000 ms, outside this pause.
    spans = full_windows.plan_pause_windows(64_000, pauses, window_s=1)
    assert spans[0] == (0, 16_000, "no-pause")


def test_cancellation_event_is_propagated_to_workers(monkeypatch):
    monkeypatch.setattr("surah_cutter.core.candidates.detect_energy_pauses", lambda *a, **kw: [])
    event = threading.Event()
    session = SampleSession()

    def cancel(_outputs, _inputs, _options=None):
        assert current_event() is event
        event.set()
        raise GenerationCancelled()

    session.run = cancel
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        full_windows.generate_pause_emissions(emitter(session), np.ones(96_000),
                                             window_s=1, workers=2)


def test_inference_failure_is_not_replaced_with_a_partial_result(monkeypatch):
    monkeypatch.setattr("surah_cutter.core.candidates.detect_energy_pauses", lambda *a, **kw: [])
    session = SampleSession()

    def fail(_outputs, _inputs):
        raise ValueError("model failure")

    session.run = fail
    with pytest.raises(ValueError, match="model failure"):
        full_windows.generate_pause_emissions(emitter(session), np.ones(96_000),
                                             window_s=1, workers=2)


def test_incompatible_frame_layout_is_rejected(monkeypatch):
    monkeypatch.setattr("surah_cutter.core.candidates.detect_energy_pauses", lambda *a, **kw: [])
    session = SampleSession()
    session.run = lambda *a: [np.zeros((1, 10, 3), np.float32)]
    with pytest.raises(ValueError, match="frame layout"):
        full_windows.generate_pause_emissions(emitter(session), np.ones(96_000), window_s=1)


@pytest.mark.parametrize("options", [
    {"workers": 0}, {"workers": 7}, {"context_s": 0}, {"context_s": .031},
    {"window_s": 0}, {"window_s": float("nan")},
])
def test_invalid_settings_are_rejected(options):
    with pytest.raises(ValueError):
        full_windows.generate_pause_emissions(emitter(), np.ones(16_000), **options)


def test_pause_only_extends_to_distant_pause_and_never_cuts_speech():
    pauses = [Pause(2950, 3050, 100), Pause(6750, 6850, 100)]
    spans = full_windows.plan_pause_windows(160_000, pauses, window_s=1, pause_only=True)
    assert spans == [(0, 48_000, "pause"), (48_000, 108_800, "pause"),
                     (108_800, 160_000, "end")]
    assert full_windows.plan_pause_windows(160_000, [], window_s=1, pause_only=True) == [
        (0, 160_000, "end"),
    ]


def test_strong_pause_gate_rejects_tonal_tail_and_records_evidence(monkeypatch):
    from surah_cutter.core import candidates

    pauses = [Pause(1000, 1800, 800), Pause(3000, 4000, 1000)]

    def detect(pcm, sr, **kwargs):
        assert sr == 16_000
        assert kwargs == {"min_pause_ms": 300, "use_ten_vad": False}
        return pauses

    monkeypatch.setattr(candidates, "detect_energy_pauses", detect)
    monkeypatch.setattr(candidates, "_dsp_pause_metrics", lambda pcm, sr, p: {
        "rms_ratio": .05 if p is pauses[0] else .5, "flatness": .01, "zcr": .01,
    })
    trace = {}
    assert full_windows.detect_strong_pauses(np.ones(80_000), trace=trace) == pauses[:1]
    assert [p["accepted"] for p in trace["pause_decisions"]] == [True, False]


@pytest.mark.parametrize("workers", [5, 6])
def test_strong_windows_support_six_workers_and_preserve_samples(monkeypatch, workers):
    pauses = [Pause(ms - 40, ms + 40, 80) for ms in range(1000, 12_000, 1000)]
    monkeypatch.setattr(full_windows, "detect_strong_pauses", lambda *a, **kw: pauses)
    pcm = np.linspace(0, 1, 192_123, dtype=np.float32)
    model = emitter()
    trace = {}
    actual = full_windows.generate_pause_emissions(
        model, pcm, window_s=1, context_s=.1, workers=workers,
        strong_pauses=True, trace=trace,
    )
    assert model.session.maximum == workers
    assert trace["fallback_cuts"] == 0
    assert all(w["cut_reason"] == "pause" for w in trace["windows"][:-1])
    values = pcm[::320]
    expected = full_windows._log_softmax(
        np.stack([values, -values, np.zeros(len(values))], axis=-1),
    )
    np.testing.assert_allclose(actual, expected)
