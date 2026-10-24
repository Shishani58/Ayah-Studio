from copy import deepcopy
import threading

import numpy as np
import pytest

from surah_cutter.core import candidates, emissions, full_boundary, full_signal
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope


def fixture():
    pcm = np.random.default_rng(12).normal(0, .1, 64000).astype(np.float32)
    pcm[16000:32000] = 0
    checks = {
        1: dict(ayah=1, onset_ms=100, raw_onset_ms=100, full_ctc_end_ms=900,
                full_ctc_tail_conf=.9, conf=.9, first_token_conf=.9),
        2: dict(ayah=2, onset_ms=2040, raw_onset_ms=2040, full_ctc_first_ms=2240,
                conf=.2, first_token_conf=.9, reliable=False, repaired=False,
                pause_center_reason="unstable-pause", pause_local_crop_ms=[100, 3240],
                pause_stability_crop_ms=[0, 4000],
                pause_local_decisions=[dict(start_ms=1000, end_ms=2000, accepted=True, owned=True)]),
    }
    return checks, pcm, {}


def run(data):
    checks, pcm, trace = data
    full_signal.refine_late_boundaries(checks, pcm, 16000, trace=trace)


def test_raw_waveform_refines_late_cut_without_any_model_calls(monkeypatch):
    data = fixture()
    original = deepcopy(data[0])

    def forbidden(*args, **kwargs):
        pytest.fail("signal refinement called an acoustic model")

    monkeypatch.setattr(emissions.OnnxEmitter, "emissions", forbidden)
    monkeypatch.setattr(full_boundary, "_muaalem_verifier", forbidden)
    run(data)
    check = data[0][2]
    assert check["onset_ms"] == 1750
    assert check["raw_onset_ms"] == 2040 and check["pre_signal_onset_ms"] == 2040
    assert check["conf"] == .2 and not check["reliable"]
    assert check["repaired"] and check["reason"] == "waveform-pause-recheck"
    assert data[0][1] == original[1]
    record = data[2]["full_signal_checks"][0]
    assert record["accepted"] and record["quiet_window_ms"] == [1730, 1770]
    assert record["rms_ratio"] == record["peak_ratio"] == record["periodicity"] == 0
    assert "full_boundary_reads" not in data[2]


@pytest.mark.parametrize("gain", [.1, 1, 4])
def test_gain_and_dsp_context_do_not_move_the_signal_cut(gain):
    data = fixture()
    data[1][:] *= gain
    data[0][2]["pause_local_crop_ms"] = [0, 4000]
    data[0][2]["pause_stability_crop_ms"] = [50, 3900]
    run(data)
    assert data[0][2]["onset_ms"] == 1750


def weak_onset_fixture():
    data = fixture()
    check = data[0][2]
    check.update(first_token_conf=.0003, token_confs=[.0003, .99, .85],
                 pause_center_reason="weak-evidence")
    for key in ("pause_local_crop_ms", "pause_stability_crop_ms", "pause_local_decisions"):
        del check[key]
    return data


@pytest.mark.parametrize("gain", [.1, 1, 4])
def test_silence_is_checked_when_only_first_phoneme_is_weak(gain, monkeypatch):
    data = weak_onset_fixture()
    data[1][:] *= gain

    def forbidden(*args, **kwargs):
        pytest.fail("weak onset silence check called an acoustic model")

    monkeypatch.setattr(emissions.OnnxEmitter, "emissions", forbidden)
    run(data)
    check = data[0][2]
    assert check["onset_ms"] == pytest.approx(1750, abs=30)
    assert check["raw_onset_ms"] == 2040 and check["first_token_conf"] == .0003
    assert check["token_confs"] == [.0003, .99, .85]
    assert check["conf"] == .2 and not check["reliable"]
    assert check["repaired"]
    assert data[2]["full_signal_checks"][0]["weak_head"]
    assert "pause_local_crop_ms" in check and "pause_stability_crop_ms" in check


@pytest.mark.parametrize("case", ["weak-word", "weak-tail", "intra-word", "tone", "speech",
                                  "ambiguous", "repaired"])
def test_weak_first_phoneme_does_not_remove_silence_and_ownership_guards(case, monkeypatch):
    data = weak_onset_fixture()
    check = data[0][2]
    if case == "weak-word":
        check["token_confs"] = [.0003, .1]
    elif case == "weak-tail":
        data[0][1]["full_ctc_tail_conf"] = .01
    elif case == "intra-word":
        data[0][1]["full_ctc_end_ms"] = 2100
    elif case == "tone":
        data[1][16000:32000] = .005 * np.sin(np.arange(16000) * 2 * np.pi * 150 / 16000)
    elif case == "speech":
        data[1][16000:32000] = np.random.default_rng(2).normal(0, .1, 16000)
    elif case == "ambiguous":
        from surah_cutter.core import full_pause_center

        original = full_pause_center._local_pauses

        def detect(*args, **kwargs):
            crop, decisions, supported = original(*args, **kwargs)
            decisions[:] = [dict(start_ms=a, end_ms=b, owned=True, accepted=True)
                            for a, b in ((1000, 1400), (1600, 2000))]
            return crop, decisions, supported

        monkeypatch.setattr(full_pause_center, "_local_pauses", detect)
    elif case == "repaired":
        check["repaired"] = True
    run(data)
    assert check["onset_ms"] == 2040


@pytest.mark.parametrize("case", ["weak-head", "weak-tail", "repaired", "positioned", "intraword",
                                  "previous-word", "order", "ambiguous", "missing-context",
                                  "already-safe", "too-far", "no-previous", "unbounded-gap"])
def test_no_guess_when_ownership_shelf_or_order_is_unsupported(case):
    data = fixture()
    checks, _, trace = data
    check = checks[2]
    if case == "weak-head":
        check["first_token_conf"] = .1
    elif case == "weak-tail":
        checks[1]["full_ctc_tail_conf"] = .1
    elif case == "repaired":
        check["repaired"] = True
    elif case == "positioned":
        check["pause_center_reason"] = "positioned"
    elif case == "intraword":
        check["pause_local_decisions"][0].update(start_ms=2100, end_ms=2240)
    elif case == "previous-word":
        checks[1]["full_ctc_end_ms"] = 1900
    elif case == "order":
        checks[1]["onset_ms"] = 1800
    elif case == "ambiguous":
        check["pause_local_decisions"] = [dict(start_ms=a, end_ms=b, accepted=True, owned=True)
                                           for a, b in ((1000, 1400), (1600, 2000))]
    elif case == "missing-context":
        del check["pause_stability_crop_ms"]
    elif case == "already-safe":
        check["onset_ms"] = 1700
    elif case == "too-far":
        check.update(onset_ms=3000, raw_onset_ms=3000, full_ctc_first_ms=3200)
    elif case == "no-previous":
        del checks[1]
    elif case == "unbounded-gap":
        check["full_ctc_first_ms"] = 10000
    old = check["onset_ms"]
    run(data)
    assert check["onset_ms"] == old
    assert not trace["full_signal_checks"]


@pytest.mark.parametrize("start,end,expected", [(1200, 2000, 1750), (1000, 1900, 1650),
                                               (1200, 1900, 1650), (1900, 2050, 2040)])
def test_context_edges_may_differ_but_only_common_silence_can_be_cut(start, end, expected):
    data = fixture()
    check = data[0][2]
    check['pause_stability_decisions'] = [dict(start_ms=start, end_ms=end, owned=True, accepted=True)]
    run(data)
    assert check['onset_ms'] == expected
    if expected != 2040:
        record = data[2]['full_signal_checks'][0]
        assert record['pause_ms'] == [max(1000, start), min(2000, end)]
        assert max(1000, start) < record['quiet_window_ms'][0]
        assert record['quiet_window_ms'][1] < min(2000, end)
        assert check['conf'] == .2 and not check['reliable']


@pytest.mark.parametrize("tail,end", [(1020, 2000), (900, 2080)])
def test_pause_edge_may_cross_speech_anchor_but_the_quiet_window_cannot(tail, end):
    data = fixture()
    data[0][1]['full_ctc_end_ms'] = tail
    data[0][2]['pause_local_decisions'][0]['end_ms'] = end
    run(data)
    record = data[2]['full_signal_checks'][0]
    assert record['accepted']
    assert record['quiet_window_ms'][0] >= tail
    assert record['quiet_window_ms'][1] <= min(data[0][2]['raw_onset_ms'], data[0][2]['full_ctc_first_ms'])
    assert record['rms_ratio'] == record['peak_ratio'] == record['periodicity'] == 0
    assert not data[0][2]['reliable']


def test_proven_pause_is_not_replaced_by_a_later_shelf_crossing_the_first_sound():
    data = fixture()
    data[0][2]['pause_stability_decisions'] = [dict(start_ms=2030, end_ms=2190, owned=True, accepted=True)]
    run(data)
    record = data[2]['full_signal_checks'][0]
    assert record['accepted'] and record['after_ms'] == 1750
    assert record['pause_ms'] == [1000, 2000]
    assert not record['pause_edges_cross_anchor']


@pytest.mark.parametrize("gain", [.005, .13])
def test_quiet_or_loud_tonal_tail_must_not_be_cut_as_silence(gain):
    data = fixture()
    data[1][16000:32000] = gain * np.sin(np.arange(16000) * 2 * np.pi * 150 / 16000)
    run(data)
    assert data[0][2]["onset_ms"] == 2040
    assert not data[2]["full_signal_checks"][0]["accepted"]
    assert data[2]["full_signal_checks"][0]["reason"] == (
        "voiced-tail-in-waveform" if gain == .005 else "waveform-dsp-conflict")


@pytest.mark.parametrize("amplitude", [.3, .01])
def test_brief_spike_or_weak_unvoiced_burst_is_not_smoothed_into_the_cut(amplitude):
    data = fixture()
    data[1][1745 * 16:1755 * 16] = np.random.default_rng(7).normal(0, amplitude, 160)
    run(data)
    record = data[2]["full_signal_checks"][0]
    if not record["accepted"]:
        assert amplitude == .3 and record["reason"] == "waveform-dsp-conflict"
        assert data[0][2]["onset_ms"] == 2040
        return
    lo, hi = record["quiet_window_ms"]
    assert hi <= 1745 or lo >= 1755
    assert data[0][2]["onset_ms"] != 1750


def test_zero_recording_without_speech_context_does_not_invent_a_cut(monkeypatch):
    data = fixture()
    data[1][:] = 0
    monkeypatch.setattr(candidates, "_dsp_accepts_pause", lambda _: True)
    run(data)
    assert data[0][2]["onset_ms"] == 2040
    assert data[2]["full_signal_checks"][0]["reason"] == "missing-waveform-speech"


def test_cancellation_preserves_original_time():
    data = fixture()
    event = threading.Event()
    event.set()
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        run(data)
    assert data[0][2]["onset_ms"] == 2040


def test_periodicity_distinguishes_a_voiced_tail_from_noise_hum_and_silence():
    t = np.arange(640) / 16000
    assert full_signal._periodicity(.001 * np.sin(2 * np.pi * 150 * t), 16000) > .99
    assert full_signal._periodicity(.001 * np.sin(2 * np.pi * 50 * t), 16000) < .8
    assert full_signal._periodicity(np.random.default_rng(4).normal(0, .01, 640), 16000) < .2
    assert full_signal._periodicity(np.zeros(640), 16000) == 0
