from copy import deepcopy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, candidates, full_transition
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope
from surah_cutter.data import transliteration


def fixture(monkeypatch, *, tail="df"):
    from surah_cutter.core import full_transition_verifier

    monkeypatch.setattr(full_transition_verifier, "_installed_verifier", lambda: None)
    trs = ["bc", tail, "gh", "jk"]
    vocab = {c: i + 1 for i, c in enumerate(dict.fromkeys("".join(trs) + "|"))}
    words = [SimpleNamespace(surah=1, ayah=1 + i // 2, index=i, text=trs[i]) for i in range(4)]
    bundle = SimpleNamespace(vocab=vocab, blank_id=0, delim_id=vocab["|"], lag_ms=200)
    targets, ws = align._build_targets(words, trs, bundle)
    # Actual local crop is 600–6400 ms, on the native 20 ms CTC grid.
    matrix = np.full((290, len(vocab) + 1), -12., np.float32)
    matrix[:, 0] = 0
    times = [[1200, 1400], [3000, 3280], [4300, 4500], [5000, 5200]]
    for (ts, te), offsets in zip(ws, times):
        for token, time_ms in zip(targets[ts:te], offsets):
            f = (time_ms - 600) // 20
            matrix[f, 0], matrix[f, token] = -12, 0
        if te < len(targets):
            f = (offsets[-1] + 100 - 600) // 20
            matrix[f, 0], matrix[f, targets[te]] = -12, 0
    calls = []

    def emit(pcm):
        calls.append(len(pcm))
        return matrix

    bundle.emitter = SimpleNamespace(emissions=emit)
    pcm = np.random.default_rng(4).normal(0, .1, 112000).astype(np.float32)
    pcm[3600 * 16:4000 * 16] = 0
    checks = {
        1: dict(ayah=1, onset_ms=500, raw_onset_ms=500, conf=.9, first_token_conf=.9,
                reliable=True, repaired=False, full_ctc_first_ms=700,
                full_ctc_end_ms=3300, full_ctc_tail_conf=.9, full_ctc_tail_start_ms=1000),
        2: dict(ayah=2, onset_ms=4100, raw_onset_ms=4100, conf=.9, first_token_conf=.9,
                reliable=False, repaired=False, full_ctc_first_ms=4300,
                full_ctc_head_end_ms=6000, pause_center_reason="unstable-pause",
                pause_local_crop_ms=[2400, 5300], pause_stability_crop_ms=[1600, 6100],
                pause_local_decisions=[dict(start_ms=4100, end_ms=4250, accepted=True, owned=True)],
                pause_stability_decisions=[dict(start_ms=3600, end_ms=4000, accepted=True, owned=True)]),
    }
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: trs)
    return bundle, pcm, words, checks, {}, calls, matrix


def run(data):
    b, pcm, words, checks, trace, *_ = data
    full_transition.refine_late_boundaries(b, pcm, 16000, words, checks, trace=trace)


def test_two_sided_evidence_moves_late_cut_into_quiet_without_promoting_confidence(monkeypatch):
    data = fixture(monkeypatch)
    before = deepcopy(data[3])
    run(data)
    check = data[3][2]
    assert check["onset_ms"] == 3800
    assert check["raw_onset_ms"] == 4100
    assert check["conf"] == .9 and check["first_token_conf"] == .9
    assert check["repaired"] and not check["reliable"]
    assert data[3][1] == before[1]
    record = data[4]["full_boundary_reads"][0]
    assert record["accepted"] and record["outcome"] == "completed"
    assert record["before_ms"] == 4100 and record["after_ms"] == 3800
    assert record["safe_interval_ms"] == [3750, 3920]
    assert record["local_checks"][0]["full_ctc_end_ms"] == 3300
    assert record["local_checks"][1]["full_ctc_first_ms"] == 4300
    assert data[5] == [5800 * 16]


def test_supported_vowel_ending_is_speech_evidence_without_inventing_a_consonant(monkeypatch):
    data = fixture(monkeypatch, tail="āī")
    run(data)
    read = data[4]["full_boundary_reads"][0]
    assert read["accepted"]
    assert read["edge_sounds"][1] == 2
    assert read["distinct_consonants"][1] == 2
    assert data[3][2]["onset_ms"] == 3800


def test_long_pause_keeps_at_most_250ms_lead_before_speech(monkeypatch):
    data = fixture(monkeypatch)
    data[1][2000 * 16:4000 * 16] = 0
    data[3][1]["full_ctc_end_ms"] = 1800
    check = data[3][2]
    check["pause_stability_decisions"][0]["start_ms"] = 2000
    check["pause_local_crop_ms"][0] = 1400
    check["pause_stability_crop_ms"][0] = 600
    for old, new in ((3000, 1600), (3280, 1780), (3380, 1880)):
        f, g = (old - 600) // 20, (new - 600) // 20
        data[6][g] = data[6][f].copy()
        data[6][f] = -12
        data[6][f, 0] = 0
    run(data)
    assert check["onset_ms"] == 3750
    assert check["repaired"]
    assert 4000 - check["onset_ms"] <= 250


@pytest.mark.parametrize("case", ["weak-head", "weak-tail", "repaired", "positioned",
                                  "intraword", "already-safe", "ambiguous", "too-long",
                                  "missing-context", "no-previous", "tonal", "no-silence"])
def test_no_read_for_unsupported_or_unnecessary_transition(case, monkeypatch):
    data = fixture(monkeypatch)
    checks = data[3]
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
        check["pause_stability_decisions"][0].update(start_ms=4140, end_ms=4280)
    elif case == "already-safe":
        check["onset_ms"] = 3800
    elif case == "ambiguous":
        check["pause_stability_decisions"].append(dict(start_ms=3500, end_ms=3950, owned=True, accepted=True))
    elif case == "too-long":
        checks[1]["full_ctc_tail_start_ms"] = -10000
        check["full_ctc_head_end_ms"] = 50000
        # Length validation uses the actual file duration, never a fake bound.
        data = (data[0], np.tile(data[1], 3), *data[2:])
    elif case == "missing-context":
        del check["pause_stability_crop_ms"]
    elif case == "no-previous":
        del checks[1]
    elif case == "tonal":
        data[1][3600 * 16:4000 * 16] = .1 * np.sin(np.arange(6400) * 2 * np.pi * 150 / 16000)
    elif case == "no-silence":
        data[1][3600 * 16:4000 * 16] = np.random.default_rng(5).normal(0, .1, 6400)
    old = check["onset_ms"]
    run(data)
    assert check["onset_ms"] == old
    assert data[5] == []


@pytest.mark.parametrize("case", ["empty-decode", "weak-local-head", "tail-disagreement",
                                  "head-disagreement", "previous-order"])
def test_local_conflict_preserves_cut_and_review(case, monkeypatch):
    data = fixture(monkeypatch)
    matrix = data[6]
    if case == "empty-decode":
        matrix[:] = -12
        matrix[:, 0] = 0
    elif case == "weak-local-head":
        matrix[(4300 - 600) // 20, data[0].vocab["g"]] = -4
        matrix[(4300 - 600) // 20, 0] = 0
    elif case == "tail-disagreement":
        data[3][1]["full_ctc_end_ms"] = 3000
    elif case == "head-disagreement":
        data[3][2]["full_ctc_first_ms"] = 4600
    elif case == "previous-order":
        data[3][1]["onset_ms"] = 3900
    run(data)
    assert data[3][2]["onset_ms"] == 4100
    assert not data[3][2]["reliable"]
    assert not data[4]["full_boundary_reads"][0]["accepted"]


def test_dsp_must_accept_every_context(monkeypatch):
    data = fixture(monkeypatch)
    calls = []

    def gate(metrics):
        calls.append(metrics)
        return len(calls) != 2

    monkeypatch.setattr(candidates, "_dsp_accepts_pause", gate)
    run(data)
    assert not data[5] and data[3][2]["onset_ms"] == 4100


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_cancelled_inference_is_recorded_without_moving_cut(cancel, monkeypatch):
    data = fixture(monkeypatch)
    event = threading.Event()

    def emit(_):
        if cancel:
            event.set()
            return data[6]
        raise RuntimeError("test inference failure")

    data[0].emitter.emissions = emit
    error = GenerationCancelled if cancel else RuntimeError
    with cancellation_scope(event), pytest.raises(error):
        run(data)
    record = data[4]["full_boundary_reads"][0]
    assert record["outcome"] == ("cancelled" if cancel else "failed")
    assert not record["accepted"] and data[3][2]["onset_ms"] == 4100


def test_cancel_before_dispatch_does_not_read_audio(monkeypatch):
    data = fixture(monkeypatch)
    event = threading.Event()
    event.set()
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        run(data)
    assert not data[5]
    assert not data[4]["full_boundary_reads"]


@pytest.mark.parametrize("conflict", [None, "local-head", "independent-head", "independent-tail"])
def test_independent_proof_resolves_global_tail_only_with_local_agreement(conflict, monkeypatch):
    from surah_cutter.core import full_transition_verifier

    data = fixture(monkeypatch)
    data[3][1]["full_ctc_end_ms"] = 3000  # primary tail disagrees with local 3300
    if conflict == "local-head":
        data[6][(4300 - 600) // 20, data[0].vocab["g"]] = -4
        data[6][(4300 - 600) // 20, 0] = 0

    def prove(*args, reads, **kwargs):
        record = dict(ayah=2, role="two-sided-verifier", outcome="completed", accepted=False)
        reads.append(record)
        return dict(tail_ms=3650 if conflict == "independent-tail" else 3320,
                    first_ms=4700 if conflict == "independent-head" else 4320,
                    onset_ms=4320, record=record, source="muaalem@test")

    monkeypatch.setattr(full_transition_verifier, "verify_transition", prove)
    run(data)
    check = data[3][2]
    record = data[4]["full_boundary_reads"][-1]
    if conflict is None:
        assert check["onset_ms"] == 3800
        assert check["reason"] == "two-sided-muaalem-recheck"
        assert check["boundary_verifier"] == "muaalem@test"
        assert check["conf"] == .9 and not check["reliable"]
        assert record["accepted"]
    else:
        assert check["onset_ms"] == 4100
        assert not record["accepted"]
        assert record["reason"] == "conflicting-independent-transition"
