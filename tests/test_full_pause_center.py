from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, candidates, full_pause_center


def fixture():
    checks = {
        1: dict(ayah=1, onset_ms=100, conf=.9, first_token_conf=.9, reliable=True, token_confs=[.9],
                full_ctc_first_ms=100, full_ctc_end_ms=900, full_ctc_tail_conf=.9),
        2: dict(ayah=2, onset_ms=2040, conf=.9, first_token_conf=.9, reliable=True, token_confs=[.9],
                full_ctc_first_ms=2040, full_ctc_end_ms=2800, full_ctc_tail_conf=.9),
    }
    pcm = np.random.default_rng(12).normal(0, .1, 64_000).astype(np.float32)
    pcm[16_000:32_000] = 0
    return checks, pcm


def test_centers_only_interayah_pause_and_preserves_confidence_and_raw_evidence():
    checks, pcm = fixture()
    before = deepcopy(checks)
    full_pause_center.position_boundaries(checks, pcm, 16000)
    assert checks[1]["onset_ms"] == 100
    assert checks[2]["onset_ms"] == pytest.approx(1750, abs=30)
    assert checks[2]["pre_center_onset_ms"] == 2040
    assert checks[2]["center_pause_start_ms"] == pytest.approx(1000, abs=30)
    assert checks[2]["center_pause_end_ms"] == pytest.approx(2000, abs=30)
    assert checks[2]["pause_center_reason"] == "positioned"
    for a in checks:
        assert checks[a]["conf"] == before[a]["conf"]
        assert checks[a]["token_confs"] == before[a]["token_confs"]
        assert checks[a]["reliable"]


@pytest.mark.parametrize("case", ["intra-previous", "intra-next", "weak-tail", "weak-head",
                                   "no-pause", "ambiguous", "order", "tonal"])
def test_uncertain_or_wrong_pause_never_moves_boundary(case, monkeypatch):
    checks, pcm = fixture()
    if case == "intra-previous":
        checks[1]["full_ctc_end_ms"] = 2100
    elif case == "intra-next":
        checks[2]["full_ctc_first_ms"] = 900
    elif case == "weak-tail":
        checks[1]["full_ctc_tail_conf"] = .1
    elif case == "weak-head":
        checks[2]["first_token_conf"] = .1
    elif case == "no-pause":
        pcm = np.random.default_rng(12).normal(0, .1, len(pcm)).astype(np.float32)
    elif case == "ambiguous":
        monkeypatch.setattr(candidates, "detect_energy_pauses", lambda *a, **kw: [
            candidates.Pause(900, 1900, 1000), candidates.Pause(950, 1850, 900)])
    elif case == "order":
        checks[1]["onset_ms"] = 1900
    elif case == "tonal":
        pcm[16000:32000] = .13 * np.sin(np.arange(16000) * 2 * np.pi * 150 / 16000)
    full_pause_center.position_boundaries(checks, pcm, 16000)
    assert checks[2]["onset_ms"] == 2040
    assert checks[2]["pause_center_reason"] != "positioned"
    if case in ("intra-previous", "ambiguous", "order"):
        assert not checks[2]["reliable"]


@pytest.mark.parametrize("duration", [180, 400, 600, 2000])
@pytest.mark.parametrize("gain", [.1, 1, 4])
def test_local_silence_short_midpoint_long_capped_lead_without_near_tail_gate(duration, gain):
    checks, _ = fixture()
    start, end = 2000, 2000 + duration
    pcm = np.random.default_rng(4).normal(0, .1, (end + 1200) * 16).astype(np.float32)
    pcm[start * 16:end * 16] *= .001
    pcm *= gain
    checks[1]["full_ctc_end_ms"] = 1300  # 700 ms before silence, not limited to 300 ms.
    checks[2].update(onset_ms=end - 100, full_ctc_first_ms=end + 60)
    full_pause_center.position_boundaries(checks, pcm, 16000)
    assert checks[2]["pause_center_reason"] == "positioned"
    assert checks[2]["onset_ms"] == pytest.approx(end - min(duration / 2, 250), abs=30)
    assert checks[2]["pause_lead_ms"] <= 250


def test_first_ayah_uses_intro_speech_not_its_delimiter():
    checks = {1: dict(ayah=1, onset_ms=2040, conf=.9, first_token_conf=.9, reliable=True)}
    # Last intro vowel is silent; the delimiter's spurious late span is not speech.
    scores = np.zeros(150)
    scores[46:51] = -12
    full_pause_center.record_edges(checks, [SimpleNamespace(ayah=1)], [(3, 4)],
                                  [(40, 44), (46, 50), (90, 95), (102, 109)], scores, 20,
                                  intro=True, targets=[1, 2, 0, 3], delim_id=0)
    assert checks[1]["full_intro_end_ms"] == 900
    _, pcm = fixture()
    full_pause_center.position_boundaries(checks, pcm, 16000)
    assert checks[1]["onset_ms"] == pytest.approx(1750, abs=30)
    assert checks[1]["pause_center_reason"] == "positioned"


def test_records_last_word_end_and_absolute_tiled_offset():
    checks = {1: {}, 2: {}}
    words = [SimpleNamespace(ayah=a) for a in [1, 1, 2]]
    full_pause_center.record_edges(checks, words, [(0, 1), (1, 2), (2, 3)],
                                  [(5, 8), (10, 12), (20, 25)], np.zeros(30), 20,
                                  base_ms=5000)
    assert checks[1] == dict(full_ctc_first_ms=5100, full_ctc_end_ms=5260, full_ctc_tail_conf=1.)
    assert checks[2]["full_ctc_first_ms"] == 5400


def test_waqf_uses_supported_tail_before_silent_final_vowel():
    checks = {1: {}}
    scores = np.zeros(30)
    scores[20:26] = -12
    full_pause_center.record_edges(checks, [SimpleNamespace(ayah=1)], [(0, 2)],
                                  [(10, 12), (20, 25)], scores, 20)
    assert checks[1]["full_ctc_end_ms"] == 260
    assert checks[1]["full_ctc_tail_conf"] == 1.


def test_centering_preserves_weak_word_review():
    checks, pcm = fixture()
    checks[2].update(reliable=False, conf=.1, reason="weak-word", token_confs=[.9, .01])
    full_pause_center.position_boundaries(checks, pcm, 16000)
    assert checks[2]["onset_ms"] == pytest.approx(1750, abs=30)
    assert checks[2]["conf"] == .1
    assert not checks[2]["reliable"]
    assert checks[2]["reason"] == "weak-word"


def test_local_recheck_is_not_downgraded_using_its_discarded_ctc_spans():
    checks, pcm = fixture()
    checks[2].update(repaired=True, full_ctc_first_ms=900, reason="split-word-local-recheck")
    full_pause_center.position_boundaries(checks, pcm, 16000)
    assert checks[2]["onset_ms"] == 2040
    assert checks[2]["reliable"]
    assert checks[2]["pause_center_reason"] == "existing-recheck"


def test_intentional_capped_lead_does_not_trigger_old_late_midpoint_warning():
    checks, pcm = fixture()
    checks[2]["reason"] = "ctc"
    full_pause_center.position_boundaries(checks, pcm, 16000)
    windows = {"pause_decisions": [dict(accepted=True, duration_ms=1000,
                                       start_ms=1000, end_ms=2000, center_ms=1500)]}
    align._review_pause_full_onsets(checks.values(), windows)
    assert checks[2]["onset_ms"] == pytest.approx(1750, abs=30)
    assert checks[2]["reliable"]


@pytest.mark.parametrize("bounded", [False, True])
def test_full_pipeline_centers_after_alignment(monkeypatch, bounded):
    from surah_cutter.core.emissions import OnnxEmitter
    from surah_cutter.data import reference_prior, transliteration

    _, pcm = fixture()
    words = [SimpleNamespace(surah=1, ayah=a, text="ا", index=a - 1) for a in [1, 2]]
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input_values")],
                              get_outputs=lambda: [SimpleNamespace(name="logits")])
    model = align._OnnxBundle(OnnxEmitter(session, {"[PAD]": 0, "a": 1, "b": 2},
                                        blank_id=0, do_normalize=False))
    model.lag_ms = 0
    matrix = np.full((200, 3), -20., dtype=np.float32)
    matrix[:, 0] = 0
    matrix[40:45, 0], matrix[40:45, 1] = -20, 0
    matrix[102:110, 0], matrix[102:110, 2] = -20, 0

    def emissions(bundle, audio, np, *, trace=None):
        if trace is not None:
            trace["full_windows"] = {}
        return matrix

    monkeypatch.setattr(align, "_full_emissions", emissions)
    monkeypatch.setattr(align, "_full_ctc_requires_tiling", lambda *a: bounded)
    monkeypatch.setattr(align, "_build_targets", lambda *a: ([1, 2], [(0, 1), (1, 2)]))
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda *a: ["a", "b"])
    monkeypatch.setattr(reference_prior, "surah_start_fractions", lambda *a: {})
    trace = {}
    rows = align.align_surah(pcm, 16000, words, model=model, trace=trace)
    assert rows[1][1] == pytest.approx(1750, abs=30)
    assert trace["ayahs"][1]["pre_center_onset_ms"] == 2040
    assert trace["ayahs"][1]["pause_center_reason"] == "positioned"
