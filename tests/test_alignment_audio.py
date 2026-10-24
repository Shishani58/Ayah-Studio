"""Opt-in regressions against actual Shatri audio and independent timings.

SC_TEST_AUDIO=1 .venv/bin/pytest -q tests/test_alignment_audio.py
"""

import os
from pathlib import Path

import pytest

from surah_cutter.core import align, align_fast, decode, emissions
from surah_cutter.data import quran_text
from tools.research import references
from tools.bench_fast import pause_contract_violations

AUDIO = Path.home() / "Desktop/Abu_Bakr_Ash-Shatri"
MODEL = Path(__file__).resolve().parents[1] / "models/wav2vec2-quran-phonetics.onnx"
pytestmark = pytest.mark.skipif(
    os.environ.get("SC_TEST_AUDIO") != "1" or not MODEL.exists() or not AUDIO.is_dir(),
    reason="Requires opt-in local Shatri audio and acoustic model",
)


@pytest.fixture(scope="module")
def bundle():
    # Pin the audited model; the app's selected model may change independently.
    return align._OnnxBundle(emissions.load_emitter(MODEL))


@pytest.mark.parametrize("surah, mode, max_error_ms", [
    (1, "full", 100),
    (112, "full", 150), (114, "full", 150),
    (112, "fast", 150), (114, "fast", 150),
    (80, "fast", 800), (88, "fast", 500),
    (86, "fast", 150),
])
def test_actual_alignment_starts_and_sequence(bundle, surah, mode, max_error_ms, monkeypatch):
    pcm = decode.decode(str(AUDIO / f"{surah:03d}.mp3"))
    words = quran_text.surah_words(surah)
    trace = {}
    actual_calls = []
    original = align._emissions

    def local_infer(bundle, audio, np_module):
        if mode == "fast":
            assert len(audio) < len(pcm)
            assert len(audio) <= 6 * 16000
            actual_calls.append(len(audio))
        return original(bundle, audio, np_module)

    monkeypatch.setattr(align, "_emissions", local_infer)
    if mode == "fast":
        from surah_cutter.core import align_tiled

        def forbidden(*args, **kwargs):
            pytest.fail("FAST escalated to FULL/tiled")

        monkeypatch.setattr(align, "align_surah", forbidden)
        monkeypatch.setattr(align_tiled, "align_surah_tiled", forbidden)
    if mode == "full":
        rows = align.align_surah(pcm, 16000, words, surah=surah, model=bundle, trace=trace)
    else:
        rows = align_fast.align_surah_fast(pcm, 16000, words, surah=surah, bundle=bundle, trace=trace)
    reference = references.load_reference("abu_bakr_ash_shatri")

    assert [a for a, _, _ in rows] == list(range(1, quran_text.ayah_count(surah) + 1))
    assert all(left[1] < right[1] for left, right in zip(rows, rows[1:]))
    if mode == "full":
        # Historical bounds used a fitted -200ms correction, now removed.
        # The triangle bound adds exactly that removed offset; it does not
        # retune the runtime to this evaluation DB. Cuts are checked separately.
        checks = {c["ayah"]: c for c in trace["ayahs"]}
        assert max(abs(checks[a]["pre_center_onset_ms"] - reference[(surah, a)])
                   for a, _, _ in rows) <= max_error_ms + 200
        centered = [c for c in checks.values() if c.get("pause_center_reason") == "positioned"]
        assert centered
        for check in centered:
            start, end = check["center_pause_start_ms"], check["center_pause_end_ms"]
            assert check["onset_ms"] == round(end - min((end - start) / 2, 250))
            assert start < check["onset_ms"] < end
        assert trace["emission_passes"] == 1
        if surah == 1:
            third = checks[3]
            assert third["pause_center_reason"] == "unstable-pause"
            # With the fitted lag removed, both shelves precede the raw
            # onset. Neither v9 nor v10 can choose between them safely.
            assert third["reason"] == "weak-word"
            assert third["onset_ms"] == third["pre_center_onset_ms"]
            shelves = {(p["start_ms"], p["end_ms"])
                       for key in ("pause_local_decisions", "pause_stability_decisions")
                       for p in third[key] if p["accepted"] and p["owned"]}
            assert len(shelves) == 2
            signal = {r["ayah"]: r for r in trace["full_signal_checks"]}
            assert 3 not in signal
            for a in (5, 7):
                assert not signal[a]["accepted"]
                assert signal[a]["reason"] == "voiced-tail-in-waveform"
                assert checks[a]["onset_ms"] == checks[a]["pre_center_onset_ms"]
            assert checks[7]["onset_ms"] == checks[7]["pre_center_onset_ms"]
            assert not third["reliable"]
            first_word = next(c for c in trace["ayahs"] if c["ayah"] == 2)
            assert first_word["reason"] == "split-word-local-recheck"
            assert first_word["raw_onset_ms"] != first_word["onset_ms"]
            phrase_reads = trace["full_boundary_reads"]
            assert len(phrase_reads) == 1 and phrase_reads[0]["accepted"]
            assert "full_transition_verifier" not in trace
    else:
        # A local FAST can explicitly decline proof. Large residual errors may
        # never be advertised as reliable merely to match former full-pass tests.
        for a, ms, confidence in rows:
            if abs(ms - reference[(surah, a)]) > max_error_ms:
                assert confidence == 0.0
                assert a in trace["review_required_ayahs"]
        assert trace["emission_passes"] == len(actual_calls)
        assert trace["audio_ms"] == pytest.approx(sum(actual_calls) / 16)
        assert trace["coverage"] <= 0.65
        assert trace["bounded_ctc_fallback"] is None
        assert "full_ctc_fallback" not in trace
        assert pause_contract_violations(rows, trace) == []


def test_actual_full_late_cut_has_both_owners_and_a_quiet_interval(bundle):
    pcm = decode.decode(str(AUDIO / "080.mp3"))
    trace = {}
    rows = align.align_surah(pcm, 16000, quran_text.surah_words(80), surah=80, model=bundle, trace=trace)
    read = next(r for r in trace["full_signal_checks"] if r["ayah"] == 35)
    assert read["accepted"] and read["reason"] == "waveform-pause-recheck"
    assert read["before_ms"] == next(c for c in trace["ayahs"] if c["ayah"] == 35)["pre_center_onset_ms"]
    assert read["pause_ms"][0] < read["after_ms"] < read["pause_ms"][1]
    assert dict((a, ms) for a, ms, _ in rows)[35] == read["after_ms"]
    left, right = [next(c for c in trace["ayahs"] if c["ayah"] == a) for a in (34, 35)]
    assert left["full_ctc_end_ms"] < read["safe_interval_ms"][0]
    assert right["full_ctc_first_ms"] > read["safe_interval_ms"][1]
    assert not trace.get("full_boundary_reads")
    assert "full_transition_verifier" not in trace
    check = next(c for c in trace["ayahs"] if c["ayah"] == 35)
    assert check["raw_onset_ms"] == read["before_ms"]
    assert check["repaired"] and not check["reliable"]


def test_actual_bounded_full_runs_the_same_waveform_check(bundle, monkeypatch):
    pcm = decode.decode(str(AUDIO / "001.mp3"))
    monkeypatch.setattr(align, "_full_ctc_requires_tiling", lambda *args: True)
    trace = {}
    rows = align.align_surah(pcm, 16000, quran_text.surah_words(1), surah=1, model=bundle, trace=trace)
    assert trace["n_chunks"] >= 1
    assert [a for a, _, _ in rows] == list(range(1, 8))
    assert all(a[1] < b[1] for a, b in zip(rows, rows[1:]))
    signal = {r["ayah"]: r for r in trace["full_signal_checks"]}
    checks = {c["ayah"]: c for c in trace["ayahs"]}
    assert trace["full_signal_policy"]["quiet_window_ms"] == 40
    assert 3 not in signal
    for a in (5, 7):
        assert not signal[a]["accepted"] and signal[a]["reason"] == "voiced-tail-in-waveform"
    assert checks[3]["pause_center_reason"] == "unstable-pause"
    assert not checks[3]["reliable"]
    for a in (3, 7):
        assert checks[a]["onset_ms"] == checks[a]["pre_center_onset_ms"]
    assert len(trace["full_boundary_reads"]) == 1
    assert "full_transition_verifier" not in trace
