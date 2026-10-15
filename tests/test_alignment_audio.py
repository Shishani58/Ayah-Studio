"""Opt-in regressions against actual Shatri audio and independent timings.

SC_TEST_AUDIO=1 .venv/bin/pytest -q tests/test_alignment_audio.py
"""

import os
from pathlib import Path

import pytest

from surah_cutter.core import align, align_fast, decode, emissions
from surah_cutter.data import quran_text, references
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
        # The reference grades the former speech-edge placement. Keep that
        # regression bound on the acoustic result, then check the new cut
        # independently: it intentionally differs from the reference in silence.
        checks = {c["ayah"]: c for c in trace["ayahs"]}
        assert max(abs(checks[a]["pre_center_onset_ms"] - reference[(surah, a)])
                   for a, _, _ in rows) <= max_error_ms
        centered = [c for c in checks.values() if c.get("pause_center_reason") == "positioned"]
        assert centered
        for check in centered:
            start, end = check["center_pause_start_ms"], check["center_pause_end_ms"]
            assert check["onset_ms"] == round(end - min((end - start) / 2, 250))
            assert start < check["onset_ms"] < end
        assert trace["emission_passes"] == 1
        if surah == 1:
            first_word = next(c for c in trace["ayahs"] if c["ayah"] == 2)
            assert first_word["reason"] == "split-word-local-recheck"
            assert first_word["raw_onset_ms"] != first_word["onset_ms"]
            assert len(trace["full_boundary_reads"]) == 1
            assert trace["full_boundary_reads"][0]["accepted"]
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
