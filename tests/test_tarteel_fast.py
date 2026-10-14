from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, align_fast, align_fast_whisper, candidates, model_store, pipeline
from surah_cutter.core.fast_reads import LocalWhisperReads
from surah_cutter.core.model_worker import Acoustic
from surah_cutter.data import quran_text


def fake_emitter(responses=None, model_id="whisper-tarteel-base"):
    calls = []
    responses = iter(responses or [])

    def read(pcm, *, total_samples):
        calls.append((len(pcm), total_samples))
        return {"text": next(responses, "noise"), "token_confidence": .95, "truncated": False}

    return SimpleNamespace(model_id=model_id, recognize_local=read), calls


def pause(start, end):
    return candidates.PauseDecision(candidates.Pause(start, end, end-start), True, None,
                                    True, "dsp", 0.0, 0.0, 0.0)


def test_whisper_budget_blocks_whole_sura_long_windows_and_sequential_coverage():
    emitter, calls = fake_emitter()
    reads = LocalWhisperReads(np.zeros(20 * 16000), 16000, emitter)
    assert reads.read(0, 20000, reason="forbidden") is None
    assert reads.read(0, 6001, reason="forbidden") is None
    assert calls == []
    for start in range(0, 20000, 2000):
        reads.read(start, start+2000, reason="test")
    assert reads.covered_ms <= 13000
    assert sum(n for n, _ in calls) <= 13 * 16000
    assert reads.denied[-1]["reason"] == "coverage_budget"


def test_whisper_exact_cache_and_repeated_reads_have_real_accounting():
    emitter, calls = fake_emitter()
    reads = LocalWhisperReads(np.zeros(20 * 16000), 16000, emitter)
    reads.read(2000, 4000, reason="first")
    reads.read(2000, 4000, reason="cached")
    reads.read(2500, 3500, reason="new_asr_not_a_slice")
    assert len(calls) == 2
    assert reads.audio_ms == 3000 and reads.covered_ms == 2000
    for start in range(50):
        reads.read(start, start+4000, reason="expansion")
    assert reads.audio_ms <= 40000
    assert reads.denied[-1]["reason"] == "inference_budget"
    assert reads.diagnostics()["model_passes"] == len(calls)


@pytest.mark.parametrize("samples,total", [(100000, 200000), (80000, 80000), (1000, 200000)])
@pytest.mark.parametrize("model_id", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_worker_local_action_rejects_nonlocal_input_before_model(model_id, samples, total):
    acoustic = Acoustic.__new__(Acoustic)
    acoustic.model_id = model_id
    with pytest.raises(ValueError, match="локальный"):
        acoustic.whisper_local(np.zeros(samples), total)


@pytest.mark.parametrize("model_id", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_tarteel_routes_to_pause_adapter_without_any_full_emissions_or_teacher_forcing(model_id, monkeypatch):
    from surah_cutter.core import align_tiled

    def forbidden(*args, **kwargs):
        pytest.fail("FAST called full/CTC/teacher-forced alignment")

    monkeypatch.setattr(model_store, "ACTIVE_ID", model_id)
    monkeypatch.setattr(align, "align_surah", forbidden)
    monkeypatch.setattr(align, "_emissions", forbidden)
    monkeypatch.setattr(align_tiled, "align_surah_tiled", forbidden)
    emitter, calls = fake_emitter(["قُلْ هُوَ", "اللَّهُ الصَّمَدُ", "لَمْ يَلِدْ", "وَلَمْ يَكُنْ"], model_id)
    emitter.align_words = forbidden
    bundle = SimpleNamespace(emitter=emitter, text_kind="whisper")
    decisions = [pause(s, s+500) for s in (2500, 6500, 10500, 14500)]
    trace = {}
    rows = align_fast.align_surah_fast(np.zeros(25 * 16000), 16000, quran_text.surah_words(112),
                                       bundle=bundle, trace=trace, pause_decisions=decisions)
    assert [a for a, _, _ in rows] == [1, 2, 3, 4]
    assert [ms for _, ms, _ in rows] == [2750, 6750, 10750, 14750]
    assert all(c > 0 for _, _, c in rows)
    assert len(calls) == 4
    assert trace["review_required_ayahs"] == []
    assert trace["model_source"].startswith("local-asr-pause:")


@pytest.mark.parametrize("model_id", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_no_pauses_or_unrelated_phrase_returns_explicit_review_without_expansion(model_id, monkeypatch):
    monkeypatch.setattr(model_store, "ACTIVE_ID", model_id)
    emitter, calls = fake_emitter(model_id=model_id)
    trace = {}
    rows = align_fast_whisper.align_surah_fast_whisper(
        np.zeros(20 * 16000), 16000, quran_text.surah_words(112),
        bundle=SimpleNamespace(emitter=emitter), trace=trace, pause_decisions=[pause(2000, 2500)],
    )
    assert len(calls) == 1
    assert all(c == 0 for _, _, c in rows)
    assert trace["review_required_ayahs"] == [1, 2, 3, 4]
    assert all(m["boundary_evidence"] == "unresolved_local" for m in trace["pause_matches"])
    assert trace["bounded_ctc_fallback"] is None


@pytest.mark.parametrize("model_id", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_short_prefix_expands_only_for_known_ambiguity_and_keeps_unknown_for_review(model_id, monkeypatch):
    monkeypatch.setattr(model_store, "ACTIVE_ID", model_id)
    emitter, calls = fake_emitter(["اللَّهُ", "اللَّهُ الصَّمَدُ"], model_id)
    trace = {}
    align_fast_whisper.align_surah_fast_whisper(
        np.zeros(20 * 16000), 16000, quran_text.surah_words(112),
        bundle=SimpleNamespace(emitter=emitter), trace=trace, pause_decisions=[pause(6500, 7000)],
    )
    assert [n for n, _ in calls] == [1500 * 16, 2700 * 16]
    assert trace["model_reads"][1]["reason"] in ("ambiguous_prefix", "incomplete_prefix")
    assert trace["review_required_ayahs"] == [1, 3, 4]


def test_repeated_internal_opening_and_truncated_asr_are_never_confident():
    result = {"text": "مِنْ شَرِّ", "token_confidence": .999, "truncated": False}
    phrases = {1: {"tokens": ["من", "شر"], "ambiguous": True}}
    assert not align_fast_whisper._rank(result, phrases)["confirmed"]
    phrases[1]["ambiguous"] = False
    assert not align_fast_whisper._rank({**result, "truncated": True}, phrases)["confirmed"]


@pytest.mark.parametrize("model_id", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_tarteel_fast_pipeline_preserves_requested_mode_and_blocks_auto_fallback(model_id, monkeypatch):
    monkeypatch.setattr(model_store, "ACTIVE_ID", model_id)
    seen = []
    monkeypatch.setattr(pipeline, "_fast_align", lambda *a, **kw: seen.append("fast") or [(112, 1, 100)])
    monkeypatch.setattr(pipeline, "_full_align", lambda *a, **kw: pytest.fail("hidden FULL"))
    assert pipeline.process_surah("unused", 112, mode="fast") == [(112, 1, 100)]
    assert seen == ["fast"]
    with pytest.raises(ValueError, match="Выбранная модель"):
        pipeline.process_surah("unused", 112, mode="auto")
    assert model_store.supports_fast() and not model_store.supports_fast("whisper-large-v3")


@pytest.mark.parametrize("model_id", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_api_settings_accept_tarteel_fast_and_keep_unsupported_models_blocked(model_id, tmp_path, monkeypatch):
    from fastapi import HTTPException

    from surah_cutter.webgui import server

    monkeypatch.setattr(server.settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(model_store, "ACTIVE_ID", model_id)
    endpoint = next(r.endpoint for r in server.app.routes if r.path == "/api/settings")
    endpoint({"mode": "fast"})
    assert server.settings.load()["mode"] == "fast"
    monkeypatch.setattr(model_store, "ACTIVE_ID", "whisper-large-v3")
    with pytest.raises(HTTPException):
        endpoint({"mode": "fast"})
    assert server.settings.load()["mode"] == "fast"
