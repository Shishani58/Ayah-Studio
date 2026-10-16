"""Reject invented MFA word ownership and keep uncalibrated results in review."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, mfa_runtime, model_store, pipeline, word_align
from surah_cutter.data.quran_text import surah_words


def output(entries):
    return {"tiers": {"words": {"entries": entries}}}


def test_mfa_output_requires_exact_words_and_ignores_only_silence():
    data = output([[0, .1, "<eps>"], [.1, .3, "قُلْ"], [.3, .5, "هُوَ"], [.5, .6, ""]])
    assert mfa_runtime.parse_word_intervals(data, ["قُلْ", "هُوَ"], 600) == [100, 300]
    with pytest.raises(ValueError, match="каноническими"):
        mfa_runtime.parse_word_intervals(data, ["قُلْ", "<unk>"], 600)


@pytest.mark.parametrize("entries", [
    [[0, .2, "<unk>"]], [], [[.3, .2, "قُلْ"]], [[0, 1.1, "قُلْ"]],
    [[float("nan"), .2, "قُلْ"]], [[0, .5, "قُلْ"], [.4, .6, "هُوَ"]],
])
def test_mfa_invalid_or_incomplete_output_never_becomes_timings(entries):
    expected = ["قُلْ", "هُوَ"] if len(entries) == 2 else ["قُلْ"]
    with pytest.raises(ValueError):
        mfa_runtime.parse_word_intervals(output(entries), expected, 1000)


def test_mfa_nfkc_lookup_and_multi_unit_word_keep_canonical_owners(tmp_path, monkeypatch):
    from surah_cutter.core.model_runtime import inference_words

    words = [w for w in surah_words(37) if w.ayah == 130]
    originals = [w.text for w in words]
    units = [u for text in inference_words(words, "muaalem") for u in text.split()]
    (tmp_path / "quran_hafs.dict").write_text("\n".join(u + "\tق ُ ل" for u in units))
    monkeypatch.setattr(mfa_runtime, "executable", lambda: tmp_path / "mfa")

    def fake_run(command, work):
        transcript = (work / "audio.lab").read_text().split()
        lexicon = {line.split()[0] for line in (work / "lookup.dict").read_text().splitlines()}
        assert all(u in lexicon for u in transcript)
        (work / "alignment.json").write_text(json.dumps(output([
            [i * .1, (i + 1) * .1, u] for i, u in enumerate(transcript)])))

    monkeypatch.setattr(mfa_runtime, "_run", fake_run)
    emitter = mfa_runtime.MfaEmitter(tmp_path)
    rows = emitter.align_words(np.ones(16000), words)
    assert [r[0] for r in rows] == [w.index for w in words]
    assert [r[1] for r in rows] == [0, 100, 200]  # Last canonical word has two units.
    assert all(r[2] == 0 for r in rows)
    assert [w.text for w in words] == originals


def test_mfa_full_and_words_use_native_alignment_without_ctc(monkeypatch):
    words = surah_words(112)
    emitter = SimpleNamespace(vocab={"unused": 0}, blank_id=0, text_kind="mfa",
                              align_words=lambda pcm, ws, **kw: [(w.index, i * 100, 0.) for i, w in enumerate(ws)])
    bundle = align._OnnxBundle(emitter)
    monkeypatch.setattr(align, "_get_onnx_bundle", lambda: bundle)
    monkeypatch.setattr(align, "bundle_source", lambda _: "mfa:test")
    monkeypatch.setattr(align, "_emissions", lambda *a, **kw: pytest.fail("MFA must not enter CTC"))
    trace = {}
    rows = align.align_surah(np.ones(32000), 16000, words, model=bundle, trace=trace)
    assert [row[0] for row in rows] == [1, 2, 3, 4]
    assert trace["backend"] == "mfa"
    assert all(not a["reliable"] for a in trace["ayahs"])
    assert len(word_align.align_ayah_words(np.ones(16000), 16000, words[:4])) == 4


@pytest.mark.parametrize("model", ["mfa-quran-hafs", "fastconformer-muno", "zipformer-quran"])
def test_new_models_cannot_enter_fast_or_decode_before_rejection(model, monkeypatch):
    monkeypatch.setattr(model_store, "ACTIVE_ID", model)
    with pytest.raises(ValueError, match="Полный"):
        pipeline.process_surah("does-not-exist.wav", 112, mode="fast")


def test_mfa_tiled_is_rejected_before_audio_decode(monkeypatch):
    monkeypatch.setattr(model_store, "ACTIVE_ID", "mfa-quran-hafs")
    with pytest.raises(ValueError, match="tiled"):
        pipeline.process_surah("does-not-exist.wav", 112, mode="tiled")


def test_gated_download_reuses_local_login_without_returning_token(tmp_path, monkeypatch):
    import threading

    token = "hf_local_secret"
    path = tmp_path / "token"
    path.write_text(token)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN_PATH", str(path))
    monkeypatch.setattr(model_store, "ROOT", tmp_path / "models")
    monkeypatch.setattr(model_store, "_JOBS", {})
    monkeypatch.setattr(model_store, "_CANCEL", {})
    captured = []
    monkeypatch.setattr(threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: captured.append(kwargs)))
    response = model_store.start_download("fastconformer-muno")
    assert captured[0]["args"][1] == token
    assert token not in json.dumps(response)


def test_zipformer_registration_does_not_claim_an_available_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "ROOT", tmp_path)
    assert model_store.runtime_path("zipformer-quran") is None
    assert model_store.spec("zipformer-quran")["integration_status"] == "awaiting-access"


def test_explicit_missing_mfa_executable_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("SURAH_CUTTER_MFA", str(tmp_path / "missing"))
    monkeypatch.setattr(mfa_runtime.shutil, "which", lambda _: str(Path("different-mfa")))
    with pytest.raises(ValueError, match="Установите MFA"):
        mfa_runtime.executable()


def test_native_call_is_killed_and_reaped_on_cancellation(tmp_path, monkeypatch):
    import sys
    import threading
    from surah_cutter.core.cancellation import cancellation_scope, GenerationCancelled

    real_popen = mfa_runtime.subprocess.Popen
    processes = []

    def start(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(mfa_runtime.subprocess, "Popen", start)
    event = threading.Event()
    timer = threading.Timer(.2, event.set)
    timer.start()
    try:
        with cancellation_scope(event), pytest.raises(GenerationCancelled):
            mfa_runtime._run([sys.executable, "-c", "import time; time.sleep(20)"], tmp_path)
    finally:
        timer.cancel()
        timer.join()
    assert len(processes) == 1 and processes[0].poll() is not None
