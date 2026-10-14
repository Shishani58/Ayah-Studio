"""Contracts at model boundaries; acoustic accuracy is in tools/bench_models."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, model_runtime, model_store, word_align
from surah_cutter.core.model_worker import Acoustic, WhisperTextOverflow
from surah_cutter.data import quran_text
from surah_cutter.webgui import server


def test_inference_renderings_cover_canonical_words_without_reordering():
    count = 0
    for surah in range(1, 115):
        words = quran_text.surah_words(surah)
        originals = [(w.index, w.text, w.ayah) for w in words]
        for model in ("muaalem", "fastconformer", "mms-arabic", "whisper-quran", "whisper-small",
                      "whisper-tarteel-tiny", "whisper-tarteel-base"):
            rendered = model_runtime.inference_words(words, model)
            assert len(rendered) == len(words)
            assert all(text.strip() for text in rendered)
        assert [(w.index, w.text, w.ayah) for w in words] == originals
        count += len(words)
    assert count == 77432
    # Canonical QPC has one word containing two orthographic units here.
    words = [w for w in quran_text.surah_words(37) if w.ayah == 130]
    assert len(words) == len(model_runtime.inference_words(words, "muaalem")) == 3


def test_mismatched_word_index_is_rejected():
    word = quran_text.Word(114, 1, "foreign text", 0)
    with pytest.raises(ValueError, match="Порядок"):
        model_runtime.inference_words([word], "fastconformer")


@pytest.mark.parametrize("model_id", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_tarteel_uses_vocalized_targets_and_pinned_generation_metadata(model_id):
    import unicodedata

    words = quran_text.surah_words(112)
    rendered = model_runtime.inference_words(words, model_id)
    assert any(unicodedata.category(c) == "Mn" for c in rendered[0])
    entry = model_store.spec(model_id)
    assert all(f["sha256"] and len(f["sha256"]) == 64 for f in entry["files"])
    assert not any(f["path"] == "training_args.bin" for f in entry["files"])
    assert entry["generation_config"]["alignment_heads"]
    assert len(entry["generation_config_source"]["revision"]) == 40


def test_tarteel_full_shrinks_audio_on_token_overflow_without_truncating_text():
    from contextlib import nullcontext

    acoustic = Acoustic.__new__(Acoustic)
    acoustic.model_id = "whisper-tarteel-tiny"
    acoustic.torch = SimpleNamespace(inference_mode=nullcontext)
    class Processor:
        def __call__(self, *a, **kw):
            return SimpleNamespace(input_features=None, attention_mask=None)

        def batch_decode(self, *a, **kw):
            return ["one two three four"]

    acoustic.processor = Processor()
    acoustic.model = SimpleNamespace(generate=lambda *a, **kw: None)
    seen = []

    def overflow(pcm, words, **kw):
        seen.append((len(pcm) // 16000, list(words)))
        raise WhisperTextOverflow("token budget")

    acoustic.whisper_clip = overflow
    words = ["one", "two", "three", "four"]
    with pytest.raises(WhisperTextOverflow):
        acoustic.whisper_align(np.zeros(35 * 16000), words, strict_words=False)
    assert seen == [(30, words), (20, words), (10, words)]


def test_all_open_models_route_to_their_own_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "ROOT", tmp_path)
    ids = [m["id"] for m in model_store.catalog() if m["id"] not in ("quran-phonetics", "arabic-ctc")]
    for model_id in ids:
        item = model_store.spec(model_id)
        folder = model_store.installed_path(item)
        folder.mkdir(parents=True)
        (folder / "installation.json").write_text("{}")
        for file in item["files"]:
            path = folder / file["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as output:
                output.truncate(file["size"])
        path = model_store.runtime_path(model_id)
        assert path.is_relative_to(folder)
        assert path.exists()


@pytest.mark.parametrize("model_id", ["muaalem", "whisper-large-v3", "fastconformer-arabic",
                                      "whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_selection_probes_requested_model_before_saving(model_id, tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "ACTIVE_ID", "quran-phonetics")
    monkeypatch.setattr(align, "_ONNX_CACHE", {})
    monkeypatch.setattr(server, "_JOB", None)
    monkeypatch.setattr(model_store, "runtime_path", lambda _: tmp_path)
    monkeypatch.setattr(model_store, "listing", lambda: {})
    seen = []

    def load(requested):
        assert requested == model_id
        return SimpleNamespace(vocab={"blank": 0}, blank_id=0, text_kind="whisper" if model_id.startswith("whisper-") else model_id,
                               probe=lambda: seen.append("probe"))

    monkeypatch.setattr(model_store, "load_emitter", load)
    monkeypatch.setattr(server.settings, "update", lambda **kw: seen.append(kw))
    monkeypatch.setattr(server.settings, "load", lambda: {"mode": "fast"})
    monkeypatch.setattr(model_runtime, "release_runtimes", lambda **kw: None)
    select = next(r.endpoint for r in server.app.routes if r.path == "/api/models/{model_id}/select")
    select(model_id)
    expected = {"strict": True, "alignment_model": model_id}
    if model_id not in ("whisper-tarteel-tiny", "whisper-tarteel-base"):
        expected["mode"] = "full"
    assert seen == ["probe", expected]
    assert model_store.ACTIVE_ID == model_id


def test_public_catalog_keeps_only_shortlist_without_deleting_archived_specs():
    expected = {"quran-phonetics", "muaalem", "fastconformer-arabic", "whisper-large-v3",
                "whisper-tarteel-tiny", "whisper-tarteel-base"}
    assert {item["id"] for item in model_store.listing()["models"]} == expected
    assert all(item["description"] and item["benchmark_summary"] for item in model_store.available_catalog())
    assert model_store.spec("whisper-medium")["hidden"]


def test_archived_selection_is_rejected_before_loading_or_saving(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(model_store, "load_emitter", lambda *a: pytest.fail("archived weights loaded"))
    monkeypatch.setattr(server.settings, "update", lambda **kw: pytest.fail("selection saved"))
    select = next(r.endpoint for r in server.app.routes if r.path == "/api/models/{model_id}/select")
    with pytest.raises(HTTPException, match="исключена"):
        select("whisper-medium")


def test_archived_saved_selection_migrates_to_visible_default(tmp_path, monkeypatch):
    monkeypatch.setattr(server.settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(server, "_restore_from_settings", lambda: None)
    monkeypatch.setattr(model_store, "ACTIVE_ID", "whisper-medium")
    server.settings.update(strict=True, alignment_model="whisper-medium", mode="full")
    server.create_app()
    assert model_store.ACTIVE_ID == "quran-phonetics"
    assert server.settings.load()["alignment_model"] == "quran-phonetics"


def test_custom_ctc_targets_do_not_use_phonetic_transliteration():
    words = quran_text.surah_words(114)[:4]
    seen = []
    emitter = SimpleNamespace(build_targets=lambda w, intro: (seen.append((w, intro)) or ([7, 8], [(0, 1), (1, 2)])))
    bundle = SimpleNamespace(emitter=emitter)
    assert align._build_targets(words, ["wrong"], bundle, ["intro"])[0] == [7, 8]
    assert seen == [(words, True)]


def test_whisper_word_path_uses_known_words_without_ctc(monkeypatch):
    words = [w for w in quran_text.surah_words(114) if w.ayah == 1]
    pcm = np.zeros(16000)
    expected = [(w.index, i * 200, .9) for i, w in enumerate(words)]
    emitter = SimpleNamespace(align_words=lambda audio, known: expected if known == words and audio is pcm else None)
    monkeypatch.setattr(align, "_get_onnx_bundle", lambda: SimpleNamespace(text_kind="whisper", emitter=emitter))
    monkeypatch.setattr(word_align, "forced_align", lambda *a, **kw: pytest.fail("Whisper must not use another CTC model"))
    assert word_align.align_ayah_words(pcm, 16000, words) == expected


def test_range_download_rejects_an_unconfirmed_slice(tmp_path, monkeypatch):
    response = SimpleNamespace(status=200, headers={})
    import contextlib

    monkeypatch.setattr(model_store, "_open_download", lambda *a: contextlib.nullcontext(response))
    import threading

    with pytest.raises(OSError, match="часть"):
        model_store._download_ranges(tmp_path / "weights", "https://example.org/weights", 1024, {},
                                     threading.Event(), lambda _: None)


def test_range_download_resumes_prefix_and_preserves_original_failure(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import io
    import threading

    block = 32 * 1024 * 1024
    path = tmp_path / "weights"
    path.write_bytes(b"a" * block)
    requested = []

    def opened(opener, url, headers):
        requested.append(headers["Range"])
        response = io.BytesIO(b"tail")
        response.status = 206
        response.headers = {"Content-Range": f"bytes {block}-{block + 3}/{block + 4}"}
        return nullcontext(response)

    monkeypatch.setattr(model_store, "_open_download", opened)
    model_store._download_ranges(path, "https://example.org/weights", block + 4, {},
                                 threading.Event(), lambda _: None)
    assert requested == [f"bytes={block}-{block + 3}"]
    assert path.read_bytes() == b"a" * block + b"tail"

    def failed(*args):
        raise TimeoutError("original transport failure")

    monkeypatch.setattr(model_store, "_open_download", failed)
    with pytest.raises(TimeoutError, match="original transport failure"):
        model_store._download_ranges(tmp_path / "failed", "https://example.org/weights", 1024, {},
                                     threading.Event(), lambda _: None)


def test_checked_text_cache_records_generator_and_canonical_digest():
    database = quran_text.DEFAULT_DB_PATH
    data = json.loads(database.with_name("model_text_targets.json").read_text())
    assert data["canonical_sha256"] == hashlib.sha256(database.read_bytes()).hexdigest()
    assert data["generator"] == "quran-transcript:0.6.4"


def test_whisper_refuses_nonadvancing_overlap_instead_of_looping():
    from contextlib import nullcontext

    class Processor:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_features=None, attention_mask=None)

        def batch_decode(self, *args, **kwargs):
            return ["one two three four"]

    acoustic = Acoustic.__new__(Acoustic)
    acoustic.torch = SimpleNamespace(inference_mode=nullcontext)
    acoustic.processor = Processor()
    acoustic.model = SimpleNamespace(generate=lambda *a, **kw: None)
    acoustic.whisper_clip = lambda *a, **kw: [(0, .9)] * 4
    with pytest.raises(ValueError, match="продвинулся по аудио"):
        acoustic.whisper_align(np.zeros(35 * 16000), ["one", "two", "three", "four"], strict_words=False)


def test_legacy_oracle_default_cannot_replace_selected_model(monkeypatch):
    monkeypatch.setattr(model_store, "ACTIVE_ID", "mms-fa")
    monkeypatch.setattr(align, "DEFAULT_BACKEND", "torch")
    monkeypatch.setattr(align, "_get_bundle", lambda *a: pytest.fail("selected MMS was replaced by Quran Phonetics"))
    monkeypatch.setattr(align, "_get_onnx_bundle", lambda: (_ for _ in ()).throw(RuntimeError("selected weights")))
    with pytest.raises(RuntimeError, match="selected weights"):
        align.align_surah(np.zeros(16000), 16000, quran_text.surah_words(114))
    with pytest.raises(ValueError, match="backend"):
        align.align_surah(np.zeros(16000), 16000, quran_text.surah_words(114), backend="torch")


@pytest.mark.parametrize("model_id", ["mms-fa", "muaalem", "fastconformer", "whisper-quran",
                                      "whisper-small", "whisper-medium", "whisper-large-v3",
                                      "whisper-large-v3-turbo", "fastconformer-arabic", "mms-arabic"])
def test_additional_models_cannot_enter_fast_or_hidden_full_fallback(model_id, monkeypatch):
    from surah_cutter.core import pipeline

    monkeypatch.setattr(model_store, "ACTIVE_ID", model_id)
    monkeypatch.setattr(pipeline, "_full_align", lambda *a, **kw: pytest.fail("hidden full pass"))
    monkeypatch.setattr(pipeline, "_fast_align", lambda *a, **kw: pytest.fail("unsupported FAST runtime"))
    with pytest.raises(ValueError, match="Выбранная модель"):
        pipeline.process_surah("unused.mp3", 114, mode="fast")


def test_single_frame_ctc_word_keeps_its_acoustic_confidence(monkeypatch):
    words = [quran_text.Word(1, 1, "a", 0), quran_text.Word(1, 1, "b", 1)]
    emitter = SimpleNamespace(build_targets=lambda *a, **kw: ([1, 2], [(0, 1), (1, 2)]))
    bundle = SimpleNamespace(emitter=emitter, text_kind="test", blank_id=0, lag_ms=0)
    monkeypatch.setattr(align, "_get_onnx_bundle", lambda: bundle)
    monkeypatch.setattr(align, "_emissions", lambda *a: np.zeros((10, 3)))
    path = np.array([[0, 1, 0, 0, 0, 2, 0, 0, 0, 0]])
    monkeypatch.setattr(word_align, "forced_align", lambda *a, **kw: (path, np.log(np.full((1, 10), .9))))
    rows = word_align.align_ayah_words(np.zeros(16000), 16000, words)
    assert [r[2] for r in rows] == [.9, .9]
