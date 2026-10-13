import hashlib
import io
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from surah_cutter.core import align, emissions, model_store, pipeline
from surah_cutter.data.quran_text import Word
from surah_cutter.webgui import server


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "ROOT", tmp_path)
    monkeypatch.setattr(model_store, "_JOBS", {})
    monkeypatch.setattr(model_store, "_CANCEL", {})
    monkeypatch.setattr(model_store, "ACTIVE_ID", "quran-phonetics")
    monkeypatch.setattr(align, "_ONNX_CACHE", {})
    monkeypatch.setattr(server, "_JOB", None)


def endpoint(path):
    return next(route.endpoint for route in server.app.routes if route.path == path)


def download_fixture(monkeypatch, payload=b"weights"):
    file = {"path": "model.onnx", "size": 7,
            "sha256": hashlib.sha256(b"weights").hexdigest(), "git_oid": None}
    item = {"id": "test", "repo": "owner/model", "revision": "a" * 40,
            "license": "MIT", "files": [file]}
    requests = []

    def open_request(request, timeout):
        requests.append(request)
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_request))
    model_store._JOBS["test"] = {"state": "downloading"}
    return item, requests


def test_download_publishes_only_checked_complete_revision(monkeypatch):
    item, requests = download_fixture(monkeypatch)
    model_store._download(item, "hf_example", threading.Event())
    assert model_store.is_installed(item)
    assert requests[0].full_url.endswith("/" + "a" * 40 + "/model.onnx")
    assert model_store._JOBS["test"]["state"] == "installed"
    assert "hf_example" not in (model_store.installed_path(item) / "installation.json").read_text()


@pytest.mark.parametrize("payload", [b"short", b"corrupt", b"too-long-file"])
def test_bad_download_never_becomes_installed(monkeypatch, payload):
    item, _ = download_fixture(monkeypatch, payload)
    model_store._download(item, "", threading.Event())
    assert not model_store.installed_path(item).exists()
    assert not list(model_store.ROOT.glob("**/*.partial"))
    assert model_store._JOBS["test"]["state"] == "error"


def test_cancel_never_publishes(monkeypatch):
    item, _ = download_fixture(monkeypatch)
    cancel = threading.Event()
    cancel.set()
    model_store._download(item, "", cancel)
    assert not model_store.is_installed(item)
    assert model_store._JOBS["test"]["state"] == "cancelled"


def test_remove_preserves_files_in_recoverable_storage(monkeypatch):
    item, _ = download_fixture(monkeypatch)
    monkeypatch.setattr(model_store, "spec", lambda model_id: item)
    model_store._download(item, "", threading.Event())
    model_store.remove_download("test")
    assert not model_store.is_installed(item)
    recovered = list((model_store.ROOT / ".trash").glob("*/model.onnx"))
    assert len(recovered) == 1
    assert recovered[0].read_bytes() == b"weights"
    assert "test" not in model_store._JOBS


def test_remove_rejects_active_and_downloading_models(monkeypatch):
    item, _ = download_fixture(monkeypatch)
    monkeypatch.setattr(model_store, "spec", lambda model_id: item)
    monkeypatch.setattr(model_store, "ACTIVE_ID", "test")
    with pytest.raises(ValueError, match="другую"):
        model_store.remove_download("test")
    monkeypatch.setattr(model_store, "ACTIVE_ID", "quran-phonetics")
    with pytest.raises(ValueError, match="загрузки"):
        model_store.remove_download("test")


def test_remove_cannot_target_bundled_fp32_weights():
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/models/{model_id}/remove")("quran-phonetics-fp32")
    assert exc.value.status_code == 400


def test_closed_models_are_excluded_and_unknown_ids_rejected():
    assert all(not item.get("gated") for item in model_store.catalog())
    assert "zipformer" not in {item["id"] for item in model_store.catalog()}
    with pytest.raises(ValueError, match="каталоге"):
        model_store.start_download("zipformer")
    with pytest.raises(ValueError, match="каталоге"):
        model_store.start_download("../outside")


def test_redirect_strips_token_from_other_hosts():
    request = urllib.request.Request("https://huggingface.co/model", headers={"Authorization": "Bearer secret"})
    redirected = model_store._Redirect().redirect_request(request, None, 302, "", {}, "https://cdn.example/model")
    assert redirected.get_header("Authorization") is None
    with pytest.raises(ValueError, match="HTTPS"):
        model_store._Redirect().redirect_request(request, None, 302, "", {}, "http://cdn.example/model")


def test_select_rejects_downloaded_unsupported_model():
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/models/{model_id}/select")("whisper-quran")
    assert exc.value.status_code == 400
    assert model_store.ACTIVE_ID == "quran-phonetics"


def test_selection_persists_only_after_runtime_probe(monkeypatch):
    monkeypatch.setattr(model_store, "runtime_path", lambda model: Path("model.onnx"))
    emitter = SimpleNamespace(vocab={"<pad>": 0}, blank_id=0, emissions=lambda pcm: np.zeros((2, 1)))
    monkeypatch.setattr(emissions, "load_emitter", lambda *a, **kw: emitter)
    seen = []
    monkeypatch.setattr(server.settings, "update", lambda **kw: seen.append(kw))
    monkeypatch.setattr(model_store, "listing", lambda: {})
    endpoint("/api/models/{model_id}/select")("arabic-ctc")
    assert seen == [{"strict": True, "alignment_model": "arabic-ctc", "mode": "full"}]
    assert model_store.ACTIVE_ID == "arabic-ctc"


def test_failed_probe_or_save_preserves_selection(monkeypatch):
    monkeypatch.setattr(model_store, "runtime_path", lambda model: Path("model.onnx"))
    emitter = SimpleNamespace(vocab={"<pad>": 0}, blank_id=0, emissions=lambda pcm: np.zeros((2, 1)))
    monkeypatch.setattr(emissions, "load_emitter", lambda *a, **kw: emitter)
    monkeypatch.setattr(server.settings, "update", lambda **kw: (_ for _ in ()).throw(OSError()))
    with pytest.raises(HTTPException):
        endpoint("/api/models/{model_id}/select")("arabic-ctc")
    assert model_store.ACTIVE_ID == "quran-phonetics"


def test_running_job_blocks_selection(monkeypatch):
    monkeypatch.setattr(server, "_JOB", SimpleNamespace(running=True))
    with pytest.raises(HTTPException) as exc:
        endpoint("/api/models/{model_id}/select")("quran-phonetics-fp32")
    assert exc.value.status_code == 409


def test_arabic_targets_use_canonical_words_not_latin_transliteration():
    vocab = {"<pad>": 0, "ا": 1, "و": 2, "ي": 3, "|": 4}
    bundle = SimpleNamespace(vocab=vocab, delim_id=4, text_kind="arabic")
    words = [Word(1, 1, "ٱۥ", 0), Word(1, 1, "ۦ۩", 1)]
    targets, spans = align._build_targets(words, ["wrong", "wrong"], bundle)
    assert targets == [1, 2, 4, 3]
    assert spans == [(0, 2), (3, 4)]
    assert words[0].text == "ٱۥ"
    with pytest.raises(ValueError, match="Символ"):
        align._arabic_target("unknown", vocab)


def test_arabic_rejects_phonetic_fast_mode(monkeypatch):
    monkeypatch.setattr(model_store, "ACTIVE_ID", "arabic-ctc")
    with pytest.raises(ValueError, match="Полный"):
        pipeline.process_surah("unused.wav", 1, mode="fast")


def test_model_specific_lag_does_not_apply_phonetic_delay():
    word = Word(1, 1, "ا", 0)
    checks = align._check_ayah_onsets([word], [(0, 1)], [(20, 21)],
                                    np.zeros(30), 20, pauses=[], lag_ms=0)
    assert checks[1]["onset_ms"] == 400
