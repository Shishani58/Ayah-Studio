import numpy as np
import pytest

from surah_cutter.core import model_store, pipeline
from surah_cutter.core.model_worker import Acoustic


@pytest.mark.parametrize("samples,total", [(100000, 200000), (80000, 80000), (1000, 200000)])
@pytest.mark.parametrize("model_id", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_worker_local_action_rejects_nonlocal_input_before_model(model_id, samples, total):
    acoustic = Acoustic.__new__(Acoustic)
    acoustic.model_id = model_id
    with pytest.raises(ValueError, match="локальный"):
        acoustic.whisper_local(np.zeros(samples), total)


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
