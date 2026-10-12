"""Build preparation must fail closed without destroying installed weights."""

import hashlib
import io
import json

import pytest

from tools import prepare_models as models


def test_download_rejects_wrong_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    target = tmp_path / "model.onnx"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        models.download_checked("https://example.test/model", target, "0" * 64)
    assert not target.exists()


def test_prepare_pins_export_and_only_copies_runtime_files(tmp_path, monkeypatch):
    sources = json.loads((models.ROOT / "tools/model_sources.json").read_text())
    sources["silero"]["sha256"] = hashlib.sha256(b"silero").hexdigest()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(sources))
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"silero"))

    def fake_export(model_id, out, *, revision, opset):
        assert model_id == sources["acoustic"]["model_id"]
        assert revision == sources["acoustic"]["revision"]
        assert opset == 14
        for name in models.RUNTIME_FILES[:-1]:
            (out.parent / name).write_bytes(b"runtime")
        (out.parent / "extra.fp32.onnx").write_bytes(b"large")

    monkeypatch.setattr(models, "export", fake_export)
    output = tmp_path / "models"
    models.prepare(output, manifest)
    assert {p.name for p in output.iterdir()} == set(models.RUNTIME_FILES)


def test_failed_export_preserves_existing_models(tmp_path, monkeypatch):
    output = tmp_path / "models"
    output.mkdir()
    target = output / "silero_vad.onnx"
    target.write_bytes(b"existing")
    monkeypatch.setattr(models, "download_checked", lambda u, p, h: p.write_bytes(b"new"))

    def fail(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(models, "export", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        models.prepare(output)
    assert target.read_bytes() == b"existing"
