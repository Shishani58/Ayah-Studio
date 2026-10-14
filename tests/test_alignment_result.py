import json

import pytest

from surah_cutter.core import model_store
from surah_cutter.data.alignment_result import result_source, write_alignment_result
from surah_cutter.data.project_db import ProjectDB


def test_writer_preserves_provenance_trace_and_confidence(tmp_path):
    path = tmp_path / "project.db"
    trace = {
        "model_source": "onnx-ctc:test:revision=abc123:backend=onnx:lag=200",
        "auto_mode": "full", "backend": "onnx", "align_lag_ms": 200,
        "lag_adjust_ms": -40,
        "ayahs": [{"ayah": 1, "conf": 0.7}],
        "local_reads": [[0, 1000]],
    }
    with ProjectDB.open(str(path), "test", 1) as db:
        written, logs = write_alignment_result(
            db, [(1, 1, 100)], trace=trace, requested_mode="auto",
            db_path=path, log_lines=["aligned"],
        )
        review = db.get_review(1, 1)
    assert written == 1
    assert "revision=abc123" in review["source"]
    assert ("requested_mode=auto:actual_mode=full:align_lag_ms=200:lag_adjust_ms=-40"
            in review["source"])
    assert review["confidence"] == 0.7
    payload = json.loads(open(logs["json"], encoding="utf-8").read())
    assert payload["trace"] == trace
    assert payload["rows"] == [[1, 1, 100]]


def test_local_export_source_identifies_artifact(tmp_path, monkeypatch):
    model = tmp_path / "weights.onnx"
    monkeypatch.setattr(model_store, "ACTIVE_ID", "quran-phonetics")
    monkeypatch.setattr(model_store, "runtime_path", lambda _id: model)
    model.write_bytes(b"first model")
    first = model_store.source()
    model.write_bytes(b"second model")
    second = model_store.source()
    assert "revision=local-export:sha256=" in first
    assert first != second


def test_reference_shortcut_reports_reference_without_model_lag():
    source = result_source({"model_source": "reference:reader"}, "auto")
    assert source.endswith(
        "backend=reference:requested_mode=auto:actual_mode=reference:"
        "align_lag_ms=0:lag_adjust_ms=0"
    )


def test_trace_backend_and_alignment_lag_override_default_model_source():
    source = result_source({"model_source": "torch-ctc:oracle", "backend": "torch",
                            "align_lag_ms": 125}, "full")
    assert "backend=torch:requested_mode=full:actual_mode=full:align_lag_ms=125" in source


def test_log_preflight_failure_leaves_project_untouched(tmp_path):
    path = tmp_path / "project.db"
    (tmp_path / "project.logs").symlink_to(tmp_path / "elsewhere")
    with ProjectDB.open(str(path), "test", 1) as db:
        with pytest.raises(ValueError, match="symlink"):
            write_alignment_result(
                db, [(1, 1, 100)], trace={"model_source": "reference:reader"},
                requested_mode="auto", db_path=path,
            )
        assert db.get_timings() == []
