"""Exercise the shipped worker with Qt signal stubs; no native UI dependency."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from surah_cutter.core import pipeline
from surah_cutter.data import alignment_result
from surah_cutter.data.project_db import ProjectDB


@pytest.fixture
def worker_class(monkeypatch):
    class Signal:
        def __init__(self, *args):
            self.values = []

        def emit(self, *args):
            self.values.append(args)

    core = ModuleType("PySide6.QtCore")
    for name in ("QObject", "Qt", "QThread"):
        setattr(core, name, object)
    core.Signal = Signal
    widgets = ModuleType("PySide6.QtWidgets")
    for name in ("QFileDialog", "QFrame", "QGridLayout", "QHBoxLayout", "QLabel",
                 "QProgressBar", "QPushButton", "QScrollArea", "QVBoxLayout", "QWidget"):
        setattr(widgets, name, object)
    monkeypatch.setitem(sys.modules, "PySide6", ModuleType("PySide6"))
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", core)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", widgets)
    path = Path(__file__).resolve().parents[1] / "surah_cutter/gui/dashboard_view.py"
    spec = importlib.util.spec_from_file_location("qt_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GenerationWorker


@pytest.mark.parametrize("mode,protected,expected_calls", [
    ("fast", False, [114]), ("fast", True, []), ("auto", False, []),
])
def test_qt_explicit_fast_does_not_skip_unknown_auto_identity(
    worker_class, tmp_path, monkeypatch, mode, protected, expected_calls,
):
    path = tmp_path / "p.db"
    (tmp_path / "114.mp3").touch()
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(114, a, a * 1000) for a in range(1, 7)], "old:requested_mode=fast")
        if protected:
            for ayah in range(1, 7):
                db.set_status(114, ayah, "verified")
    calls = []

    def process(_audio, surah, **kwargs):
        calls.append(surah)
        assert kwargs["mode"] == mode
        kwargs["trace"]["model_source"] = "reference:test"
        return [(114, a, a * 1000) for a in range(1, 7)]

    monkeypatch.setattr(pipeline, "process_surah", process)
    worker = worker_class(str(tmp_path), str(path), {114}, mode=mode)
    worker.run()
    assert not worker.error.values
    assert calls == expected_calls


@pytest.mark.parametrize("cancel_before_commit", [False, True])
def test_qt_cancel_scope_obeys_commit_boundary(worker_class, tmp_path, monkeypatch, cancel_before_commit):
    path = tmp_path / "p.db"
    (tmp_path / "114.mp3").touch()
    worker = worker_class(str(tmp_path), str(path), set(), mode="fast")

    def process(_audio, surah, **kwargs):
        kwargs["trace"]["model_source"] = "reference:test"
        if cancel_before_commit:
            worker.cancel()
        return [(114, a, a * 1000) for a in range(1, 7)]

    stage = alignment_result._stage_text

    def cancel_after_commit(*args):
        worker.cancel()
        return stage(*args)

    monkeypatch.setattr(pipeline, "process_surah", process)
    if not cancel_before_commit:
        monkeypatch.setattr(alignment_result, "_stage_text", cancel_after_commit)
    worker.run()
    assert not worker.error.values
    assert worker.surah_done.values == ([] if cancel_before_commit else [(114,)])
    with ProjectDB.open(str(path), "test", 1) as db:
        assert len(db.get_timings()) == (0 if cancel_before_commit else 6)
