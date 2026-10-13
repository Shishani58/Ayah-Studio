"""Cancellation must interrupt native/RPC waits and leave them reusable."""

import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core.cancellation import (
    GenerationCancelled, cancellation_scope, check_cancelled, current_event,
)
from surah_cutter.core.emissions import _run
from surah_cutter.core.model_runtime import RuntimeEmitter


def test_scope_restores_and_does_not_cancel_other_threads():
    event = threading.Event()
    seen = []
    with cancellation_scope(event):
        event.set()
        thread = threading.Thread(target=lambda: seen.append(current_event()))
        thread.start()
        thread.join(timeout=2)
        with pytest.raises(GenerationCancelled):
            check_cancelled()
    assert seen == [None]
    assert current_event() is None


def test_onnx_wait_interrupts_and_next_run_has_fresh_options():
    event = threading.Event()
    entered = threading.Event()
    outcomes = []

    class Session:
        def run(self, names, inputs, options=None):
            if options is not None:
                entered.set()
                while not options.terminate:
                    event.wait(0.01)
                raise RuntimeError("ONNX terminated")
            return [inputs["audio"]]

    def calculate():
        with cancellation_scope(event):
            try:
                _run(Session(), np.ones((1, 10)), "audio", "out")
            except GenerationCancelled:
                outcomes.append("cancelled")

    thread = threading.Thread(target=calculate)
    thread.start()
    assert entered.wait(2)
    event.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert outcomes == ["cancelled"]
    assert _run(Session(), np.ones((1, 10)), "audio", "out").shape == (1, 10)


def test_runtime_blocking_read_kills_real_child():
    event = threading.Event()
    # No model weights: exercise the actual blocking pipe and process kill.
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"],
        stdout=subprocess.PIPE, text=True,
    )
    emitter = RuntimeEmitter.__new__(RuntimeEmitter)
    emitter.process = child
    ready = threading.Event()
    outcomes = []

    def calculate():
        with cancellation_scope(event):
            ready.set()
            try:
                emitter._receive()
            except GenerationCancelled:
                outcomes.append("cancelled")

    try:
        assert child.stdout.readline().strip() == "ready"
        thread = threading.Thread(target=calculate)
        thread.start()
        assert ready.wait(2)
        event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert outcomes == ["cancelled"]
        assert child.wait(timeout=2) != 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
        child.stdout.close()


def test_dead_runtime_bundle_is_reloaded(tmp_path, monkeypatch):
    from surah_cutter.core import align, model_store

    path = tmp_path / "model.onnx"
    path.touch()
    dead = SimpleNamespace(emitter=SimpleNamespace(process=SimpleNamespace(poll=lambda: -9)))
    replacement = object()
    monkeypatch.setattr(align, "_ONNX_CACHE", {str(path): dead})
    monkeypatch.setattr(model_store, "ACTIVE_ID", "quran-phonetics")
    monkeypatch.setattr(model_store, "runtime_path", lambda _: path)
    monkeypatch.setattr(model_store, "load_emitter", lambda _: replacement)
    monkeypatch.setattr(align, "_OnnxBundle", lambda emitter: emitter)
    assert align._get_onnx_bundle() is replacement
