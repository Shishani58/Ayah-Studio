"""Native calculation activity must not outlive the job that owns it."""

from contextlib import nullcontext
import sys
import threading
from types import SimpleNamespace

import pytest

from surah_cutter.core.cancellation import GenerationCancelled, current_event
from surah_cutter.webgui import activity, server


@pytest.mark.parametrize("failure", [None, ValueError, GenerationCancelled])
def test_native_activity_balanced_for_every_exit(monkeypatch, failure):
    calls = []
    token = object()
    process = SimpleNamespace(
        beginActivityWithOptions_reason_=lambda flags, reason: calls.append((flags, reason)) or token,
        endActivity_=lambda ended: calls.append(ended),
    )
    monkeypatch.setattr(activity.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(
        NSActivityUserInitiated=16777215,
        NSProcessInfo=SimpleNamespace(processInfo=lambda: process),
    ))
    monkeypatch.setitem(sys.modules, "objc", SimpleNamespace(autorelease_pool=nullcontext))

    def run():
        with activity.generation_activity():
            assert len(calls) == 1
            if failure:
                raise failure()
    if failure:
        with pytest.raises(failure):
            run()
    else:
        run()
    assert calls == [(16777215, "Ayah Studio calculation"), token]


def test_other_platforms_need_no_cocoa(monkeypatch):
    monkeypatch.setattr(activity.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "Foundation", None)
    with activity.generation_activity():
        pass


def test_browser_fallback_can_calculate_without_pyobjc(monkeypatch, caplog):
    monkeypatch.setattr(activity.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "Foundation", None)
    with activity.generation_activity():
        pass
    assert "PyObjC is not installed" in caplog.text


def test_sync_model_failure_releases_activity_and_runtime_lock(monkeypatch):
    from contextlib import contextmanager

    calls = []

    @contextmanager
    def held():
        calls.append("begin")
        try:
            yield
        finally:
            calls.append("end")

    @server._model_runtime
    def calculate():
        assert calls == ["begin"]
        raise ValueError("model failed")

    monkeypatch.setattr(activity, "generation_activity", held)
    with pytest.raises(ValueError, match="model failed"):
        calculate()
    assert calls == ["begin", "end"]
    acquired = []

    def check_unlocked():
        if server.model_store.RUNTIME_LOCK.acquire(blocking=False):
            acquired.append(True)
            server.model_store.RUNTIME_LOCK.release()

    thread = threading.Thread(target=check_unlocked)
    thread.start()
    thread.join(timeout=2)
    assert acquired == [True]


@pytest.mark.parametrize("failure", [None, ValueError, GenerationCancelled])
def test_job_holds_activity_and_releases_on_exit(monkeypatch, tmp_path, failure):
    calls = []
    job = server._GenJob(tmp_path / "probe.db", {})

    from contextlib import contextmanager

    @contextmanager
    def held():
        calls.append("begin")
        try:
            yield
        finally:
            calls.append("end")

    def calculate(suras, mode):
        assert calls == ["begin"]
        assert current_event() is job.cancel
        if failure:
            raise failure()

    monkeypatch.setattr(activity, "generation_activity", held)
    monkeypatch.setattr(job, "_run_selected", calculate)
    if failure:
        with pytest.raises(failure):
            job._run([2], "full")
    else:
        job._run([2], "full")
    assert calls == ["begin", "end"]
