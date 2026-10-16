"""Multiple actual SSE iterators must see the same outcome without blocking."""

import asyncio
import json

import pytest
from fastapi import HTTPException

from surah_cutter.webgui import server
from surah_cutter.webgui.generation import GenerationEvents


class Request:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.disconnected = False

    async def is_disconnected(self):
        return self.disconnected


def endpoint(path):
    return next(route.endpoint for route in server.app.routes if route.path == path)


def payload(chunk):
    return json.loads(next(line[6:] for line in chunk.splitlines() if line.startswith("data: ")))


def test_two_stream_iterators_receive_every_event_and_terminal(tmp_path, monkeypatch):
    job = server._GenJob(tmp_path / "timings.db", {})
    monkeypatch.setattr(server, "_JOB", job)
    stream = endpoint("/api/generate/stream")

    async def run():
        first = stream(Request(), job.job_id).body_iterator
        second = stream(Request(), job.job_id).body_iterator
        pending = [asyncio.create_task(anext(reader)) for reader in (first, second)]
        await asyncio.sleep(.02)
        job._emit({"type": "start", "total": 1, "total_ayahs": 6})
        starts = await asyncio.wait_for(asyncio.gather(*pending), 1)
        assert starts[0] == starts[1]
        job._emit({"type": "sura_log", "line": "diagnostic"})
        job._emit({"type": "ayah_progress", "ayah": 2})
        job._emit({"type": "done", "completed": 1})
        job.events.finish()
        outputs = await asyncio.gather(
            collect(first), collect(second),
        )
        assert outputs[0] == outputs[1]
        assert [payload(e)["type"] for e in outputs[0]] == ["sura_log", "ayah_progress", "done"]

    asyncio.run(run())


async def collect(reader):
    return [chunk async for chunk in reader]


@pytest.mark.parametrize("terminal", ["done", "cancelled", "error"])
def test_late_connect_and_reconnect_replay_terminal(tmp_path, monkeypatch, terminal):
    job = server._GenJob(tmp_path / "timings.db", {})
    monkeypatch.setattr(server, "_JOB", job)
    job._emit({"type": "start", "total": 1})
    job._emit({"type": terminal})
    job.events.finish()

    async def run():
        stream = endpoint("/api/generate/stream")
        late = await collect(stream(Request(), job.job_id).body_iterator)
        assert [payload(e)["type"] for e in late] == ["start", terminal]
        resumed = await collect(stream(Request({"last-event-id": "1"}), job.job_id).body_iterator)
        assert [payload(e)["type"] for e in resumed] == [terminal]
        current = await collect(stream(Request(), job.job_id, after=2).body_iterator)
        assert [payload(e)["type"] for e in current] == [terminal]

    asyncio.run(run())
    status = endpoint("/api/generate/status")()
    assert status["terminal"] == {"type": terminal}
    assert status["latest_event_id"] == 2
    assert not status["running"]


def test_disconnected_reader_and_cancelled_wait_do_not_affect_other_readers():
    history = GenerationEvents()

    async def run():
        dropped = Request()
        dropped.disconnected = True
        assert await collect(history.stream(dropped)) == []
        cancelled = history.stream(Request())
        pending = asyncio.create_task(anext(cancelled))
        await asyncio.sleep(.02)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        history.emit({"type": "cancelled", "completed": 0})
        history.finish()
        active = await asyncio.wait_for(collect(history.stream(Request())), 1)
        assert payload(active[-1]) == {"type": "cancelled", "completed": 0}

    asyncio.run(run())


def test_bounded_history_preserves_sequence_progress_and_terminal():
    history = GenerationEvents(capacity=3)
    history.emit({"type": "ayah_progress", "ayah": 4})
    for index in range(100):
        history.emit({"type": "sura_log", "line": str(index)})
    assert len(history.snapshot()) == 3
    assert history.status()["progress"] == {"type": "ayah_progress", "ayah": 4}
    history.emit({"type": "done", "completed": 1})
    history.finish()
    output = asyncio.run(collect(history.stream(Request())))
    assert output[0].startswith("id: 100\n")
    assert output[-1].startswith("id: 102\n")
    assert payload(output[-1]) == {"type": "done", "completed": 1}


def test_word_failure_total_survives_history_eviction_and_reconnect():
    history = GenerationEvents(capacity=2)
    history.emit({"type": "sura", "index": 0, "word_failures": 2})
    history.emit({"type": "sura", "index": 1, "word_failures": 3})
    history.emit({"type": "sura", "index": 2})
    history.emit({"type": "sura_log", "line": "Finished"})
    history.emit({"type": "done", "completed": 3})
    history.finish()
    assert history.status()["word_failures"] == 5
    assert all("word_failures" not in event for event in history.snapshot())
    output = asyncio.run(collect(history.stream(Request(), after=4)))
    assert [payload(event)["type"] for event in output] == ["done"]
    assert history.status()["word_failures"] == 5


def test_idle_heartbeat_and_disconnect_are_bounded():
    history = GenerationEvents()
    request = Request()

    async def run():
        reader = history.stream(request, heartbeat=.01, poll_interval=.005)
        assert await asyncio.wait_for(anext(reader), .5) == ": heartbeat\n\n"
        request.disconnected = True
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(reader), .5)

    asyncio.run(run())
    assert history.snapshot() == []


def test_status_identity_and_invalid_cursors(tmp_path, monkeypatch):
    job = server._GenJob(tmp_path / "timings.db", {})
    job.suras = [114]
    job.mode = "fast"
    job.running = True
    job.request_cancel()
    monkeypatch.setattr(server, "_JOB", job)
    status = endpoint("/api/generate/status")()
    assert (status["job_id"], status["suras"], status["mode"], status["cancel_requested"]) == (
        job.job_id, [114], "fast", True,
    )
    stream = endpoint("/api/generate/stream")
    for request, job_id, after, code in [
        (Request(), "replaced", 0, 409), (Request(), job.job_id, -1, 400),
        (Request({"last-event-id": "invalid"}), job.job_id, 0, 400),
    ]:
        with pytest.raises(HTTPException) as exc:
            stream(request, job_id, after)
        assert exc.value.status_code == code


def test_status_retains_counters_and_job_start_time_for_resume(tmp_path, monkeypatch):
    job = server._GenJob(tmp_path / "timings.db", {})
    monkeypatch.setattr(server, "_JOB", job)
    monkeypatch.setattr(server._GenJob, "_run", lambda *_args: None)
    monkeypatch.setattr(server.time, "time", lambda: 123456.789)
    monkeypatch.setattr(server.STATE, "db_path", job.db_path)
    job.start([113, 114], "fast")
    job.thread.join(timeout=1)
    job._emit({"type": "start", "total": 2, "total_ayahs": 11})
    job._emit({"type": "ayah_progress", "ayahs_done": 5, "total_ayahs": 11})
    job._emit({"type": "sura", "index": 1, "skipped": "no-audio"})
    status = endpoint("/api/generate/status")()
    assert status["started_at_ms"] == 123456789
    assert status["total_ayahs"] == 11
    assert status["ayahs_done"] == 5
    assert status["completed"] == 2
    assert status["same_project"]
    monkeypatch.setattr(server.STATE, "db_path", tmp_path / "another.db")
    assert not endpoint("/api/generate/status")()["same_project"]


def test_cancel_targets_job_identity_and_legacy_empty_call(tmp_path, monkeypatch):
    job = server._GenJob(tmp_path / "timings.db", {})
    job.running = True
    monkeypatch.setattr(server, "_JOB", job)
    cancel = endpoint("/api/generate/cancel")
    with pytest.raises(HTTPException) as exc:
        cancel({"job_id": "previous-job"})
    assert exc.value.status_code == 409
    assert not job.cancel.is_set()
    assert cancel({"job_id": job.job_id}) == {"cancelling": True}
    assert job.cancel.is_set()
    job.cancel.clear()
    assert cancel() == {"cancelling": True}
    assert job.cancel.is_set()
