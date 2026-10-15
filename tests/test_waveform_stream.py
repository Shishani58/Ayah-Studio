"""Real decoder coverage: progressive PCM peaks, partial tail and cleanup."""

import asyncio
import json
import wave

import numpy as np
import pytest
from fastapi import HTTPException

from surah_cutter.core.decode import DecodeError
from surah_cutter.webgui import server, state


def audio_file(tmp_path, seconds=5.137):
    path = tmp_path / "114.wav"
    samples = (np.sin(np.arange(round(seconds * 16000)) * 0.03) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())
    return path, samples


def test_progressive_wave_matches_pcm_and_keeps_partial_tail(tmp_path):
    path, samples = audio_file(tmp_path)

    async def collect():
        return [event async for event in state.waveform_events(str(path))]

    events = asyncio.run(collect())
    chunks = [event for event in events if event["type"] == "chunk"]
    assert len(chunks) > 1
    assert chunks[0]["decoded_ms"] < events[-1]["duration_ms"]
    peaks = [value for chunk in chunks for value in chunk["peaks"]]
    bucket_samples = round(events[0]["bucket_ms"] * 16)
    expected = [float(np.abs(samples[i:i + bucket_samples].astype(float)).max()) / 32768
                for i in range(0, len(samples), bucket_samples)]
    assert peaks == pytest.approx(expected, abs=1e-6)
    assert [chunk["offset"] for chunk in chunks] == list(range(0, len(peaks), 200))
    assert events[-1]["duration_ms"] == 5137
    assert events[-1]["maximum"] == pytest.approx(max(expected), abs=1e-6)


def test_closing_wave_stream_reaps_decoder(tmp_path, monkeypatch):
    path, _ = audio_file(tmp_path, seconds=60)
    original = asyncio.create_subprocess_exec
    processes = []

    async def capture(*args, **kwargs):
        proc = await original(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)

    async def cancel():
        source = state.waveform_events(str(path))
        await anext(source)
        await anext(source)
        await source.aclose()

    asyncio.run(cancel())
    assert all(proc.returncode is not None for proc in processes)


def test_failed_decoder_is_not_a_completed_wave(tmp_path):
    async def collect():
        return [event async for event in state.waveform_events(str(tmp_path / "missing.mp3"))]

    with pytest.raises(DecodeError):
        asyncio.run(collect())


def test_http_disconnect_reaps_decoder(tmp_path, monkeypatch):
    path, _ = audio_file(tmp_path, seconds=60)
    app_state = state.AppState()
    app_state.audio_map = {114: path}
    monkeypatch.setattr(server, "STATE", app_state)
    original = asyncio.create_subprocess_exec
    processes = []

    async def capture(*args, **kwargs):
        proc = await original(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    endpoint = next(route.endpoint for route in server.app.routes
                    if route.path == "/api/sura/{surah}/waveform/stream")

    async def disconnect():
        first_chunk = asyncio.Event()

        async def send(message):
            if message.get("body") and b'"chunk"' in message["body"]:
                first_chunk.set()

        async def receive():
            await first_chunk.wait()
            return {"type": "http.disconnect"}

        await endpoint(114)({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)

    asyncio.run(asyncio.wait_for(disconnect(), timeout=10))
    assert len(processes) == 2
    assert all(proc.returncode is not None for proc in processes)


def test_wave_route_reports_error_and_checks_audio(monkeypatch, tmp_path):
    app_state = state.AppState()
    app_state.audio_map = {114: tmp_path / "missing.mp3"}
    monkeypatch.setattr(server, "STATE", app_state)
    endpoint = next(route.endpoint for route in server.app.routes
                    if route.path == "/api/sura/{surah}/waveform/stream")

    async def collect():
        response = endpoint(114)
        return [json.loads(chunk.removeprefix("data: "))
                async for chunk in response.body_iterator]

    events = asyncio.run(collect())
    assert events[-1]["type"] == "error"
    assert not any(event["type"] == "done" for event in events)
    with pytest.raises(HTTPException) as error:
        endpoint(113)
    assert error.value.status_code == 404
