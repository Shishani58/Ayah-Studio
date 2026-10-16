"""Bounded generation event history with independent, reconnectable SSE cursors."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any


class GenerationEvents:
    """Retain recent events once; readers never consume another reader's events."""

    def __init__(self, capacity: int = 512) -> None:
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._sequence = 0
        self._finished = False
        self._terminal: tuple[int, dict[str, Any]] | None = None
        self._progress: dict[str, Any] | None = None
        self._total_ayahs = 0
        self._ayahs_done = 0
        self._completed = 0
        self._word_failures = 0
        self._unresolved_boundaries = 0
        self._stage = None

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            self._events.append((self._sequence, dict(event)))
            if "total_ayahs" in event:
                self._total_ayahs = event["total_ayahs"]
            if "ayahs_done" in event:
                self._ayahs_done = event["ayahs_done"]
            if "completed" in event:
                self._completed = event["completed"]
            elif event["type"] == "sura" and "index" in event:
                self._completed = event["index"] + 1
            if event["type"] == "sura":
                self._word_failures += event.get("word_failures", 0)
                self._unresolved_boundaries += len(event.get("unresolved_ayahs", ()))
            if event["type"] == "stage_progress":
                self._stage = dict(event)
            elif event["type"] == "sura_start":
                self._stage = None
            if event["type"] in ("start", "sura_start", "ayah_progress", "sura"):
                self._progress = dict(event)
            if event["type"] in ("done", "cancelled", "error"):
                self._terminal = self._events[-1]

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for _, event in self._events]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "latest_event_id": self._sequence,
                "total_ayahs": self._total_ayahs,
                "ayahs_done": self._ayahs_done,
                "completed": self._completed,
                "word_failures": self._word_failures,
                "unresolved_boundaries": self._unresolved_boundaries,
                "stage": dict(self._stage) if self._stage else None,
                "progress": dict(self._progress) if self._progress else None,
                "terminal": dict(self._terminal[1]) if self._terminal else None,
            }

    async def stream(self, request, after: int = 0, *, heartbeat: float = 15.0,
                     poll_interval: float = 0.1):
        """Async waits are cancellable; no blocked executor thread/subscriber queue."""
        cursor = after
        last_sent = time.monotonic()
        delivered_terminal = False
        while True:
            if await request.is_disconnected():
                return
            with self._lock:
                events = [(n, dict(e)) for n, e in self._events if n > cursor]
                finished = self._finished
                terminal = self._terminal
            # A completed job must explain its outcome even to an up-to-date late reader.
            if finished and terminal and not events and not delivered_terminal:
                events = [terminal]
            for sequence, event in events:
                if await request.is_disconnected():
                    return
                cursor = sequence
                delivered_terminal |= event["type"] in ("done", "cancelled", "error")
                last_sent = time.monotonic()
                yield f"id: {sequence}\ndata: {json.dumps(event)}\n\n"
            if finished:
                return
            if time.monotonic() - last_sent >= heartbeat:
                last_sent = time.monotonic()
                yield ": heartbeat\n\n"
            await asyncio.sleep(poll_interval)
