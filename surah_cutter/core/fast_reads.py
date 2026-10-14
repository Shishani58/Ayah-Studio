"""Guard and account actual local acoustic inference in FAST."""

from __future__ import annotations

from math import isfinite

import numpy as np

from surah_cutter.core.cancellation import GenerationCancelled, check_cancelled


class LocalAcousticReads:
    """Shared local bounds and irreversible accounting of model calls.

    Hard limits are independent of FAST environment tuning. Over-budget callers
    receive no evidence and must leave a review flag, rather than escalate.
    """

    def __init__(self, pcm, sr, bundle, *, max_read_ms=6000.0, log_event=None):
        self.pcm, self.sr, self.bundle = pcm, sr, bundle
        self.log_event = log_event
        self.total_ms = len(pcm) * 1000.0 / sr
        self.max_read_ms = min(6000.0, max_read_ms)
        self.audio_ms = 0.0
        self.covered_ms = 0.0
        self.frames = 0
        self.calls = []
        self.denied = []
        self.intervals = []
        self.cache_limit_bytes = None
        self.cache_bytes = 0
        self.peak_cache_bytes = 0
        self.cache_hits = 0
        self.cache_evictions = 0

    def _window(self, start_ms: float, end_ms: float) -> tuple[int, int] | None:
        """Reject invalid bounds before rounding or looking up cached evidence."""
        try:
            if not isfinite(start_ms) or not isfinite(end_ms) or end_ms <= start_ms:
                return None
            s0 = round(min(self.total_ms, max(0.0, start_ms)) * self.sr / 1000.0)
            s1 = round(min(self.total_ms, max(0.0, end_ms)) * self.sr / 1000.0)
        except (TypeError, ValueError, OverflowError):
            return None
        return s0, s1

    def _begin_read(
        self, s0: int, s1: int, allowed: tuple[list[tuple[int, int]], float], *, phase: str,
    ) -> dict[str, object]:
        """Charge a started model call even if it fails or is cancelled."""
        self.intervals, self.covered_ms = allowed
        self.audio_ms += (s1 - s0) * 1000.0 / self.sr
        attempt: dict[str, object] = {
            "phase": phase,
            "window_ms": [s0 * 1000.0 / self.sr, s1 * 1000.0 / self.sr],
            "frames": 0,
            "outcome": "started",
        }
        self.calls.append(attempt)
        return attempt

    def _permit(self, s0, s1, reason=None):
        """Shared hard budget for CTC and Whisper; cache hits are handled first."""
        duration = (s1 - s0) * 1000.0 / self.sr
        if duration < 300.0:
            return None
        if duration > self.max_read_ms or s1 - s0 >= len(self.pcm):
            reason = "nonlocal_window"
        intervals = sorted(self.intervals + [(s0, s1)])
        merged = []
        for left, right in intervals:
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        covered = sum(right - left for left, right in merged) * 1000.0 / self.sr
        if covered > self.total_ms * 0.65:
            reason = reason or "coverage_budget"
        if self.audio_ms + duration > self.total_ms * 2.0:
            reason = reason or "inference_budget"
        if reason:
            event = {"window_ms": [s0 * 1000.0 / self.sr, s1 * 1000.0 / self.sr], "reason": reason}
            if event not in self.denied:
                self.denied.append(event)
                if self.log_event:
                    self.log_event(f"FAST: модель не читает {event['window_ms']}, причина={reason}")
            return None
        return merged, covered

    def diagnostics(self):
        return {
            "emission_passes": len(self.calls), "emissions_frames": self.frames,
            "acoustic_reads": self.calls, "denied_acoustic_reads": self.denied,
            "audio_ms": self.audio_ms, "covered_audio_ms": self.covered_ms,
            "coverage": self.covered_ms / self.total_ms if self.total_ms else 0.0,
            "acoustic_intervals_ms": [[a * 1000.0 / self.sr, b * 1000.0 / self.sr] for a, b in self.intervals],
            "cache_hits": self.cache_hits, "cache_bytes": self.cache_bytes,
            "peak_cache_bytes": self.peak_cache_bytes, "cache_evictions": self.cache_evictions,
            "local_read_limits": {"cache_limit_bytes": self.cache_limit_bytes, "max_read_ms": self.max_read_ms, "max_coverage": 0.65, "max_audio_ratio": 2.0},
        }
class LocalEvidenceReads(LocalAcousticReads):
    """One sequential ledger for adapter observations; exact-window cache only."""

    def __init__(self, pcm, sr, adapter, *, cache_limit_bytes=32 * 1024 * 1024):
        super().__init__(pcm, sr, None)
        self.adapter = adapter
        self.cache_limit_bytes = min(cache_limit_bytes, 32 * 1024 * 1024)
        self.observations = {}
        self.last_read_id = None

    def read(self, start_ms, end_ms, *, reason):
        from dataclasses import fields
        from copy import deepcopy
        import sys
        from time import monotonic

        check_cancelled()
        window = self._window(start_ms, end_ms)
        if window is None:
            return None
        s0, s1 = window
        if window in self.observations:
            observation, self.last_read_id, _ = self.observations[window]
            self.cache_hits += 1
            return observation
        allowed = self._permit(s0, s1)
        if allowed is None:
            return None
        check_cancelled()
        attempt = self._begin_read(s0, s1, allowed, phase="independent-local")
        self.last_read_id = len(self.calls) - 1
        attempt.update(id=self.last_read_id, reason=reason,
                       requested_window_ms=[start_ms, end_ms], samples=[s0, s1])
        started = monotonic()
        try:
            observation = self.adapter.observe(
                np.ascontiguousarray(self.pcm[s0:s1], dtype=np.float32), self.sr,
                total_samples=len(self.pcm),
            )
            check_cancelled()
            # Posterior matrices belong only to the bounded cache. The retained
            # ledger has a hash and scalar evidence, never another array copy.
            observed = {f.name: deepcopy(getattr(observation, f.name))
                        for f in fields(observation) if f.name != "posterior"}
            attempt.update(outcome="completed", observed=observed, frames=observation.frames)
            self.frames += observation.frames
        except GenerationCancelled:
            attempt["outcome"] = "cancelled"
            raise
        except BaseException:
            attempt["outcome"] = "failed"
            raise
        finally:
            attempt["seconds"] = monotonic() - started
        # Account resident Python containers and values, not serialized bytes.
        seen = set()
        def resident_size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            size = sys.getsizeof(value)
            if isinstance(value, dict):
                size += sum(resident_size(k) + resident_size(v) for k, v in value.items())
            elif isinstance(value, (tuple, list)):
                size += sum(resident_size(item) for item in value)
            return size
        size = resident_size(vars(observation)) + sys.getsizeof(observation) + resident_size(window) + 256
        self.observations[window] = (observation, self.last_read_id, size)
        self.cache_bytes += size
        while self.observations and self.cache_bytes > self.cache_limit_bytes:
            key = next(iter(self.observations))
            self.cache_bytes -= self.observations.pop(key)[2]
            self.cache_evictions += 1
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)
        return observation


    def diagnostics(self):
        result = super().diagnostics()
        result["model_passes"] = len(self.calls)
        result["emission_passes"] = len(self.calls) if self.adapter.metric == "ctc_greedy_probability" else 0
        return result
