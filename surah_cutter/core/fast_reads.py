"""Guard and account actual local acoustic inference in FAST."""

from __future__ import annotations

import numpy as np

from surah_cutter.core import align


class LocalAcousticReads:
    """Cache individual probes; never assemble a whole recording from windows.

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
        self.cache = []
        self.intervals = []

    def read(self, start_ms, end_ms):
        s0 = max(0, round(start_ms * self.sr / 1000.0))
        s1 = min(len(self.pcm), round(end_ms * self.sr / 1000.0))
        duration = (s1 - s0) * 1000.0 / self.sr
        reason = None
        if duration < 300.0:
            return None
        if duration > self.max_read_ms or s1 - s0 >= len(self.pcm):
            reason = "nonlocal_window"
        if reason is None:
            for c0, c1, emis, stride in self.cache:
                if c0 <= s0 and s1 <= c1:
                    f0, f1 = align_fast_frame_bounds(s0 - c0, s1 - c0, c1 - c0, len(emis))
                    if f1 - f0 >= 2:
                        return emis[f0:f1], c0 * 1000.0 / self.sr + f0 * stride, stride, c0 * 1000.0 / self.sr + f1 * stride
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
        if self.log_event:
            self.log_event(f"FAST: вызов акустической модели {s0 / self.sr:.2f}-{s1 / self.sr:.2f} с")
        emis = align._emissions(self.bundle, np.ascontiguousarray(self.pcm[s0:s1], dtype=np.float32), np)
        stride = align._stride_ms(s1 - s0, self.sr, len(emis))
        self.cache.append((s0, s1, emis, stride))
        self.intervals = merged
        self.audio_ms += duration
        self.covered_ms = covered
        self.frames += len(emis)
        self.calls.append({"phase": "acoustic-local", "window_ms": [s0 * 1000.0 / self.sr, s1 * 1000.0 / self.sr], "frames": len(emis)})
        return emis, s0 * 1000.0 / self.sr, stride, s1 * 1000.0 / self.sr

    def diagnostics(self):
        return {
            "emission_passes": len(self.calls), "emissions_frames": self.frames,
            "acoustic_reads": self.calls, "denied_acoustic_reads": self.denied,
            "audio_ms": self.audio_ms, "covered_audio_ms": self.covered_ms,
            "coverage": self.covered_ms / self.total_ms if self.total_ms else 0.0,
            "acoustic_intervals_ms": [[a * 1000.0 / self.sr, b * 1000.0 / self.sr] for a, b in self.intervals],
            "local_read_limits": {"max_read_ms": self.max_read_ms, "max_coverage": 0.65, "max_audio_ratio": 2.0},
        }


def align_fast_frame_bounds(start, end, samples, frames):
    return max(0, round(start * frames / samples)), min(frames, round(end * frames / samples))
