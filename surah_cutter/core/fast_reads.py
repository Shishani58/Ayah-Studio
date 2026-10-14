"""Guard and account actual local acoustic inference in FAST."""

from __future__ import annotations

from math import isfinite

import numpy as np

from surah_cutter.core import align
from surah_cutter.core.cancellation import GenerationCancelled, check_cancelled


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
        self.cache_limit_bytes = None  # Legacy behavior; bounded experimental gateway sets its own limit.
        self.cache_bytes = 0
        self.peak_cache_bytes = 0
        self.cache_hits = 0
        self.cache_evictions = 0

    def read(self, start_ms, end_ms):
        check_cancelled()
        window = self._window(start_ms, end_ms)
        if window is None:
            return None
        s0, s1 = window
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
                        self.cache_hits += 1
                        return emis[f0:f1], c0 * 1000.0 / self.sr + f0 * stride, stride, c0 * 1000.0 / self.sr + f1 * stride
        allowed = self._permit(s0, s1, reason)
        if allowed is None:
            return None
        if self.log_event:
            self.log_event(f"FAST: вызов акустической модели {s0 / self.sr:.2f}-{s1 / self.sr:.2f} с")
        pcm = np.ascontiguousarray(self.pcm[s0:s1], dtype=np.float32)
        check_cancelled()
        attempt = self._begin_read(s0, s1, allowed, phase="acoustic-local")
        try:
            emis = align._emissions(self.bundle, pcm, np)
            check_cancelled()
            stride = align._stride_ms(s1 - s0, self.sr, len(emis))
        except GenerationCancelled:
            attempt["outcome"] = "cancelled"
            raise
        except BaseException:
            attempt["outcome"] = "failed"
            raise
        self.cache.append((s0, s1, emis, stride))
        # Legacy cache behavior stays unchanged until the migration gates pass.
        self.cache_bytes += emis.nbytes
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)
        self.frames += len(emis)
        attempt.update(frames=len(emis), outcome="completed")
        return emis, s0 * 1000.0 / self.sr, stride, s1 * 1000.0 / self.sr

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


class LocalWhisperReads(LocalAcousticReads):
    """Independent short ASR probes with the same audio budget as CTC FAST."""

    def __init__(self, pcm, sr, emitter, *, log_event=None):
        super().__init__(pcm, sr, None, log_event=log_event)
        self.emitter = emitter
        self.results = {}

    def read(self, start_ms, end_ms, *, reason):
        from time import monotonic

        check_cancelled()
        window = self._window(start_ms, end_ms)
        if window is None:
            return None
        s0, s1 = window
        if (s0, s1) in self.results:
            self.cache_hits += 1
            return self.results[s0, s1]
        allowed = self._permit(s0, s1)
        if allowed is None:
            return None
        check_cancelled()
        started = monotonic()
        attempt = self._begin_read(s0, s1, allowed, phase="whisper-local-asr")
        attempt["reason"] = reason
        try:
            result = self.emitter.recognize_local(self.pcm[s0:s1], total_samples=len(self.pcm))
            check_cancelled()
            attempt.update(result)
        except GenerationCancelled:
            attempt["outcome"] = "cancelled"
            raise
        except BaseException:
            attempt["outcome"] = "failed"
            raise
        finally:
            attempt["seconds"] = monotonic() - started
        frames = (s1 - s0 + 319) // 320
        self.frames += frames
        attempt.update(frames=frames, outcome="completed")
        self.results[s0, s1] = result
        if self.log_event:
            self.log_event(f"FAST Tarteel: {s0 / self.sr:.2f}-{s1 / self.sr:.2f} с, {result['text']}")
        return result

    def diagnostics(self):
        return {**super().diagnostics(), "emission_passes": 0, "model_passes": len(self.calls)}


def align_fast_frame_bounds(start, end, samples, frames):
    return max(0, round(start * frames / samples)), min(frames, round(end * frames / samples))


class LocalAlignmentReads(LocalAcousticReads):
    """One FAST acoustic phase: guarded probes, strict CTC and independent decode."""

    def __init__(self, pcm, sr, bundle, words, word_trs, *, log_event=None):
        super().__init__(pcm, sr, bundle, max_read_ms=6000.0, log_event=log_event)
        self.words, self.word_trs = words, word_trs
        self.inv_vocab = {value: key for key, value in bundle.vocab.items()}
        self._onset_pauses = None

    def align_slice(
        self, i0: int, i1: int, t0_ms: float, t1_ms: float, basmala: bool,
        *, check_onsets: bool = False,
    ):
        from surah_cutter.core import candidates
        from surah_cutter.core._ctc import forced_align, aligned_segments
        from surah_cutter.data import transliteration

        A = align
        pcm, sr, bundle = self.pcm, self.sr, self.bundle
        words, word_trs, total_ms = self.words, self.word_trs, self.total_ms
        min_win_ms = 800.0
        emissions_for = self.read
        if t1_ms - t0_ms < min_win_ms:
            mid = 0.5 * (t0_ms + t1_ms)
            t0_ms, t1_ms = mid - min_win_ms / 2, mid + min_win_ms / 2
        t0_ms = max(0.0, t0_ms)
        t1_ms = min(total_ms, t1_ms)
        if t1_ms - t0_ms < min_win_ms:  # near an edge: grow inward
            if t0_ms <= 0.0:
                t1_ms = min(total_ms, min_win_ms)
            else:
                t0_ms = max(0.0, total_ms - min_win_ms)
        if i1 <= i0:  # no words
            return None
        sub_words = words[i0:i1]
        sub_trs = word_trs[i0:i1]
        basmala_trs = A._basmala_trs(transliteration) if basmala else None
        targets, spans = A._build_targets(sub_words, sub_trs, bundle, basmala_trs)
        if not targets or any(end <= start for start, end in spans):
            return None
        em = emissions_for(t0_ms, t1_ms)
        if em is None:
            return None
        emis, base, stride, end = em
        nf = emis.shape[0]
        try:
            path, fscores = forced_align(
                np.ascontiguousarray(emis[None, :, :], dtype=np.float32),
                np.asarray(targets, dtype=np.int64)[None, :],
                blank=bundle.blank_id,
            )
        except ValueError:
            return None  # The caller grows/retries an insufficient window.
        path = np.asarray(path).reshape(-1)
        fscores = np.asarray(fscores).reshape(-1)
        try:
            segs = aligned_segments(path, targets, blank=bundle.blank_id)
        except ValueError:
            return None
        tspans = [(s.start, s.end) for s in segs]
        # CTC emission-lag compensation, same as the full path.
        lag = A.ALIGN_LAG_MS
        starts, ends, confs = [], [], []
        for (ts, te) in spans:
            sf = tspans[ts][0] if ts < len(tspans) else nf - 1
            ef = tspans[te - 1][1] if te > ts and te - 1 < len(tspans) else sf
            starts.append(max(0.0, base + sf * stride - lag))
            ends.append(max(0.0, base + ef * stride - lag))
            seg = fscores[sf:max(sf + 1, ef + 1)]
            confs.append(float(np.exp(seg.mean())) if seg.size else 0.0)
        onset_checks = {}
        if check_onsets and len(segs) == len(targets):
            if self._onset_pauses is None:
                self._onset_pauses = candidates.detect_energy_pauses(pcm, sr, use_ten_vad=False)
            onset_checks = A._check_ayah_onsets(
                sub_words, spans, tspans, fscores, stride, base_ms=base,
                pauses=self._onset_pauses,
            )
            for idx, word in enumerate(sub_words):
                if idx == 0 or sub_words[idx - 1].ayah != word.ayah:
                    check = onset_checks[word.ayah]
                    starts[idx] = float(check["onset_ms"])
                    confs[idx] = check["conf"]
        return {"starts": starts, "ends": ends, "confs": confs, "lag_ms": lag,
                "onset_checks": onset_checks,
                "complete": len(segs) == len(targets),
                "base": base, "end": end, "win_ms": end - base}


    def greedy_decode(self, t0_ms: float, t1_ms: float) -> str:
        bundle, inv_vocab = self.bundle, self.inv_vocab
        em = self.read(t0_ms, t1_ms)
        if em is None:
            return ""
        ids = em[0].argmax(-1)
        out: list[str] = []
        prev = -1
        for i in ids.tolist():
            if i != prev and i != bundle.blank_id:
                out.append(inv_vocab.get(int(i), ""))
            prev = i
        return "".join(out)


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
