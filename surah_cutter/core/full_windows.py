"""Quran Phonetics FULL: pause cuts and independent single-window ONNX calls.

This module is never called by FAST or the general-purpose emitter. It preserves the
global waveform normalization and 20 ms frame grid of Quran Phonetics, trims
context logits, and leaves canonical targets and CTC alignment unchanged.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import math
import threading
import time

import numpy as np

from surah_cutter.core.cancellation import cancellation_scope, check_cancelled, current_event
from surah_cutter.core.emissions import (
    DEFAULT_CONTEXT_S, DEFAULT_WINDOW_S, EXPECTED_SR, FRAME_STRIDE_MS,
    _log_softmax, _normalize, _run,
)

_FRAME_SAMPLES = EXPECTED_SR * FRAME_STRIDE_MS // 1000
STRONG_PAUSE_MIN_MS = 300.0
FULL_WINDOW_S = 20
FULL_WORKERS = 3
FULL_MAX_WINDOW_S = 60
FULL_POLICY = "qpc-full-pause-v2-text-only"
FULL_REVIEW_POLICY = "qpc-pause-review-v1"
FULL_ONSET_POLICY = "qpc-waveform-pause-v11"


def detect_strong_pauses(pcm: np.ndarray, *, trace: dict | None = None):
    """Signal-confirmed silence, not confirmation of a canonical ayah boundary."""
    from surah_cutter.core.candidates import (
        _dsp_accepts_pause, _dsp_pause_metrics, detect_energy_pauses,
    )

    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    candidates = detect_energy_pauses(
        audio, EXPECTED_SR, min_pause_ms=STRONG_PAUSE_MIN_MS, use_ten_vad=False,
    )
    # The same recording RMS is used for every pause. Computing it inside
    # each metric call scans hours of PCM hundreds of times.
    audio_rms = float(np.sqrt(np.mean(audio * audio))) if candidates and audio.size else 1e-10
    accepted = []
    decisions = []
    for pause in candidates:
        check_cancelled()
        metrics = _dsp_pause_metrics(audio, EXPECTED_SR, pause, audio_rms=audio_rms)
        ok = pause.depth_ms >= STRONG_PAUSE_MIN_MS and _dsp_accepts_pause(metrics)
        decisions.append({"start_ms": pause.start_ms, "end_ms": pause.end_ms,
                          "center_ms": pause.center_ms, "duration_ms": pause.depth_ms,
                          "accepted": ok, "metrics": metrics})
        if ok:
            accepted.append(pause)
    if trace is not None:
        trace.update(pause_policy="strong-dsp-v1", min_pause_ms=STRONG_PAUSE_MIN_MS,
                     pause_decisions=decisions)
    return accepted


def plan_pause_windows(
    n_samples: int, pauses: list, *, window_s: float = DEFAULT_WINDOW_S,
    pause_only: bool = False,
):
    """Return contiguous sample spans and cut reasons, quantized to CTC frames.

    Prefer a pause within 50% of the desired length, with at least half a window
    on each side of an internal cut. Without one, cut at the desired length.
    The final span may be up to 1.5 windows to avoid a tiny independent tail.
    With pause_only, prefer the nearest eligible pause even beyond that range;
    without a pause keep the remainder intact. There is no hard duration cap.
    Only the last end can exceed real samples (less than one frame of padding).
    """
    if n_samples <= 0 or not math.isfinite(window_s) or window_s < .1:
        raise ValueError("audio must be nonempty and window_s must be at least 0.1")
    target = max(1, round(window_s * EXPECTED_SR / _FRAME_SAMPLES))
    total = math.ceil(n_samples / _FRAME_SAMPLES)
    minimum = max(1, target // 2)
    maximum = max(target, target * 3 // 2)
    pause_frames = []
    for pause in pauses:
        frame = round(pause.center_ms / FRAME_STRIDE_MS)
        # Quantization must not move a claimed pause cut into sounding audio.
        if (0 < frame < total
                and pause.start_ms <= frame * FRAME_STRIDE_MS <= pause.end_ms):
            pause_frames.append(frame)
    pause_frames = sorted(set(pause_frames))
    spans = []
    start = 0
    while start < total:
        check_cancelled()
        if total - start <= maximum:
            end, reason = total, "end"
        else:
            desired = start + target
            lo = bisect_left(pause_frames, start + minimum)
            limit = total - minimum if pause_only else min(start + maximum, total - minimum)
            hi = bisect_right(pause_frames, limit)
            near = bisect_left(pause_frames, desired, lo, hi)
            choices = pause_frames[max(lo, near - 1):min(hi, near + 1)]
            if choices:
                end = min(choices, key=lambda frame: (abs(frame - desired), frame))
                reason = "pause"
            else:
                end, reason = (total, "end") if pause_only else (desired, "no-pause")
        spans.append((start * _FRAME_SAMPLES, end * _FRAME_SAMPLES, reason))
        start = end
    return spans


def generate_pause_emissions(
    emitter, pcm: np.ndarray, *, window_s: float = DEFAULT_WINDOW_S,
    context_s: float = DEFAULT_CONTEXT_S, workers: int = 1, trace: dict | None = None,
    strong_pauses: bool = False,
    max_window_s: float | None = None,
) -> np.ndarray:
    """Quran Phonetics windowing; no files or DB writes.

    At most ``workers`` calls are in flight and each has batch size one. The
    shared ONNX session's intra-op thread budget must be set by the caller when
    comparing concurrency. Other model families have different framing.
    max_window_s rejects an unsafe span before inference; it never cuts speech.
    """
    from surah_cutter.core.candidates import detect_energy_pauses

    if (not isinstance(workers, int) or not 1 <= workers <= 6
            or not math.isfinite(context_s) or context_s < .02
            or not math.isclose(context_s * 1000 / FRAME_STRIDE_MS,
                                round(context_s * 1000 / FRAME_STRIDE_MS))):
        raise ValueError("workers must be 1..6; context must be positive on the 20 ms grid")
    if max_window_s is not None and (not math.isfinite(max_window_s) or max_window_s <= 0):
        raise ValueError("max_window_s must be positive and finite")
    audio = np.asarray(pcm, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be mono")
    # Validate even when the original short-clip path can be preserved.
    plan_pause_windows(len(audio), [], window_s=window_s)
    check_cancelled()
    if trace is not None:
        trace.update(window_s=window_s, context_s=context_s, workers=workers,
                     batch_size=1, dsp_s=0.0, windows=[], max_window_s=max_window_s,
                     runtime=getattr(emitter, "runtime_identity", None),
                     pause_policy="strong-dsp-v1" if strong_pauses else "energy-preferred")
    if len(audio) <= window_s * EXPECTED_SR:
        if max_window_s is not None and len(audio) > max_window_s * EXPECTED_SR:
            raise ValueError("FULL: участок без подтверждённой паузы слишком длинный")
        start = time.perf_counter()
        cpu_start = time.process_time()
        result = emitter.emissions(audio, window_s=window_s, context_s=context_s)
        check_cancelled()
        if trace is not None:
            trace.update(inference_s=time.perf_counter() - start, short_clip=True,
                         inference_cpu_s=time.process_time() - cpu_start,
                         windows=[{"start_ms": 0, "end_ms": len(audio) / 16,
                                   "cut_reason": "end", "frames": len(result)}])
        return result

    start = time.perf_counter()
    pauses = (detect_strong_pauses(audio, trace=trace) if strong_pauses else
              detect_energy_pauses(audio, EXPECTED_SR, use_ten_vad=False))
    spans = plan_pause_windows(len(audio), pauses, window_s=window_s, pause_only=strong_pauses)
    if trace is not None:
        trace["dsp_s"] = time.perf_counter() - start
        trace["short_clip"] = False
    check_cancelled()
    if max_window_s is not None:
        for left, right, _ in spans:
            if right - left > max_window_s * EXPECTED_SR:
                raise ValueError(
                    "FULL: между подтверждёнными паузами слишком длинный участок "
                    f"{left / EXPECTED_SR:.2f}–{min(right, len(audio)) / EXPECTED_SR:.2f} с "
                    f"(предел {max_window_s:g} с). Требуется проверить паузы записи."
                )
    wav = _normalize(audio) if emitter._do_normalize else np.ascontiguousarray(audio)
    context_frames = round(context_s * 1000 / FRAME_STRIDE_MS)
    context = context_frames * _FRAME_SAMPLES
    event = current_event()
    stop = threading.Event()

    def run_window(span):
        left, right, reason = span
        with cancellation_scope(event):
            check_cancelled()
            if stop.is_set():
                raise RuntimeError("FULL window processing stopped")
            real_left, real_right = max(0, left - context), min(len(wav), right + context)
            clip = np.pad(wav[real_left:real_right],
                          (real_left - (left - context), right + context - real_right))
            logits = _run(emitter.session, clip[None],
                          emitter._input_name, emitter._output_name)[0]
            frames = (right - left) // _FRAME_SAMPLES
            # wav2vec2 produces input_frames - 1; keeping a fixed central count
            # avoids cumulative seam rounding and discards all context logits.
            if len(logits) != len(clip) // _FRAME_SAMPLES - 1:
                raise ValueError("pause windows require the Quran Phonetics 20 ms frame layout")
            kept = _log_softmax(logits[context_frames:context_frames + frames])
            check_cancelled()
            return kept, {"start_ms": left / 16, "end_ms": min(right, len(wav)) / 16,
                          "read_start_ms": real_left / 16, "read_end_ms": real_right / 16,
                          "input_samples": len(clip), "frames": frames, "cut_reason": reason}

    from surah_cutter.core.progress import advance

    advance("acoustic", 0, len(spans))
    start = time.perf_counter()
    cpu_start = time.process_time()
    results = [None] * len(spans)
    intervals = [None] * len(spans)
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    iterator = iter(enumerate(spans))
    completed = 0
    try:
        for _ in range(min(workers, len(spans))):
            index, span = next(iterator)
            pending[pool.submit(run_window, span)] = index
        while pending:
            check_cancelled()
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Refill any freed slot, even while an earlier long window runs.
            # Only collection order changes; inputs and final frame order do not.
            for future in sorted(finished, key=pending.__getitem__):
                index = pending.pop(future)
                results[index], intervals[index] = future.result()
                completed += 1
                advance("acoustic", completed, len(spans))
                item = next(iterator, None)
                if item is not None:
                    index, span = item
                    pending[pool.submit(run_window, span)] = index
    finally:
        stop.set()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    result = np.concatenate(results)
    if trace is not None:
        trace.update(inference_s=time.perf_counter() - start, windows=intervals,
                     inference_cpu_s=time.process_time() - cpu_start,
                     pause_cuts=sum(s[2] == "pause" for s in spans),
                     fallback_cuts=sum(s[2] == "no-pause" for s in spans))
    return result
