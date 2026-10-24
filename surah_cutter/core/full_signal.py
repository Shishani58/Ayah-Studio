"""Refine late FULL cuts with raw PCM envelopes; no additional inference."""

import numpy as np

from surah_cutter.core.cancellation import check_cancelled


def _periodicity(samples, sr):
    """Detect a quiet voiced tail which low RMS alone could call silence."""
    signal = np.asarray(samples, dtype=np.float64)
    if signal.size < 4:
        return 0.0
    signal = signal - np.mean(signal)
    energy = np.cumsum(signal * signal)
    if not len(energy) or energy[-1] <= np.finfo(float).eps:
        return 0.0
    first, last = max(2, round(sr / 400)), min(round(sr / 75), len(signal) // 2)
    lags = np.arange(first - 1, last + 2)
    if not len(lags):
        return 0.0
    correlation = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
    denominator = np.sqrt((energy[-1] - energy[lags - 1]) * energy[len(signal) - lags - 1])
    normalized = correlation[lags] / np.maximum(denominator, 1e-30)
    # A slowly varying noise floor has large short-lag correlation without
    # a repeated period. Require a local peak, including both neighbours.
    peaks = ((normalized[1:-1] > normalized[:-2])
             & (normalized[1:-1] >= normalized[2:]))
    return float(np.max(normalized[1:-1][peaks], initial=0))


def refine_late_boundaries(checks, pcm, sr, *, trace=None):
    """Place cuts in owned quiet shelves using unsmoothed RMS and peaks.

    FULL supplies neighbouring ayah ownership. Different DSP contexts may
    expose different edges; only their common quiet shelf is eligible. Check
    it in every context, then require a quiet 40 ms window in the raw waveform.
    Never join across a brief sound or cut a supported previous ending.
    """
    from surah_cutter.core import candidates, full_pause_center

    policy = dict(frame_ms=5, quiet_window_ms=40, max_shift_ms=500, max_lead_ms=250,
                  min_pause_ms=120, min_safe_ms=40, margin_ms=20,
                  max_rms_ratio=.30, max_peak_ratio=.50, max_periodicity=.8, max_gap_ms=6000,
                  noise_multiplier=3, min_rms_floor_ratio=.03, min_peak_floor_ratio=.05,
                  voice_range_hz=[75, 400], min_word_token_conf=.5)
    if trace is not None:
        trace["full_signal_policy"] = policy
    records = trace.setdefault("full_signal_checks", []) if trace is not None else []
    for ayah in sorted(checks):
        check_cancelled()
        check, previous = checks[ayah], checks.get(ayah - 1)
        if (previous is None or check.get("repaired")
                or check.get("pause_center_reason") == "positioned"
                or previous.get("full_ctc_tail_conf", 0) < .25):
            continue
        tail, first, raw = previous.get("full_ctc_end_ms"), check.get("full_ctc_first_ms"), check.get("raw_onset_ms")
        if None in (tail, first, raw) or not 0 < first - tail <= policy["max_gap_ms"]:
            continue
        weak_head = check.get("first_token_conf", 0) < .25
        if weak_head:
            # A weak first phoneme is not proof that the whole opening word
            # is absent. Its other aligned phonemes may support ownership.
            # Only the raw-signal verifier may move this cut; keep review.
            if (check.get("pause_center_reason") != "weak-evidence"
                    or max(check.get("token_confs", []), default=0) < policy["min_word_token_conf"]):
                continue
            for prefix, extra in (("pause_local", 0), (
                    "pause_stability", full_pause_center.POSITION_POLICY["stability_extra_ms"])):
                crop_ms, decisions, _ = full_pause_center._local_pauses(
                    pcm, sr, tail, first, full_pause_center.POSITION_POLICY, extra_ms=extra)
                check[f"{prefix}_crop_ms"] = crop_ms
                check[f"{prefix}_decisions"] = decisions
        margin = policy["margin_ms"]
        shelves = sorted({(p["start_ms"], p["end_ms"])
                          for key in ("pause_local_decisions", "pause_stability_decisions")
                          for p in check.get(key, [])
                          if p.get("accepted") and p.get("owned")
                          and p["start_ms"] < min(raw, first) and p["end_ms"] > tail
                          and p["end_ms"] - p["start_ms"] >= policy["min_pause_ms"]})
        if not shelves:
            continue
        # Prefer shelves wholly supported by the speech brackets. A later
        # candidate crossing the first sound must not veto an earlier proven
        # pause (or pull the cut into that first word).
        bounded_shelves = [p for p in shelves
                           if tail <= p[0] and p[1] <= raw + margin
                           and p[1] - margin < check["onset_ms"]]
        if bounded_shelves:
            shelves = bounded_shelves
        # Different crops may trim the same pause differently. Use only the
        # intersection; never bridge a sound between disjoint quiet shelves.
        start, end = max(p[0] for p in shelves), min(p[1] for p in shelves)
        if end - start < policy["min_pause_ms"]:
            continue
        half = policy["quiet_window_ms"] / 2
        guard = max(margin, min(80, (end - start) * .22))
        safe_start = max(start + guard, end - policy["max_lead_ms"],
                         check["onset_ms"] - policy["max_shift_ms"], tail + half)
        safe_end = min(end - guard, raw - margin, first - half, check["onset_ms"] - margin)
        following = checks.get(ayah + 1)
        if (safe_end - safe_start < policy["min_safe_ms"]
                or safe_start <= previous["onset_ms"]
                or (following is not None and safe_end >= following["onset_ms"])):
            continue
        lo = max(0, int((tail - 400) // 5) * 5)
        hi = min(len(pcm) * 1000 / sr, int((first + 600) // 5) * 5)
        contexts = [check.get(k) for k in ("pause_local_crop_ms", "pause_stability_crop_ms")]
        contexts.append([lo, hi])
        if any(c is None or not c[0] <= start < end <= c[1] for c in contexts):
            continue
        metrics = [candidates._dsp_pause_metrics(
            pcm[round(a * sr / 1000):round(b * sr / 1000)], sr,
            candidates.Pause(start - a, end - a, end - start)) for a, b in contexts]
        record = dict(ayah=ayah, crop_ms=[lo, hi], pause_ms=[start, end],
                      before_ms=check["onset_ms"], safe_interval_ms=[safe_start, safe_end],
                      accepted=False, dsp_context_metrics=metrics, weak_head=weak_head,
                      pause_edges_cross_anchor=not bool(bounded_shelves))
        records.append(record)
        if not all(candidates._dsp_accepts_pause(m) for m in metrics):
            record["reason"] = "waveform-dsp-conflict"
            continue
        frame = max(1, round(sr * policy["frame_ms"] / 1000))
        crop = pcm[round(lo * sr / 1000):round(hi * sr / 1000)]
        samples = crop[:len(crop) // frame * frame].reshape(-1, frame)
        rms = np.sqrt(np.mean(samples * samples, axis=1))
        peak = np.max(np.abs(samples), axis=1)
        times = lo + np.arange(len(rms)) * frame * 1000 / sr
        sides = ((times >= tail - 250) & (times < tail + 50)) | (
            (times >= first - 200) & (times < first + 250))
        if not np.any(sides):
            record["reason"] = "missing-waveform-context"
            continue
        rms_ref, peak_ref = np.percentile(rms[sides], 75), np.percentile(peak[sides], 75)
        if min(rms_ref, peak_ref) <= 1e-8:
            record["reason"] = "missing-waveform-speech"
            continue
        interior = (times >= start + guard) & (times + policy["frame_ms"] <= end - guard)
        if not np.any(interior):
            record["reason"] = "missing-waveform-noise-floor"
            continue
        rms_floor, peak_floor = np.percentile(rms[interior], 20), np.percentile(peak[interior], 20)
        rms_limit = min(rms_ref * policy["max_rms_ratio"], max(
            rms_floor * policy["noise_multiplier"], rms_ref * policy["min_rms_floor_ratio"]))
        peak_limit = min(peak_ref * policy["max_peak_ratio"], max(
            peak_floor * policy["noise_multiplier"], peak_ref * policy["min_peak_floor_ratio"]))
        record.update(rms_limit_ratio=float(rms_limit / rms_ref), peak_limit_ratio=float(peak_limit / peak_ref))
        desired = end - min((end - start) / 2, policy["max_lead_ms"])
        proposals = times[(times >= safe_start) & (times <= safe_end)]
        proposals = sorted(proposals, key=lambda t: (abs(t - desired), t))
        for proposed in proposals:
            # No smoothing/merging: a brief syllable or click in this window
            # vetoes the cut instead of being absorbed into a quiet shelf.
            window = (times >= proposed - half) & (times < proposed + half)
            if not np.any(window) or proposed - half < start or proposed + half > end:
                continue
            rms_ratio = float(np.max(rms[window]) / rms_ref)
            peak_ratio = float(np.max(peak[window]) / peak_ref)
            if rms_ratio <= rms_limit / rms_ref and peak_ratio <= peak_limit / peak_ref:
                a = max(0, round((proposed - half - lo) * sr / 1000))
                b = min(len(crop), round((proposed + half - lo) * sr / 1000))
                periodicity = _periodicity(crop[a:b], sr)
                if periodicity > policy["max_periodicity"]:
                    record["periodic_rejections"] = record.get("periodic_rejections", 0) + 1
                    continue
                new_ms = round(proposed)
                check_cancelled()
                record.update(accepted=True, reason="waveform-pause-recheck", after_ms=new_ms,
                              quiet_window_ms=[proposed - half, proposed + half],
                              rms_ratio=rms_ratio, peak_ratio=peak_ratio, periodicity=periodicity)
                check.update(pre_signal_onset_ms=check["onset_ms"], onset_ms=new_ms, repaired=True,
                             reliable=False, reason=record["reason"], signal_pause_start_ms=start,
                             signal_pause_end_ms=end)
                break
        else:
            record["reason"] = ("voiced-tail-in-waveform" if record.get("periodic_rejections")
                                else "no-quiet-waveform-window")
