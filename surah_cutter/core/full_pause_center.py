"""Place QPC FULL cuts inside pauses, after all acoustic alignment is finished."""

from surah_cutter.core.cancellation import check_cancelled


def record_edges(checks, words, word_token_spans, target_spans, frame_scores,
                 stride_ms, *, base_ms=0, intro=False, targets=None, delim_id=None):
    """Retain speech evidence across trusted tiles without moving tile anchors."""
    from surah_cutter.core.align import _span_confidence

    for i, (word, (ts, te)) in enumerate(zip(words, word_token_spans)):
        if word.ayah not in checks or te <= ts or te > len(target_spans):
            continue
        check = checks[word.ayah]
        check.setdefault("full_ctc_first_ms", base_ms + target_spans[ts][0] * stride_ms)
        if i == 0 and intro and ts:
            for token in range(ts - 1, -1, -1):
                if targets is not None and targets[token] == delim_id:
                    continue
                sf, ef = target_spans[token]
                conf = _span_confidence(frame_scores, sf, ef)
                if conf >= .25:
                    check["full_intro_end_ms"] = base_ms + (ef + 1) * stride_ms
                    check["full_intro_tail_conf"] = conf
                    break
        if i + 1 < len(words) and words[i + 1].ayah == word.ayah:
            continue
        # Canonical final vowels may be silent at waqf. Use the last supported
        # sound of the final word; DSP must still prove the quiet shelf after it.
        check["full_ctc_tail_conf"] = 0.0
        for sf, ef in reversed(target_spans[ts:te]):
            conf = _span_confidence(frame_scores, sf, ef)
            if conf >= .25:
                check["full_ctc_end_ms"] = base_ms + (ef + 1) * stride_ms
                check["full_ctc_tail_conf"] = conf
                break


def position_boundaries(checks, pcm, sr, *, trace=None):
    """Locate quiet shelves locally; cap the silence preceding the next ayah.

    CTC brackets the owners, not the silence edges. A supported first sound
    cannot be discarded to reach a later intraword pause. Local phrase repairs
    keep their separate proof. No acoustic inference is performed here.
    """
    from surah_cutter.core import candidates

    policy = dict(max_lead_ms=250, min_pause_ms=120, frame_ms=10, smooth_frames=3,
                  max_gap_ms=6000, context_before_ms=800, context_after_ms=1000,
                  edge_tolerance_ms=80, max_onset_lag_ms=450)
    if trace is not None:
        trace["full_pause_position"] = policy
    ordered = sorted(checks.values(), key=lambda c: c["ayah"])
    original = [c["onset_ms"] for c in ordered]
    for i, check in enumerate(ordered):
        check_cancelled()
        check["pre_center_onset_ms"] = original[i]
        check["pause_center_reason"] = "no-pause"
        if i == 0 and "full_intro_end_ms" not in check:
            check["pause_center_reason"] = "first-ayah"
            continue
        if check.get("repaired"):
            # Local phrase verification may have replaced a bad CTC opening.
            # Its original spans cannot corroborate or disprove that new cut.
            check["pause_center_reason"] = "existing-recheck"
            continue
        previous = ordered[i - 1] if i else None
        tail = previous.get("full_ctc_end_ms") if previous else check.get("full_intro_end_ms")
        tail_conf = (previous.get("full_ctc_tail_conf", 0) if previous else
                     check.get("full_intro_tail_conf", 0))
        if (check.get("first_token_conf", 0) < .25
                or tail_conf < .25 or tail is None or "full_ctc_first_ms" not in check):
            check["pause_center_reason"] = "weak-evidence"
            continue
        first = check["full_ctc_first_ms"]
        if not 0 <= first - tail <= policy["max_gap_ms"]:
            check.update(pause_center_reason="unbounded-gap", reliable=False)
            continue
        # Align crops to the DSP grid so rounding cannot drift between tiles.
        lo = max(0, int((tail - policy["context_before_ms"]) // 10) * 10)
        hi = min(len(pcm) * 1000 / sr, first + policy["context_after_ms"])
        crop = pcm[round(lo * sr / 1000):round(hi * sr / 1000)]
        check["pause_local_crop_ms"] = [lo, hi]
        local = candidates.detect_energy_pauses(
            crop, sr, min_pause_ms=policy["min_pause_ms"], frame_ms=policy["frame_ms"],
            smooth_frames=policy["smooth_frames"], use_ten_vad=False)
        decisions = []
        supported = []
        for local_pause in local:
            p = candidates.Pause(lo + local_pause.start_ms, lo + local_pause.end_ms,
                                 local_pause.end_ms - local_pause.start_ms)
            metrics = candidates._dsp_pause_metrics(crop, sr, local_pause)
            owned = (p.start_ms >= tail - policy["edge_tolerance_ms"]
                     and -policy["edge_tolerance_ms"] <= first - p.end_ms <= policy["max_onset_lag_ms"])
            accepted = owned and candidates._dsp_accepts_pause(metrics)
            decisions.append(dict(start_ms=p.start_ms, end_ms=p.end_ms,
                                  owned=owned, accepted=accepted, **metrics))
            if accepted:
                supported.append(p)
        check["pause_local_decisions"] = decisions
        if len(supported) != 1:
            if local:
                check["pause_center_reason"] = "ambiguous-pause" if supported else "unproven-pause"
                check["reliable"] = False
            continue
        pause = supported[0]
        lead = min((pause.end_ms - pause.start_ms) / 2, policy["max_lead_ms"])
        new_ms = round(pause.end_ms - lead)
        right = original[i + 1] if i + 1 < len(original) else len(pcm) * 1000 / sr
        if new_ms > first or not (previous["onset_ms"] if previous else -1) < new_ms < right:
            check.update(pause_center_reason="order-conflict", reliable=False)
            continue
        check.update(onset_ms=new_ms, pause_center_reason="positioned", pause_lead_ms=lead,
                     center_pause_start_ms=pause.start_ms, center_pause_end_ms=pause.end_ms)
