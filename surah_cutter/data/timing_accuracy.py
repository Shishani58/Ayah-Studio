"""Prioritise review from the current recording's acoustic and DSP evidence."""

from __future__ import annotations

import math

TOLERANCE_MS = 500


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def timing_accuracy(check, time_ms, trace, boundary):
    """Return a timing estimate and reproducible reasons for prioritising review.

    No reference times, cross-run predictions, model calls or DB writes enter
    this calculation. The caller has already checked row/time provenance.
    """
    result = dict(probability=None, tolerance_ms=TOLERANCE_MS,
                  target="playback-cut", calibration="unavailable",
                  review_reasons=[], cut_span_ms=None, cut_span_source=None)
    reasons = result["review_reasons"]
    if boundary.get("level") == "low":
        reasons.append("boundary-conflict")
    ayah = check.get("ayah")
    previous = [c for c in trace.get("ayahs", [])
                if type(ayah) is int and c.get("ayah") == ayah - 1]
    tail = (previous[0].get("full_ctc_end_ms") if len(previous) == 1
            else check.get("full_intro_end_ms") if ayah == 1 else None)
    first = check.get("full_ctc_first_ms")
    # Prefer the owned local DSP pause. CTC gaps can include unrecognised speech;
    # they are explicitly labelled as model gaps, never as guaranteed silence.
    pauses = [p for p in check.get("pause_local_decisions", [])
              if p.get("accepted") is True and p.get("owned") is True]
    start = end = None
    if len(pauses) == 1:
        start, end = pauses[0].get("start_ms"), pauses[0].get("end_ms")
        source = "local-pause"
    else:
        start, end, source = tail, first, "ctc-gap"
    if (all(_number(v) for v in (start, end, time_ms))
            and 0 <= start < end and start - 80 <= time_ms <= end + 80):
        span = round(max(abs(time_ms - start), abs(end - time_ms)))
        result.update(cut_span_ms=span, cut_span_source=source)
        if span > TOLERANCE_MS:
            reasons.append("wide-cut-range")
    if boundary.get("level") == "unknown":
        reasons.append("missing-proof")
    result["review_priority"] = ("suspect" if any(
        r in reasons for r in ("boundary-conflict", "wide-cut-range"))
        else "unassessed" if reasons else "supported")
    return result
