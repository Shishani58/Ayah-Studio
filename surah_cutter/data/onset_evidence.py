"""Read acoustic onset evidence without rewriting timings or model scores."""

from __future__ import annotations

import json
import math
from pathlib import Path

from surah_cutter.data.timing_accuracy import timing_accuracy


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _boundary_context(check, trace):
    """Adjacent canonical owners from this alignment, never another run."""
    ayah = check.get("ayah")
    if type(ayah) is not int:
        return None, None
    checks = trace.get("ayahs", [])
    previous = [c for c in checks if c.get("ayah") == ayah - 1]
    following = [c for c in checks if c.get("ayah") == ayah + 1]
    return (previous[0] if len(previous) == 1 else None,
            following[0] if len(following) == 1 else None)


def _local_pause_assessment(check, time_ms, trace):
    """Corroborate a positioned cut using both CTC owners and local DSP.

    This is model confidence, not independent phrase verification. Recognition
    of every vowel in the first word is unnecessary for a cut inside silence.
    Recheck the stored geometry so a flag alone cannot promote a stale cut.
    """
    if check.get("pause_center_reason") != "positioned":
        return None
    previous, following = _boundary_context(check, trace)
    tail = previous.get("full_ctc_end_ms") if previous else check.get("full_intro_end_ms")
    tail_conf = (previous.get("full_ctc_tail_conf") if previous else
                 check.get("full_intro_tail_conf"))
    first = check.get("full_ctc_first_ms")
    start, end = check.get("center_pause_start_ms"), check.get("center_pause_end_ms")
    lead = check.get("pause_lead_ms")
    values = (tail, tail_conf, first, start, end, lead, check.get("first_token_conf"))
    if not all(_finite(v) for v in values):
        return {"level": "unknown", "reason": "missing-local-proof"}
    pauses = [p for p in check.get("pause_local_decisions", [])
              if p.get("accepted") is True]
    # These bounds are the shipped DSP ownership/position policy, not fitted
    # timing-error thresholds. Long silence alone is not contradictory evidence.
    supported = (
        (previous is not None or check.get("ayah") == 1)
        and .25 <= tail_conf <= 1 and .25 <= check["first_token_conf"] <= 1
        and 0 <= first - tail <= 6000
        and 0 <= start < end and end - start >= 120
        and start >= tail - 80 and -80 <= first - end <= 450
        and lead == min((end - start) / 2, 250)
        and time_ms == round(end - lead) and time_ms <= first
        and len(pauses) == 1 and pauses[0].get("owned") is True
        and pauses[0].get("start_ms") == start and pauses[0].get("end_ms") == end
        and (previous is None or previous.get("onset_ms", time_ms) < time_ms)
        and (following is None or time_ms < following.get("onset_ms", time_ms))
    )
    return {"level": "high" if supported else "low",
            "reason": "local-ctc-pause" if supported else "boundary-context-conflict"}


def _contextual_onset(check, time_ms, trace):
    """Support a weak first phoneme only with nearby text AND temporal context."""
    previous, following = _boundary_context(check, trace)
    if previous is None or following is None or check.get("repaired"):
        return None
    tail, first = previous.get("full_ctc_end_ms"), check.get("full_ctc_first_ms")
    tail_conf = previous.get("full_ctc_tail_conf")
    raw, lag = check.get("raw_onset_ms"), trace.get("align_lag_ms")
    if not all(_finite(v) for v in (tail, first, tail_conf, raw, lag)):
        return None
    tokens = check.get("token_confs", [])[:3]
    # A distant strong suffix cannot rescue an unaligned opening.
    opening = any(_finite(v) and .45 <= v <= 1 for v in tokens)
    consistent = (opening and .25 <= tail_conf <= 1 and 0 <= lag <= 450
            and time_ms == raw and abs(first - lag - time_ms) <= 1
            and 0 <= first - tail <= 6000 and time_ms >= tail - 80
            and previous.get("onset_ms", time_ms) < time_ms
            and time_ms < following.get("onset_ms", time_ms))
    if not consistent:
        return None
    # A weak phoneme can sit next to a clear breath. Match both owners; an
    # unrelated nearby global pause must not rescue a misplaced word prefix.
    pauses = [p for p in trace.get("full_windows", {}).get("pause_decisions", [])
              if p.get("accepted") is True
              and all(_finite(p.get(k)) for k in ("start_ms", "end_ms"))
              and 0 <= p["start_ms"] < p["end_ms"]
              and p["start_ms"] >= tail - 80 and -80 <= first - p["end_ms"] <= 450
              and p["start_ms"] <= time_ms <= p["end_ms"] + 80]
    if len(pauses) == 1:
        return "contextual-pause"
    if first - tail <= 1000:
        return "contextual-onset"
    return None


def boundary_assessment(check, time_ms, trace):
    """Grade evidence for the stored position, not probability of a phoneme.

    Combine local pause ownership, neighboring speech, placement and opening
    support. None of these grades is a calibrated probability of accuracy.
    """
    reason = check.get("reason")
    if reason == "late-onset-in-pause":
        return {"level": "low", "reason": reason}
    if check.get("repaired"):
        for read in trace.get("full_boundary_reads", []):
            if (read.get("ayah") == check.get("ayah") and read.get("accepted") is True
                    and read.get("outcome") == "completed" and read.get("after_ms") == time_ms
                    and read.get("reason") == reason
                    and reason in {"split-word-local-recheck", "split-word-muaalem-proof"}):
                return {"level": "high", "reason": "local-phrase-and-pause"}
        if reason in {"weak-prefix-across-pause", "weak-prefix-across-local-pause"}:
            return {"level": "medium", "reason": "pause-rechecked"}
        return {"level": "unknown", "reason": "missing-local-proof"}
    local = _local_pause_assessment(check, time_ms, trace)
    if local is not None:
        return local
    if check.get("pause_center_reason") in {"ambiguous-pause", "order-conflict", "unbounded-gap"}:
        return {"level": "low", "reason": "boundary-context-conflict"}
    context = _contextual_onset(check, time_ms, trace)
    if context:
        return {"level": "medium", "reason": context}
    if reason == "weak-onset":
        return {"level": "low", "reason": reason}
    if check.get("reliable") is True and reason == "ctc":
        return {"level": "medium", "reason": "alignment-only"}
    score = check.get("first_token_conf")
    if type(score) in (int, float) and math.isfinite(score) and score >= .25:
        # Independent DSP corroborates a nearby speech edge, not the ayah owner.
        edges = [p for p in trace.get("full_windows", {}).get("pause_decisions", [])
                 if p.get("accepted") is True and type(p.get("end_ms")) in (int, float)
                 and abs(time_ms - p["end_ms"]) <= 250]
        if len(edges) == 1:
            return {"level": "medium", "reason": "pause-edge-only"}
    return {"level": "unknown", "reason": "insufficient-timing-evidence"}


def onset_evidence(check, time_ms, confidence, *, trace=None):
    """Attach a position assessment and separate recognition diagnostics.

    A moved boundary cannot inherit the old sound's probability. Only the
    local phrase verifier explicitly replaces that probability at the new time.
    These values are acoustic scores, never calibrated timing probabilities.
    """
    if not isinstance(check, dict) or check.get("onset_ms") != time_ms:
        return None
    if check.get("conf") != confidence:
        return None
    boundary = boundary_assessment(check, time_ms, trace or {})
    accuracy = timing_accuracy(check, time_ms, trace or {}, boundary)
    score = check.get("first_token_conf")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        return None
    moved = check.get("raw_onset_ms", time_ms) != time_ms
    if moved and check.get("reason") != "split-word-local-recheck":
        score = None
    return {"first_sound": score, "word": confidence,
            "boundary_warning": check.get("reason") == "late-onset-in-pause",
            "boundary": boundary,
            "timing_accuracy": accuracy}


def saved_onset_evidence(db_path, surah, timings, reviews, audio_signature):
    """Use only evidence matching this recording and the persisted row.

    Latest logs suffice normally. Immutable snapshots cover mixed manual/auto
    runs and an overwritten latest log. Missing or stale evidence stays unknown.
    """
    if db_path is None or audio_signature is None:
        return {}
    folder = Path(db_path).with_name(f"{Path(db_path).stem}.logs")
    if folder.is_symlink():
        return {}
    pending = {a for a, rv in reviews.items() if rv.get("source") != "manual"
               and rv.get("status") != "edited" and rv.get("confidence") is not None}
    if not pending:
        return {}
    result = {}
    conflicts = set()
    paths = list(folder.glob(f"sura_{surah:03d}_*.json"))
    history = folder / "runs"
    if not history.is_symlink():
        paths += list(history.glob(f"sura_{surah:03d}_*.json"))
    for path in sorted(paths, key=lambda p: p.name):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("surah") != surah:
                continue
            trace = payload.get("trace", {})
            if not isinstance(trace, dict):
                continue
            signature = payload.get("audio_signature", trace.get("audio_signature"))
            if signature != audio_signature:
                continue
            stored = {r["ayah"]: r for r in payload.get("persisted_review", [])}
            for check in trace.get("ayahs", []):
                a = check.get("ayah")
                if a not in pending:
                    continue
                rv, saved = reviews[a], stored.get(a, {})
                if (saved.get("source") != rv.get("source")
                        or saved.get("time") != timings.get(a)
                        or saved.get("confidence") != rv.get("confidence")):
                    continue
                evidence = onset_evidence(check, timings[a], rv["confidence"], trace=trace)
                if evidence is not None:
                    if a in result and result[a] != evidence:
                        conflicts.add(a)
                    result[a] = evidence
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return {a: evidence for a, evidence in result.items() if a not in conflicts}
