"""Read acoustic onset evidence without rewriting timings or model scores."""

from __future__ import annotations

import json
import math
from pathlib import Path


def boundary_assessment(check, time_ms, trace):
    """Grade evidence for the stored position, not probability of a phoneme.

    Only an accepted local phrase+pause verification confirms a position.
    CTC agreement or a supported sound near a DSP edge is preliminary. None
    of these grades is a calibrated error bound or a probability of accuracy.
    """
    reason = check.get("reason")
    if reason in {"weak-onset", "late-onset-in-pause"}:
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
    score = check.get("first_token_conf")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        return None
    moved = check.get("raw_onset_ms", time_ms) != time_ms
    if moved and check.get("reason") != "split-word-local-recheck":
        score = None
    return {"first_sound": score, "word": confidence,
            "boundary_warning": check.get("reason") == "late-onset-in-pause",
            "boundary": boundary_assessment(check, time_ms, trace or {})}


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
