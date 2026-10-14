"""Tarteel Tiny/Base FAST: DSP pauses, independent local phrases, monotone anchors.

No whole-recording recognition, teacher-forced FULL, or tiled fallback.
Missing evidence is a structural estimate with confidence zero, for review.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import unicodedata

import numpy as np

from surah_cutter.core import candidates, model_store
from surah_cutter.core.cancellation import check_cancelled
from surah_cutter.core.fast_reads import LocalWhisperReads
from surah_cutter.core.model_runtime import inference_words

FAST_VERSION = "tarteel-pause-v1"
SETTINGS = {
    "speech_ms": [1300, 2500, 4000], "pre_roll_ms": 200,
    "min_similarity": .9, "min_margin": .12, "min_token_confidence": .6,
    "max_phrase_words": 6, "lag_ms": 0,
}


def _tokens(text):
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({
        "ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ـ": "",
    }))
    # Normalize a comparison copy only; punctuation and harakat are not words.
    text = "".join(c if unicodedata.category(c).startswith("L") or c.isspace()
                   else "" if unicodedata.category(c).startswith("M") else " " for c in text)
    return text.split()


def _phrases(words, model_id="whisper-tarteel-base"):
    by_ayah = defaultdict(list)
    flat = []
    for word, rendering in zip(words, inference_words(words, model_id)):
        units = _tokens(rendering)
        by_ayah[word.ayah].extend(units)
        flat.extend(units)
    counts = {n: Counter(tuple(flat[i:i+n]) for i in range(len(flat)-n+1))
              for n in range(1, SETTINGS["max_phrase_words"] + 1)}
    result = {}
    for ayah, units in by_ayah.items():
        maximum = min(len(units), SETTINGS["max_phrase_words"])
        length = min(2, maximum)
        while length < maximum and counts[length][tuple(units[:length])] > 1:
            length += 1
        result[ayah] = {"tokens": units[:length],
                        "ambiguous": counts[length][tuple(units[:length])] > 1}
    return result


def _rank(result, phrases):
    """Compare only the start of independent ASR with the known opening phrase."""
    observed = _tokens(result["text"])
    ranked = []
    for ayah, phrase in phrases.items():
        target = phrase["tokens"]
        expected = " ".join(target)
        text = " ".join(observed[:len(target)])
        prefix = 0
        for a, b in zip(expected, text):
            if a != b:
                break
            prefix += 1
        complete = 0
        for a, b in zip(target, observed):
            if a != b:
                break
            complete += 1
        similarity = SequenceMatcher(None, expected, text, autojunk=False).ratio() if prefix >= 3 else 0.0
        enough = (complete >= len(target) and (complete >= 2 or len(expected) >= 5))
        ranked.append({"ayah": ayah, "similarity": similarity, "prefix_chars": prefix,
                       "complete_words": complete, "enough": enough,
                       "ambiguous": phrase["ambiguous"]})
    ranked.sort(key=lambda x: (-x["similarity"], -x["prefix_chars"], x["ayah"]))
    best = dict(ranked[0])
    best["margin"] = best["similarity"] - (ranked[1]["similarity"] if len(ranked) > 1 else 0.0)
    best["confirmed"] = (best["enough"] and not best["ambiguous"] and not result["truncated"]
                         and best["similarity"] >= SETTINGS["min_similarity"]
                         and best["margin"] >= SETTINGS["min_margin"]
                         and result["token_confidence"] >= SETTINGS["min_token_confidence"])
    best["alternatives"] = ranked[:3]
    return best


def align_surah_fast_whisper(pcm, sr, words, *, bundle, surah=None, trace=None,
                             log_event=None, ayah_progress=None, pause_decisions=None):
    from surah_cutter.core.align_fast import (
        AnchorHypothesis, _enforce_strictly_increasing_starts, _select_monotonic_anchor_skeleton,
    )

    if sr != 16000 or getattr(bundle.emitter, "model_id", None) not in ("whisper-tarteel-tiny", "whisper-tarteel-base"):
        raise ValueError("Локальный Whisper FAST доступен только для Tarteel Tiny/Base, 16 kHz.")
    if not words:
        return []
    surah = surah or words[0].surah
    total_ms = len(pcm) * 1000.0 / sr
    phrases = _phrases(words, bundle.emitter.model_id)
    ayahs = list(phrases)
    if total_ms < len(ayahs) + 1:
        raise ValueError("Недостаточно аудио для порядка аятов.")
    # All DSP/VAD evidence is acquired before any text model read.
    decisions = (pause_decisions if pause_decisions is not None
                 else candidates.pause_pipeline_decisions(pcm, sr, surah=surah))
    accepted = sorted((d for d in decisions if d.accepted), key=lambda d: d.pause.center_ms)
    acoustic = LocalWhisperReads(pcm, sr, bundle.emitter, log_event=log_event)
    observations = {}
    progress_index = {a: i for i, a in enumerate(ayahs)}

    def observe(index, speech_ms, reason):
        check_cancelled()
        pause = accepted[index].pause
        start = max(pause.start_ms, pause.end_ms - SETTINGS["pre_roll_ms"])
        end = min(total_ms, pause.end_ms + speech_ms)
        result = acoustic.read(start, end, reason=reason)
        if result is not None:
            best = _rank(result, phrases)
            observations[index] = {**best, "text": result["text"], "speech_ms": speech_ms,
                                   "token_confidence": result["token_confidence"],
                                   "window_ms": [start, end]}

    # Read the minimum phrase at each candidate first, preserving budget for late ayahs.
    for index in range(len(accepted)):
        observe(index, SETTINGS["speech_ms"][0], "minimal_opening_phrase")
    for speech_ms in SETTINGS["speech_ms"][1:]:
        for index, observation in list(observations.items()):
            if (not observation["confirmed"] and observation["prefix_chars"] >= 3
                    and (observation["similarity"] >= .55 or observation["complete_words"] >= 1)):
                reason = "ambiguous_prefix" if observation["ambiguous"] or observation["margin"] < .12 else "incomplete_prefix"
                observe(index, speech_ms, reason)

    hypotheses = []
    for index, observation in observations.items():
        if observation["confirmed"]:
            hypotheses.append(AnchorHypothesis(
                progress_index[observation["ayah"]], observation["ayah"], index,
                accepted[index].pause.center_ms, observation["similarity"],
                observation["similarity"] - observation["margin"], observation["margin"],
                observation["prefix_chars"], 0.0,
            ))
    chain = _select_monotonic_anchor_skeleton(hypotheses)
    anchors = {h.ayah: h for h in chain}
    fractions = candidates.expected_boundary_fractions(surah)
    intro = next((d.pause.center_ms for d in accepted
                  if d.reason.startswith("intro-vad")), None)
    origin = intro if intro is not None else (accepted[0].pause.center_ms
                                             if accepted and surah not in (1, 9) else 0.0)
    if chain and chain[0].ayah != ayahs[0] and origin >= chain[0].pause_ms:
        # The earliest pause can belong to ayah 2+, not to the basmala.
        # An unsupported intro prior must not displace an acoustic anchor.
        origin = 0.0
    knots = {0.0: float(origin), 1.0: total_ms}
    knots.update({fractions[h.ayah]: h.pause_ms for h in chain})
    points = sorted(knots.items())
    estimates = np.interp([fractions[a] for a in ayahs], [p[0] for p in points], [p[1] for p in points])
    raw = []
    for ayah, estimate in zip(ayahs, estimates):
        anchor = anchors.get(ayah)
        confidence = 0.0
        if anchor is not None and accepted[anchor.pause_index].reason != "weak-dsp":
            confidence = min(.95, observations[anchor.pause_index]["token_confidence"], anchor.similarity)
        raw.append((ayah, int(round(estimate)), confidence))
    rows, repairs = _enforce_strictly_increasing_starts(raw, min_gap_ms=1)
    moved = {r["ayah"] for r in repairs}
    review = [a for a, _, confidence in rows if confidence == 0.0]
    matches = []
    for ayah, start, confidence in rows:
        anchor = anchors.get(ayah) if ayah not in moved else None
        match = {"ayah": ayah, "from_pause": anchor is not None,
                 "match_action": "confirmed_local_phrase" if anchor else "unresolved_current",
                 "boundary_evidence": "whisper_local_phrase" if anchor else "unresolved_local",
                 "pause_sim": anchor.similarity if anchor else None, "onset_ms": start,
                 "conf": confidence, "timing_strategy": "pause_center" if anchor else "structural_prior"}
        if anchor:
            pause = accepted[anchor.pause_index].pause
            match.update(pause_start_ms=pause.start_ms, pause_end_ms=pause.end_ms,
                         pause_index=anchor.pause_index,
                         observed=observations[anchor.pause_index]["text"])
        matches.append(match)
        if ayah_progress:
            ayah_progress(ayah)
    if trace is not None:
        trace.update({
            "model_source": model_store.source(mode="fast"), "backend": "whisper-local-asr",
            "fast_adapter": FAST_VERSION, "fast_settings": SETTINGS, "align_lag_ms": 0,
            "total_ms": total_ms, **acoustic.diagnostics(), "model_reads": acoustic.calls,
            "phase1_pauses": [{"start_ms": d.pause.start_ms, "end_ms": d.pause.end_ms,
                               "center_ms": d.pause.center_ms, "quality": "weak" if d.reason == "weak-dsp" else "trusted",
                               "reason": d.reason} for d in accepted],
            "pause_matches": matches, "pause_sequence_events": [
                {"ayah": a, "action": "unresolved_current"} for a in review],
            "local_phrase_observations": [{"pause_index": i, **o} for i, o in observations.items()],
            "review_required_ayahs": review, "monotonic_repairs": repairs,
            "ayahs": [{"ayah": a, "onset_ms": ms, "conf": c, "reliable": c > 0} for a, ms, c in rows],
            "n_pause": sum(m["from_pause"] for m in matches), "n_wasl": 0,
            "n_refine_skipped": len(review), "mean_conf": sum(c for _, _, c in rows) / len(rows),
            "drift_detected": not chain, "bounded_ctc_fallback": None,
        })
    return rows
