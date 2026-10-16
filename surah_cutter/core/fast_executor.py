"""Shared sequential FAST for CTC and Tarteel, with conservative local proofs.

Only this executor chooses candidates, phrases, recovery, and boundary ownership.
Adapters never receive the recording, pause list, ayah cursor, or a writer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from time import monotonic

import numpy as np

from surah_cutter.core.cancellation import GenerationCancelled, check_cancelled
from surah_cutter.core.fast_adapters import make_adapter, representation_sha, loaded_implementation_sha, arabic_units
from surah_cutter.core.fast_reads import LocalEvidenceReads

VERSION = "local-fast-v4-partial-identity-anchors"


@dataclass(frozen=True)
class Policy:
    independent_search: bool = True
    landmark_probes: int = 1
    short_word_pause_radius_ms: float = 500.
    speech_ms: tuple[int, ...] = (1300, 2500, 4000)
    noisy_speech_ms: tuple[int, ...] = (1700, 1900, 2100)
    max_candidates: int = 8
    first_search_ms: int = 15000
    cache_bytes: int = 32 * 1024 * 1024
    ctc_quality: float = .25
    asr_quality: float = .6
    max_block_ayahs: int = 2
    tempo_lower_ratio: float = .2
    tempo_upper_ratio: float = 2.5
    beam_log_margin: float = math.log(9.)
    sparse_anchors: bool = True
    max_unresolved_ayahs: int | None = None
    expansion_lookahead: int = 2
    ctc_prefix_probability: float = .9
    ctc_expansion_probability: float = .1


@dataclass(frozen=True)
class Boundary:
    ayah: int
    start_ms: int | None
    evidence_kind: str
    confidence: float
    review_reason: str | None = None
    read_id: int | None = None
    proof: dict | None = None


def plan_phrase(groups, ayah, word_ends=None):
    """Shortest complete-word opening distinct from all relevant word starts.

    Full-surah text is cheap and safe to inspect. Acoustic input remains local.
    Every normalized unit retains a canonical-word index in prepare().
    """
    target = groups[ayah]
    competitors = [(a, i) for a, units in groups.items()
                   for i in ([0] + word_ends[a][:-1] if word_ends else range(len(units)))
                   if (a, i) != (ayah, 0)]
    ends = (word_ends or {}).get(ayah, list(range(1, len(target) + 1)))
    # A complete two-letter word may authorize expansion, never confirmation
    # by itself. Otherwise start at ONE whole canonical word.
    sizes = [end for end in ends if sum(map(len, target[:end])) > 2 or end > ends[0]] or ends
    size = sizes[-1]
    for size in sizes:
        if not any(groups[a][i:i + size] == target[:size] for a, i in competitors):
            break
    phrase = target[:size]
    collisions = [(a, i) for a, i in competitors if groups[a][i:i + size] == phrase]
    return {"units": phrase, "ambiguous": bool(collisions), "collisions": collisions,
            "competitors": competitors}


def plan_tail(groups, ayah, word_ends):
    """Shortest whole-word suffix that identifies the previous canonical text."""
    units = groups[ayah]
    starts = [0] + word_ends[ayah][:-1]
    for start in reversed(starts):
        tail = units[start:]
        collisions = [(a, i) for a, text in groups.items()
                      for i in [0] + word_ends[a][:-1]
                      if (a, i) != (ayah, start) and text[i:i + len(tail)] == tail]
        if not collisions and sum(map(len, tail)) > 2:
            return {"units": tail, "ambiguous": False, "start_unit": start}
    return {"units": units, "ambiguous": True, "start_unit": 0}


def _prefix(observed, expected):
    n = 0
    for left, right in zip(observed, expected):
        if left != right:
            break
        n += 1
    return n


def _strong_continuation_conflict(units, quality, expected, prefix_size, threshold):
    """A shorter proof cannot hide strongly heard contradictory continuation.

    Missing or weak continuation is incomplete evidence, not a contradiction.
    Stop at this ayah's known end: subsequent ayahs are legitimate neighbours.
    """
    return any(actual != wanted and math.isfinite(q) and q >= threshold
               for actual, q, wanted in zip(units[prefix_size:], quality[prefix_size:],
                                            expected[prefix_size:]))


def plan_seam(groups, ayah, word_ends):
    """Minimal whole-word tail+opening distinct from every textual occurrence.

    Order is not evidence: repeated openings require actually hearing the
    adjoining canonical context. A repeated seam remains ambiguous too.
    """
    if ayah - 1 not in groups:
        return None
    previous, current = groups[ayah - 1], groups[ayah]
    starts = [0] + word_ends[ayah - 1][:-1]
    texts = {a: units + groups.get(a + 1, ()) for a, units in groups.items()}
    options = []
    for start in reversed(starts):
        tail = previous[start:]
        for end in word_ends[ayah]:
            phrase = current[:end]
            if sum(map(len, phrase)) <= 2:
                continue
            combined = tail + phrase
            collisions = [(a, i) for a, text in texts.items()
                          for i in [0] + word_ends[a][:-1]
                          if (a, i) != (ayah - 1, start) and text[i:i + len(combined)] == combined]
            if not collisions:
                options.append({"units": phrase, "tail": tail, "combined": combined,
                                "ambiguous": False, "start_unit": start})
                break
    return min(options, key=lambda p: (len(p["combined"]), len(p["units"]))) if options else None


def _beam_view(observation, phrase, threshold, margin):
    """Independent decoder alternative, with retained rival and null scores.

    This is a text-free prefix beam, not a forced Quran path. The finite beam
    margin is an experimental acceptance gate; held-out calibration is pending.
    """
    hypotheses = observation.hypotheses
    if not hypotheses or _prefix(tuple(hypotheses[0]["units"]), phrase) != len(phrase):
        return observation, {}
    best = hypotheses[0]
    rivals = [h["log_probability"] for h in hypotheses
              if _prefix(tuple(h["units"]), phrase) != len(phrase)]
    if observation.null_log_probability is not None:
        rivals.append(observation.null_log_probability)
    difference = best["log_probability"] - max(rivals, default=best["log_probability"])
    quality = best["unit_quality"][:len(phrase)]
    if difference < margin or len(quality) != len(phrase) or min(quality, default=0.) < threshold:
        return observation, {}
    selected = replace(observation, units=tuple(best["units"]), positions_ms=tuple(best["positions_ms"]),
                       unit_quality=tuple(best["unit_quality"]))
    return selected, {"decoder_hypothesis": 0, "beam_log_margin": difference,
                      "beam_required_margin": margin, "negative_log_scores": rivals}


def _expansion_evidence(observation, phrase, ends, threshold, groups=None, ayah=None, word_ends=None):
    """Specific exact acoustic anchors authorize one finite larger local crop.

    An independently decoded alternative prefix, or a complete later canonical
    word at its observed position, can survive a noisy first word. Neither
    authorizes confirmation; the complete discriminating proof is still needed.
    """
    if _prefix(observation.units, phrase) >= ends[0]:
        return "exact_first_word"
    matched = _prefix(observation.units, phrase)
    # The model can spell one canonical word as multiple independently spoken
    # words (for example vocative ya + ayyuha). A complete short spoken word is
    # a cue to resolve the remaining ambiguity, never a canonical-word proof.
    if (groups is not None and matched == 1 and len(phrase[0]) == 2 and ends[0] > 1
            and not observation.truncated and observation.unit_quality
            and math.isfinite(observation.unit_quality[0])
            and observation.unit_quality[0] >= threshold
            and not _strong_continuation_conflict(observation.units, observation.unit_quality,
                                                groups[ayah], matched, threshold)):
        return "whole_spoken_prefix_incomplete_canonical_word"
    if groups is not None and matched > 0 and sum(map(len, phrase[:matched])) >= 3:
        fragment = phrase[:matched]
        rivals = [(a, i) for a, units in groups.items() for i in [0] + word_ends[a][:-1]
                  if (a, i) != (ayah, 0) and units[i:i + matched] == fragment]
        if not rivals and min(observation.unit_quality[:matched], default=0.) >= threshold:
            return "unique_exact_fragment_incomplete_word"
    if groups is not None and observation.units and phrase and len(observation.units[0]) >= 3:
        fragment = observation.units[0]
        if phrase[0].startswith(fragment) and len(fragment) < len(phrase[0]):
            rivals = [(a, i) for a, units in groups.items() for i in [0] + word_ends[a][:-1]
                      if (a, i) != (ayah, 0) and units[i].startswith(fragment)]
            if not rivals and observation.unit_quality and observation.unit_quality[0] >= threshold:
                return "unique_exact_spelling_partial_word"
    for hypothesis in observation.hypotheses:
        if (hypothesis["log_probability"] >= observation.hypotheses[0]["log_probability"] - math.log(9.)
                and _prefix(tuple(hypothesis["units"]), phrase) >= ends[0]):
            return "independent_beam_prefix_competition"
    for start, end in zip(ends, ends[1:]):
        if (end <= len(phrase) and tuple(observation.units[start:end]) == phrase[start:end]
                and sum(map(len, phrase[start:end])) > 2
                and min(observation.unit_quality[start:end], default=0.) >= threshold):
            return "exact_later_word_noisy_opening"
    return None


def _split_opening(obs, groups, ends, ayah, threshold):
    """Confirm exact letters across a split ASR head, preserving token evidence.

    A unique complete first word can authorize finite context expansion. Proof
    additionally needs two whole canonical words, symmetric char-prefix rivals
    at every word start, and all contributing selected tokens above threshold.
    Word segmentation supplies identity only; the DSP pause supplies timing.
    """
    target = groups[ayah]
    # Only plain one-unit canonical words. Named letters need their own proof.
    if ends[ayah] != list(range(1, len(target) + 1)) or len(target) < 2 or obs.truncated:
        return None, False
    units, qualities, attrs = obs.units, obs.unit_quality, obs.attribution
    if len(units) != len(qualities) or len(attrs) != len(units):
        return None, False
    joined, first_count = '', None
    for i, unit in enumerate(units):
        joined += unit
        if joined == target[0]:
            first_count = i + 1
            break
        if not target[0].startswith(joined):
            return None, False
    if first_count is None or first_count < 2 or len(target[0]) < 3:
        return None, False
    def strong(count):
        for u, q, a in zip(units[:count], qualities[:count], attrs[:count]):
            ps = a.get('probabilities', ())
            if (not math.isfinite(q) or q < threshold or a.get('word') != u
                    or 'spoken_alias' in a or 'unsupported' in a or not ps
                    or len(a.get('selected_indices', ())) != len(ps)
                    or any(not math.isfinite(p) or not threshold <= p <= 1 for p in ps)
                    or q != min(ps)):
                return False
        return True
    if not strong(first_count):
        return None, False
    starts = [(a, i, ''.join(text[i:])) for a, text in groups.items()
              for i in [0] + ends[a][:-1] if (a, i) != (ayah, 0)]
    # No partial first-word identity can authorize scanning repeated owners.
    if any(text.startswith(target[0]) for _, _, text in starts):
        return None, False
    # At least TWO whole canonical words; exact source unit endpoints only.
    for count in range(2, len(target) + 1):
        text = ''.join(target[:count])
        if any(other.startswith(text) for _, _, other in starts):
            continue
        observed = ''
        for n, unit in enumerate(units, 1):
            observed += unit
            if observed == text and strong(n):
                return {'phrase': target[:count], 'segmented_opening': {
                    'source_count': n, 'first_source_count': first_count,
                    'source_units': list(units[:n]), 'selected_indices': sorted({i for a in attrs[:n] for i in a['selected_indices']}),
                    'competitor_count': len(starts), 'onset_supported': False}}, True
            if not text.startswith(observed):
                break
    # A strong contradictory next unit is not excused as word segmentation.
    rest = ''.join(units[first_count:])
    expected = ''.join(target[1:])
    if rest and not expected.startswith(rest):
        if any(q >= threshold for q in qualities[first_count:]):
            return None, False
    return None, True


def _noisy_opening(observation, groups, word_ends, ayah, threshold):
    """One missing onset unit can trigger a bounded, independent identity proof.

    Never manufacture the missing unit. Require the rest of the first word at
    the crop opening AND an additional whole word from at most two following
    words. Both anchors must exclude every other canonical word start, allowing
    the same single-unit omission for rivals. No forced path participates.
    Any intervening words must match too. The omitted onset supplies no timing.
    """
    units, ends = groups[ayah], word_ends[ayah]
    if len(ends) < 2 or ends[0] < 4 or observation.truncated:
        return None, False
    remainder = units[1:ends[0]]
    initial_quality = observation.unit_quality[:len(remainder)]
    if (_prefix(observation.units, remainder) != len(remainder)
            or len(initial_quality) != len(remainder)
            or any(not math.isfinite(q) or q < threshold for q in initial_quality)):
        return None, False
    rivals = []
    for owner, text in groups.items():
        for start in [0] + word_ends[owner][:-1]:
            if (owner, start) == (ayah, 0):
                continue
            # Wrong owner and internal starts get the same tolerated defect.
            if any(text[start + missing:start + missing + len(remainder)] == remainder
                   for missing in (0, 1)):
                rivals.append((owner, start))
    if rivals:
        return None, False
    for word_index in range(1, min(3, len(ends))):
        left, right = ends[word_index - 1], ends[word_index]
        anchor = units[left:right]
        if len(anchor) < 3:
            continue
        # The crop starts at the pause right edge; do not search distant repeats.
        limit = right + 2
        offsets = [i for i in range(len(remainder), min(len(observation.units), limit))
                   if observation.units[i:i + len(anchor)] == anchor]
        if len(offsets) != 1:
            continue
        offset = offsets[0]
        # No arbitrary inserted or contradictory intervening speech. The
        # tolerated defect is the initial unit only; any intervening complete
        # canonical words must themselves be independently heard in order.
        if observation.units[len(remainder):offset] != units[ends[0]:left]:
            continue
        quality = observation.unit_quality[:offset + len(anchor)]
        if len(quality) != offset + len(anchor) or any(not math.isfinite(q) or q < threshold for q in quality):
            continue
        # A complete conflicting canonical opening cannot be excused as noise.
        for owner, text in groups.items():
            first = word_ends[owner][0]
            if owner == ayah or _prefix(observation.units, text[:first]) != first:
                continue
            other = plan_phrase(groups, owner, word_ends)
            if not other["ambiguous"] and _prefix(observation.units, other["units"]) == len(other["units"]):
                return None, False
        return {"phrase": units[:right], "opening_recovery": {
            "missing_initial_units": 1, "heard_remainder": remainder,
            "additional_word": anchor, "additional_word_index": word_index,
            "additional_word_offset": offset, "rivals": rivals,
            "onset_supported": False}, "completeness": "identity_anchors_only"}, True
    return None, True


def _incomplete_speech_edge(pcm, sr, end_ms, observation, phrase, groups, word_ends, ayah, threshold):
    """DSP explains a finite crop expansion, never identifies a canonical owner."""
    if (observation.positions_ms is None or not observation.units or observation.truncated
            or len(observation.units) >= len(phrase)
            or min(observation.unit_quality, default=0.) < threshold):
        return None
    rivals = [owner for owner, units in groups.items() if owner != ayah
              and tuple(observation.units[:word_ends[owner][0]]) == units[:word_ends[owner][0]]]
    if rivals:
        return None
    edge = round(end_ms * sr / 1000)
    width, frame = round(sr * .2), max(1, round(sr * .02))
    if edge < width or edge + width > len(pcm):
        return None
    around = np.asarray(pcm[edge - width:edge + width], dtype=np.float64)
    around = around[:len(around) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(around ** 2, axis=1))
    cutoff = max(.003, float(np.percentile(rms, 70)) * .2)
    left, right = rms[:len(rms) // 2], rms[len(rms) // 2:]
    fractions = [float(np.mean(part >= cutoff)) for part in (left, right)]
    if min(fractions) < .8 or left[-1] < cutoff or right[0] < cutoff:
        return None
    return {"edge_ms": end_ms, "speech_fraction": fractions, "rms_cutoff": cutoff,
            "observed_units": len(observation.units), "required_units": len(phrase),
            "complete_competitor_owners": rivals}


def _competing_prefix(observation, phrase, groups, ayah, word_ends, adapter, read, policy, threshold):
    """Posterior proof or an explicitly unresolved canonical-vs-noise prefix.

    Same local frames, all canonical word starts, intro, blank and unrestricted
    residual mass. A 26% expected prefix cannot defeat its 74% residual; it can
    only justify a finite larger crop, never confirmation.
    """
    if observation.posterior is None or not hasattr(adapter, "vocab"):
        return observation, {}, False
    from surah_cutter.core.fast_ctc import prefix_scores, prefix_path
    if not phrase or any(unit not in adapter.vocab for unit in phrase):
        return observation, {}, False
    # Neither proof nor expansion can survive a target below BOTH gates.
    # Score it before constructing/scoring every canonical rival. This uses
    # the same local posterior and never relaxes the residual/quality checks.
    expected = prefix_scores(observation.posterior,
                             [tuple(adapter.vocab[u] for u in phrase)], adapter.blank_id)[0]
    if not math.isfinite(expected) or expected < math.log(min(
            policy.ctc_prefix_probability, policy.ctc_expansion_probability)):
        read.setdefault("prefix_rejections", []).append({
            "units": phrase, "reason": "target_below_proof_and_expansion_gates",
            "expected_log_probability": expected if math.isfinite(expected) else None,
            "proof_probability": policy.ctc_prefix_probability,
            "expansion_probability": policy.ctc_expansion_probability,
            "posterior_sha256": observation.posterior_sha256,
        })
        return observation, {}, False
    alternatives = [(ayah, 0, phrase)]
    for owner, units in groups.items():
        stream = units + groups.get(owner + 1, ())
        for offset in [0] + word_ends[owner][:-1]:
            if (owner, offset) != (ayah, 0):
                alternatives.append((owner, offset, stream[offset:offset + len(phrase)]))
    if adapter.model_id == "quran-phonetics":
        from surah_cutter.core import align
        from surah_cutter.data import transliteration
        intro = tuple(c for c in "".join(align._basmala_trs(transliteration)) if c not in ("-", "|", " "))
        alternatives.append((0, 0, intro[:len(phrase)]))
    alternatives = [(a, i, units) for a, i, units in alternatives
                    if units and all(u in adapter.vocab for u in units)]
    if not alternatives or alternatives[0][:2] != (ayah, 0):
        return observation, {}, False
    scores = prefix_scores(observation.posterior,
                           [tuple(adapter.vocab[u] for u in units) for _, _, units in alternatives], adapter.blank_id)
    expected = scores[0]
    if not math.isfinite(expected):
        return observation, {}, False
    residual = math.log(-math.expm1(expected)) if expected < 0 else -math.inf
    null = observation.null_log_probability
    canonical_rival = max(scores[1:], default=0.)
    rival = max(canonical_rival, null if null is not None else 0., residual)
    support = prefix_path(observation.posterior, [adapter.vocab[u] for u in phrase],
                          adapter.blank_id, observation.stride_ms, adapter.lag_ms)
    comparison = {"units": phrase, "expected_log_probability": expected,
                  "residual_log_probability": residual if math.isfinite(residual) else None, "null_log_probability": null,
                  "alternatives": [[a, i, score if math.isfinite(score) else None]
                                   for (a, i, _), score in zip(alternatives, scores)],
                  "posterior_sha256": observation.posterior_sha256, "support": support,
                  "required_probability": policy.ctc_prefix_probability,
                  "required_log_margin": policy.beam_log_margin}
    comparisons = read.setdefault("prefix_comparisons", [])
    comparisons.append(comparison)
    expansion = (support is not None and expected >= math.log(policy.ctc_expansion_probability)
                 and expected - max(canonical_rival, null if null is not None else 0.) >= policy.beam_log_margin
                 and min(support["unit_quality"], default=0.) >= threshold)
    if (support is None or expected < math.log(policy.ctc_prefix_probability)
            or expected - rival < policy.beam_log_margin
            or min(support["unit_quality"], default=0.) < threshold):
        return observation, {}, expansion
    selected = replace(observation, units=tuple(phrase), positions_ms=tuple(support["positions_ms"]),
                       unit_quality=tuple(support["unit_quality"]))
    return selected, {"prefix_comparison_id": len(comparisons) - 1}, expansion


def pause_geometry(pcm, sr, pause):
    """A long decaying tail is not an established quiet-interval midpoint.

    Text at the right edge cannot settle where an echo stopped at the left.
    Keep the original DSP interval, but require review of its midpoint when
    the interior quarter levels show a sustained audible decay.
    """
    left, right = int(pause.start_ms * sr / 1000), int(pause.end_ms * sr / 1000)
    local = pcm[left:right]
    frame = max(1, sr // 50)
    count = len(local) // frame
    if count < 8:
        return {"accepted": False, "reason": "unmeasurable_pause_interval"}
    levels = np.sqrt(np.mean(local[:count * frame].reshape(count, frame) ** 2, axis=1))
    quarter = max(1, count // 4)
    beginning, ending = float(np.median(levels[:quarter])), float(np.median(levels[-quarter:]))
    # Measure stationarity even for short pauses. Ordinary reverberant pauses
    # commonly decline; that is not evidence of an incorrect midpoint. Preserve
    # the existing audible 4x-decay rejection until independent listening labels
    # establish a better temporal gate, rather than fitting a new ratio to GT.
    deviations = np.concatenate((np.abs(levels[:quarter] - beginning),
                                 np.abs(levels[-quarter:] - ending)))
    variation = float(np.median(deviations)) * 1.4826
    stationarity = beginning - ending <= max(.003, 3 * variation)
    decay = beginning > max(.003, ending * 4)
    return {"accepted": not decay, "reason": "decaying_tail_midpoint_unproved" if decay else "measured_dsp_interval",
            "first_quarter_rms": beginning, "last_quarter_rms": ending,
            "within_quarter_variation": variation, "midpoint_interval_radius_ms": (right - left) * 500 / sr,
            "quiet_stationarity": stationarity, "timing_status": "uncalibrated_pause_midpoint",
            "temporal_probability": None}


def _review_weak_sparse_onsets(boundaries, reads, geometry, threshold):
    """Longer ASR context proves identity, not a previously weak pause onset.

    A complete unique prefix already heard below threshold gains no new onset
    evidence just because its probability rises in a longer crop. If DSP also
    lacks a stationary quiet interval, retain the owner for sequence recovery
    and leave its timestamp unknown. No replacement time is inferred.
    """
    initial = {}
    for read in reads:
        if ":pause:" not in read["reason"]:
            continue
        start, end = read["window_ms"]
        if start not in initial or end < initial[start]["window_ms"][1]:
            initial[start] = read
    result, reviews = [], []
    for boundary in boundaries:
        proof = boundary.proof or {}
        pause_id = proof.get("pause_id")
        if (not proof.get("sparse_anchor") or proof.get("quality_metric") != "asr_token_probability"
                or pause_id is None or geometry[pause_id].get("quiet_stationarity", False)):
            result.append(boundary)
            continue
        final_read = reads[boundary.read_id]
        first = initial.get(final_read["window_ms"][0])
        observed = first.get("observed", {}) if first else {}
        phrase = proof["phrase"]
        quality = observed.get("unit_quality", ())
        if (first and first["window_ms"][1] < final_read["window_ms"][1]
                and not observed.get("truncated") and quality and math.isfinite(quality[0])
                and quality[0] < threshold and _prefix(observed.get("units", ()), phrase) == len(phrase)):
            reason = "weak_complete_sparse_prefix_unproved_time"
            reviews.append({"ayah": boundary.ayah, "reason": reason, "initial_read_id": first["id"],
                            "initial_first_unit_quality": quality[0], "read_id": boundary.read_id,
                            "internal_anchor_ms": boundary.start_ms, "identity_proof": proof})
            boundary = replace(boundary, start_ms=None, evidence_kind="unresolved", confidence=0., review_reason=reason,
                               read_id=None, proof=None)
        result.append(boundary)
    return result, reviews


def _review_addressed_recovery_onsets(boundaries, geometry, policy):
    """Newly accessible identity anchors do not resolve a wide pause cut.

    Retain the internally validated owner/order for sparse recovery. Leave
    uncertain timestamps unknown; never infer a replacement onset from ASR.
    The original search path has no new publication restrictions.
    """
    recovered, disconnected = False, False
    result, reviews = [], []
    for boundary in boundaries:
        proof = boundary.proof or {}
        disconnected |= ("independent_region_origin_ayah" in proof or "independent_target_origin_ayah" in proof)
        recovered |= "bounded_following_origin_ayah" in proof or disconnected
        index = proof.get("pause_id")
        wide = recovered and index is not None and geometry[index]["midpoint_interval_radius_ms"] > policy.short_word_pause_radius_ms
        unstable = index is not None and not geometry[index].get("quiet_stationarity", False)
        if (recovered and boundary.evidence_kind == "pause" and index is not None
                and (wide or (disconnected and unstable))):
            reason = ("recovered_wide_midpoint_unproved" if wide else
                      "reacquired_nonstationary_midpoint_unproved")
            reviews.append({"ayah": boundary.ayah, "reason": reason, "read_id": boundary.read_id,
                            "internal_anchor_ms": boundary.start_ms, "identity_proof": proof})
            boundary = replace(boundary, start_ms=None, evidence_kind="unresolved", confidence=0., review_reason=reason,
                               read_id=None, proof=None)
        result.append(boundary)
    return result, reviews


def validate_boundaries(boundaries, ayahs, duration_ms, reads, *, groups=None, word_ends=None,
                        canonical_ends=None, threshold=0., pauses=None, policy=Policy(), pause_geometry_data=None,
                        search_only=False):
    if [b.ayah for b in boundaries] != list(ayahs):
        raise ValueError("canonical_ayah_set_mismatch")
    last = -1
    for boundary in boundaries:
        if boundary.evidence_kind == "unresolved":
            if (boundary.start_ms is not None or boundary.confidence != 0 or not boundary.review_reason
                    or boundary.read_id is not None or boundary.proof is not None):
                raise ValueError("invalid_unresolved_boundary")
            continue
        ms = boundary.start_ms
        if type(ms) is not int or not last < ms < duration_ms or ms < 0:
            raise ValueError("invalid_boundary_timeline")
        last = ms
        if not math.isfinite(boundary.confidence) or not 0 <= boundary.confidence <= 1:
            raise ValueError("invalid_boundary_quality")
        proof = boundary.proof or {}
        if boundary.review_reason or boundary.read_id is None or not 0 <= boundary.read_id < len(reads):
            raise ValueError("missing_acoustic_proof")
        read = reads[boundary.read_id]
        if read["outcome"] != "completed" or not proof.get("complete") or proof.get("ambiguous"):
            raise ValueError("incomplete_acoustic_proof")
        queries = read.get("queries", [{"reason": read.get("reason", ""),
                                       "window_ms": read["window_ms"]}])
        if "independent_target_origin_ayah" in proof:
            origin = proof["independent_target_origin_ayah"]
            attempt = proof.get("independent_target_attempt")
            target = proof.get("search_target_ayah")
            if (type(origin) is not int or not 1 <= origin < boundary.ayah
                    or type(attempt) is not int or not 1 <= attempt <= policy.max_block_ayahs
                    or type(target) is not int or not origin < target <= boundary.ayah
                    or not any(q.get("reason", "").startswith(f"ayah={target}:")
                               and q.get("window_ms") == read["window_ms"] for q in queries)):
                raise ValueError("invalid_independent_target_provenance")
        if "independent_region_origin_ayah" in proof:
            origin = proof["independent_region_origin_ayah"]
            attempt = proof.get("independent_region_attempt")
            nominations = {r["window_ms"][0] for r in reads
                           for q in r.get("queries", [{"reason": r.get("reason", "")}])
                           if ":independent_region_identity" in q.get("reason", "")}
            if (type(origin) is not int or not 1 <= origin < boundary.ayah
                    or type(attempt) is not int or not 1 <= attempt <= policy.landmark_probes
                    or not any(q.get("reason") == f"ayah={origin}:independent_region_identity"
                               and q.get("window_ms") == read["window_ms"] for q in queries)
                    or len(nominations) > policy.landmark_probes
                    or "bounded_following_origin_ayah" in proof):
                raise ValueError("invalid_independent_region_provenance")
        observed = read["observed"]
        if "prefix_comparison_id" in proof:
            if boundary.evidence_kind != "pause":
                raise ValueError("unsupported_competing_prefix_timing")
            index = proof["prefix_comparison_id"]
            comparisons = read.get("prefix_comparisons", ())
            if type(index) is not int or not 0 <= index < len(comparisons):
                raise ValueError("unknown_prefix_comparison")
            comparison = comparisons[index]
            expected = comparison["expected_log_probability"]
            alternatives = comparison.get("alternatives", ())
            numeric = [expected, comparison.get("null_log_probability"),
                       comparison.get("required_probability"), comparison.get("required_log_margin")]
            numeric += [score for _, _, score in alternatives if score is not None]
            if (any(type(value) not in (float, int) or not math.isfinite(value) for value in numeric)
                    or not alternatives or alternatives[0] != [boundary.ayah, 0, expected]
                    or expected > 0 or comparison["null_log_probability"] > 0
                    or any(score is not None and score > 0 for _, _, score in alternatives)
                    or comparison["required_probability"] != policy.ctc_prefix_probability
                    or comparison["required_log_margin"] != policy.beam_log_margin):
                raise ValueError("invalid_prefix_comparison")
            residual = math.log(-math.expm1(expected)) if expected < 0 else -math.inf
            rivals = [score for _, _, score in comparison["alternatives"][1:] if score is not None]
            rivals += [comparison["null_log_probability"], residual]
            support = comparison["support"]
            if (not support or tuple(comparison["units"]) != tuple(proof["phrase"])
                    or comparison["posterior_sha256"] != observed.get("posterior_sha256")
                    or comparison["residual_log_probability"] != (residual if math.isfinite(residual) else None)
                    or expected < math.log(comparison["required_probability"])
                    or expected - max(rivals) < comparison["required_log_margin"]):
                raise ValueError("unexcluded_prefix_residual")
            observed = {**observed, "units": comparison["units"], **support}
        if "decoder_hypothesis" in proof:
            hypothesis = proof["decoder_hypothesis"]
            hypotheses = observed.get("hypotheses", ())
            if hypothesis != 0 or not hypotheses:
                raise ValueError("unknown_independent_hypothesis")
            selected = hypotheses[0]
            phrase = tuple(proof.get("phrase", ()))
            rivals = [h["log_probability"] for h in hypotheses
                      if _prefix(tuple(h["units"]), phrase) != len(phrase)]
            if observed.get("null_log_probability") is not None:
                rivals.append(observed["null_log_probability"])
            margin = selected["log_probability"] - max(rivals, default=selected["log_probability"])
            if (not math.isfinite(margin) or margin < proof["beam_required_margin"]
                    or proof["beam_required_margin"] != policy.beam_log_margin
                    or margin != proof["beam_log_margin"] or rivals != proof["negative_log_scores"]):
                raise ValueError("unexcluded_decoder_competitor")
            observed = {**observed, **selected}
        offset = proof.get("offset", 0)
        phrase = tuple(proof.get("phrase", ()))
        if "bounded_following_origin_ayah" in proof:
            origin = proof["bounded_following_origin_ayah"]
            if type(origin) is not int or origin < 1 or not 0 <= boundary.ayah - origin <= policy.expansion_lookahead:
                raise ValueError("unbounded_following_owner")
        recovery = proof.get("opening_recovery")
        segmented = proof.get("segmented_opening")
        if segmented:
            from surah_cutter.core.fast_adapters import Observation
            if (boundary.evidence_kind != "pause" or groups is None or word_ends is None
                    or offset != 0 or recovery or proof.get("contextual_owner")
                    or "prefix_comparison_id" in proof or "decoder_hypothesis" in proof):
                raise ValueError("unsupported_segmented_proof")
            decoded = Observation(tuple(observed["units"]), None, observed["quality"],
                observed["truncated"], unit_quality=tuple(observed["unit_quality"]),
                attribution=tuple(observed.get("attribution", ())))
            expected, _ = _split_opening(decoded, groups, word_ends, boundary.ayah, threshold)
            if not expected or expected["segmented_opening"] != segmented or expected["phrase"] != phrase:
                raise ValueError("unheard_segmented_phrase")
            quality = observed["unit_quality"][:segmented["source_count"]]
        elif recovery:
            from surah_cutter.core.fast_adapters import Observation
            if (boundary.evidence_kind != "pause" or groups is None or word_ends is None
                    or offset != 0 or "prefix_comparison_id" in proof or "decoder_hypothesis" in proof
                    or proof.get("contextual_owner")):
                raise ValueError("unsupported_noisy_opening_proof")
            decoded = Observation(tuple(observed["units"]), None, observed["quality"],
                                  observed["truncated"], unit_quality=tuple(observed["unit_quality"]))
            expected, _ = _noisy_opening(decoded, groups, word_ends, boundary.ayah, threshold)
            if (not expected or expected["opening_recovery"] != recovery or expected["phrase"] != phrase
                    or proof.get("completeness") != "identity_anchors_only"):
                raise ValueError("unheard_noisy_opening_identity")
            i = recovery["additional_word_offset"]
            quality = observed["unit_quality"][:i + len(recovery["additional_word"])]
        elif not phrase or tuple(observed["units"][offset:offset + len(phrase)]) != phrase:
            raise ValueError("unheard_discriminating_phrase")
        else:
            quality = observed.get("unit_quality", ())[offset:offset + len(phrase)]
        if ((not recovery and not segmented and len(quality) != len(phrase)) or not quality or any(not math.isfinite(q) for q in quality)
                or min(quality) < threshold or boundary.confidence > min(quality)):
            raise ValueError("weak_discriminating_unit")
        if groups is not None:
            planned = plan_phrase(groups, boundary.ayah, word_ends)
            if proof.get("contextual_owner"):
                seam = plan_seam(groups, boundary.ayah, word_ends)
                if not seam or phrase != seam["units"] or tuple(proof.get("previous_tail", ())) != seam["tail"]:
                    raise ValueError("ambiguous_contextual_owner")
                if _strong_continuation_conflict(observed["units"][offset:], observed["unit_quality"][offset:],
                                                 groups[boundary.ayah], len(phrase), threshold):
                    raise ValueError("contradictory_contextual_continuation")
            elif not recovery and not segmented and (phrase != planned["units"] or planned["ambiguous"]):
                raise ValueError("non_distinguishing_phrase")
            if "spoken_owner_units" in proof:
                if canonical_ends is None:
                    raise ValueError("missing_canonical_spoken_owner")
                owner_units = tuple(groups[boundary.ayah][:canonical_ends[boundary.ayah][0]])
                if tuple(proof["spoken_owner_units"]) != owner_units:
                    raise ValueError("mutated_canonical_spoken_owner")
                for i, (expected, actual) in enumerate(zip(owner_units, observed["units"][offset:])):
                    if expected != actual and observed["unit_quality"][offset + i] >= threshold:
                        raise ValueError("contradictory_spoken_continuation")
            for attribution in observed.get("attribution", ())[offset:offset + len(phrase)]:
                if "spoken_alias" in attribution:
                    alias = tuple(attribution["source_units"])
                    if tuple(groups[boundary.ayah][:len(alias)]) != alias:
                        raise ValueError("contradictory_spoken_letter_alias")
        if observed["truncated"] or not proof.get("competitors_checked"):
            raise ValueError("unchecked_acoustic_proof")
        root = proof.get("search_region_root")
        if root is not None:
            if (not isinstance(root, dict) or not isinstance(root.get("proof"), dict)
                    or "search_region_root" in root["proof"]
                    or not ({"independent_region_origin_ayah", "independent_target_origin_ayah"} & root["proof"].keys())
                    or type(root.get("ayah")) is not int or root["ayah"] > boundary.ayah
                    or type(root.get("start_ms")) is not int or root["start_ms"] > ms):
                raise ValueError("invalid_search_region_root")
            validate_boundaries([Boundary(**root)], [root["ayah"]], duration_ms, reads,
                                groups=groups, word_ends=word_ends, canonical_ends=canonical_ends,
                                threshold=threshold, pauses=pauses, policy=policy,
                                pause_geometry_data=pause_geometry_data, search_only=True)
        if boundary.evidence_kind == "pause":
            left, right = proof["pause_ms"]
            if not left <= right or ms != round((left + right) / 2):
                raise ValueError("pause_center_mismatch")
            if pauses is not None:
                index = proof.get("pause_id")
                if type(index) is not int or not 0 <= index < len(pauses):
                    raise ValueError("unknown_pause_owner")
                pause = pauses[index]
                if [left, right] != [pause.start_ms, pause.end_ms]:
                    raise ValueError("mutated_pause_interval")
                if (pause_geometry_data is None or index >= len(pause_geometry_data)
                        or (not search_only and not pause_geometry_data[index]["accepted"])):
                    raise ValueError("unproved_pause_geometry")
            if not search_only and (root is not None or "independent_region_origin_ayah" in proof
                                    or "independent_target_origin_ayah" in proof):
                index = proof.get("pause_id")
                if (type(index) is not int or pause_geometry_data is None or not 0 <= index < len(pause_geometry_data)
                        or pause_geometry_data[index]["midpoint_interval_radius_ms"] > policy.short_word_pause_radius_ms
                        or not pause_geometry_data[index].get("quiet_stationarity", False)):
                    raise ValueError("unproved_reacquired_midpoint")
            r0, r1 = read["window_ms"]
            expected_left = max(left, right - 200.)
            if abs(r0 - expected_left) > .1 or r1 <= right or offset != 0:
                raise ValueError("pause_read_mismatch")
            if proof.get("contextual_owner"):
                context_id = proof.get("previous_context_read_id")
                if type(context_id) is not int or not 0 <= context_id < len(reads):
                    raise ValueError("missing_previous_context")
                context = reads[context_id]
                tail = tuple(proof["previous_tail"])
                context_obs = context.get("observed", {})
                quality = context_obs.get("unit_quality", ())[-len(tail):]
                if (context["outcome"] != "completed" or abs(context["window_ms"][1] - left) > .1
                        or context_obs.get("truncated", True)
                        or tuple(context_obs.get("units", ())[-len(tail):]) != tail
                        or len(quality) != len(tail) or min(quality) < threshold
                        or boundary.confidence > min(quality)
                        or not all(math.isfinite(q) for q in quality)):
                    raise ValueError("unheard_previous_context")
        elif boundary.evidence_kind in ("first_onset", "wasl"):
            positions = observed["positions_ms"]
            if positions is None and search_only and proof.get("identity_only") and boundary.evidence_kind == "first_onset":
                if offset != 0 or ms != round(read["window_ms"][0]):
                    raise ValueError("invalid_identity_search_location")
                continue
            if positions is None or offset >= len(positions):
                raise ValueError("unsupported_temporal_proof")
            onset = round(read["window_ms"][0] + positions[offset])
            if onset != ms or not read["window_ms"][0] <= ms < read["window_ms"][1]:
                raise ValueError("onset_proof_mismatch")
            if boundary.evidence_kind == "wasl":
                tail = tuple(proof.get("previous_tail", ()))
                if not tail or offset < len(tail) or tuple(observed["units"][offset-len(tail):offset]) != tail:
                    raise ValueError("unidentified_wasl_context")
                tail_quality = observed.get("unit_quality", ())[offset-len(tail):offset]
                if (len(tail_quality) != len(tail) or any(not math.isfinite(q) for q in tail_quality)
                        or min(tail_quality) < threshold or boundary.confidence > min(tail_quality)):
                    raise ValueError("weak_wasl_tail")
                if groups is not None and word_ends is not None:
                    previous = boundary.ayah - 1
                    planned_tail = (plan_seam(groups, boundary.ayah, word_ends) if proof.get("contextual_owner")
                                    else plan_tail(groups, previous, word_ends))
                    expected_tail = planned_tail["tail"] if proof.get("contextual_owner") else planned_tail["units"]
                    if tail != expected_tail or planned_tail["ambiguous"]:
                        raise ValueError("ambiguous_wasl_tail")
        else:
            raise ValueError("unknown_boundary_evidence")


def _complete_result(confirmed, ayahs, reason, unresolved=None):
    """Unknown boundaries carry no timestamp; identity anchors stay internal."""
    anchors = {b.ayah: b for b in confirmed}
    return [anchors.get(a) or Boundary(a, None, "unresolved", 0.,
            (unresolved or {}).get(a, reason)) for a in ayahs]


def _silero_model_path():
    from inspect import signature
    from pathlib import Path
    from surah_cutter.core import candidates
    parameter = signature(candidates.detect_silero_pauses).parameters.get("model_path")
    return Path(parameter.default if parameter else candidates.SILERO_VAD_PATH)


def dsp_model_identity():
    """Capture external VAD assets too; source/config alone cannot identify DSP."""
    from importlib import metadata, util
    from pathlib import Path
    silero = _silero_model_path()
    identity = {"silero": {"path": str(silero.resolve()), "available": silero.is_file(),
                           "sha256": hashlib.sha256(silero.read_bytes()).hexdigest() if silero.is_file() else None,
                           "provider": "CPUExecutionProvider"},
                "onnxruntime_version": metadata.version("onnxruntime"), "numpy_version": np.__version__}
    spec = util.find_spec("ten_vad")
    if spec is None:
        identity["ten_vad"] = {"available": False}
    else:
        # TEN embeds weights in its native library. Hash package source/assets,
        # excluding generated bytecode; never scan the whole site-packages tree.
        roots = spec.submodule_search_locations
        if not roots:
            raise ValueError("unidentifiable TEN-VAD package")
        assets = {}
        for location in sorted(roots):
            root = Path(location)
            for path in sorted(root.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    assets[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not assets:
            raise ValueError("unidentifiable TEN-VAD assets")
        identity["ten_vad"] = {"available": True, "assets_sha256": assets}
    return identity


def first_pause_hypotheses(pcm, sr, first, search_ms):
    """Target-independent VAD recall in the first speech region, refined by DSP.

    VAD may call a sustained syllable silence. Use the existing raw-energy edge
    refinement before admitting a pause; never use its broad VAD center directly.
    This is a single bounded DSP/VAD pass, not a text/acoustic model read.
    """
    from surah_cutter.core import candidates
    left = int(first * sr / 1000)
    clip = pcm[left:left + int(search_ms * sr / 1000)]
    frame = sr // 50
    if len(clip) < frame:
        return []
    model_path = _silero_model_path()
    if not model_path.is_file():
        return []  # optional recall unavailable; the common DSP pool remains
    vad = candidates.detect_silero_pauses(clip, sr, model_path=model_path, min_pause_ms=200)
    rms = np.array([np.sqrt(np.mean(clip[i:i + frame] ** 2)) for i in range(0, len(clip), frame)])
    noise, speech = np.percentile(rms, [15, 70])
    cutoff = float(noise + .18 * (speech - noise))
    refined = candidates._refine_pause_edges(vad, rms, frame_ms=20, cutoff=cutoff,
                                             min_pause_ms=200, pad_frames=5)
    offset = left * 1000. / sr
    return [candidates.Pause(p.start_ms + offset, p.end_ms + offset, p.depth_ms) for p in refined]


def align_surah_sequential(pcm, sr, words, *, bundle=None, adapter=None, surah=None,
                           trace=None, pause_decisions=None, policy=Policy(),
                           log_event=None, ayah_progress=None):
    from surah_cutter.core import align, candidates
    from surah_cutter.data import quran_text

    if sr != 16000 or not words or not len(pcm):
        raise ValueError("invalid_fast_input")
    if (policy.max_candidates < 1 or policy.max_block_ayahs != 2 or not policy.speech_ms
            or type(policy.landmark_probes) is not int or policy.landmark_probes not in (0, 1)):
        raise ValueError("invalid_fast_policy")
    surah = surah or words[0].surah
    canonical = quran_text.surah_words(surah)
    if [(w.ayah, w.text) for w in canonical] != [(w.ayah, w.text) for w in words]:
        raise ValueError("canonical_text_mismatch")
    ayahs = sorted({w.ayah for w in words})
    duration = len(pcm) * 1000. / sr
    events, confirmed = [], []
    trace = trace if trace is not None else {}

    def event(state, ayah=None, **details):
        check_cancelled()
        events.append({"state": state, "ayah": ayah, **details})

    from surah_cutter.core.progress import report

    report("model")
    event("PREPARE")
    adapter = adapter or make_adapter(bundle or align.default_bundle())
    rendered = adapter.represent(words)
    if len(rendered) != len(words) or any(not u or any(not s for s in u) for u in rendered):
        raise ValueError("canonical_word_map_mismatch")
    groups, word_map, word_ends, canonical_ends = {}, {}, {}, {}
    proof_ends = (adapter.proof_word_ends(words, rendered) if hasattr(adapter, "proof_word_ends")
                  else tuple((len(units),) for units in rendered))
    for index, (word, units, ends) in enumerate(zip(words, rendered, proof_ends)):
        base = len(groups.get(word.ayah, ()))
        groups.setdefault(word.ayah, []).extend(units)
        word_map.setdefault(word.ayah, []).extend([index] * len(units))
        word_ends.setdefault(word.ayah, []).extend(base + end for end in ends)
        canonical_ends.setdefault(word.ayah, []).append(base + len(units))
    groups = {a: tuple(units) for a, units in groups.items()}
    plans = {}

    def planned_opening(ayah):
        # Text is immutable within this run. Retain the small result, not a
        # separate full-surah competitor list for every ayah (long-surah RAM).
        if ayah not in plans:
            planned = plan_phrase(groups, ayah, word_ends)
            planned["competitor_count"] = len(planned.pop("competitors"))
            plans[ayah] = planned
        return plans[ayah]

    from pathlib import Path
    dsp_config = {name: value for name, value in vars(candidates).items()
                  if name.isupper() and isinstance(value, (str, bool, int, float))}
    snapshot = {"algorithm": VERSION, "adapter": adapter.version,
                "model": adapter.model_id, "revision": adapter.revision,
                "policy": asdict(policy), "lag_ms": adapter.lag_ms,
                "text_sha256": representation_sha(words, rendered),
                "pcm_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest(),
                "sample_rate": sr, "dsp_version": hashlib.sha256(Path(candidates.__file__).read_bytes()).hexdigest(),
                "dsp_config": dsp_config, "dsp_models": dsp_model_identity(),
                "decode": "ffmpeg-mono-float32-16000", "normalization": adapter.version,
                "limits": {"max_read_ms": 6000, "coverage": .65, "input_ratio": 2.}}
    snapshot["text_runtime"] = getattr(adapter, "runtime_identity", {"test_adapter": adapter.version})
    snapshot["implementation_sha256"] = loaded_implementation_sha()
    import sys
    import unicodedata
    snapshot["root_text_runtime"] = {"python": sys.version, "unicode": unicodedata.unidata_version}
    snapshot["proof_word_ends_sha256"] = hashlib.sha256(repr(proof_ends).encode()).hexdigest()
    from surah_cutter.data.alignment_identity import fast_run_key
    key = fast_run_key(snapshot)
    trace.update(mode="fast", surah=surah, ayah_count=len(ayahs), result_contract="partial-v1", align_lag_ms=adapter.lag_ms, backend=adapter.metric, fast_adapter=adapter.version,
                 fast_version=VERSION, fast_run_key=key, run_snapshot=snapshot,
                 model_source=f"{adapter.model_id}:revision={adapter.revision}:adapter={adapter.version}",
                 total_ms=duration, duration_ms=duration, transitions=events)
    report("pauses")
    event("DETECT_PAUSES")
    started = monotonic()
    decisions = pause_decisions if pause_decisions is not None else candidates.pause_pipeline_decisions(pcm, sr, surah=surah)
    pauses = [d.pause for d in decisions if d.accepted]
    pauses.sort(key=lambda p: p.start_ms)
    trace["dsp_seconds"] = monotonic() - started
    trace["phase1_pauses"] = [{"start_ms": p.start_ms, "end_ms": p.end_ms,
                               "center_ms": p.center_ms} for p in pauses]
    reads = LocalEvidenceReads(pcm, sr, adapter, cache_limit_bytes=policy.cache_bytes)
    trace["acoustic_reads"] = reads.calls
    threshold = policy.ctc_quality if adapter.metric == "ctc_greedy_probability" else policy.asr_quality
    # Only DSP determines a first speech hypothesis; no fixed basmala duration.
    frame = max(1, sr // 100)
    levels = np.array([np.sqrt(np.mean(pcm[i:i + frame] ** 2)) for i in range(0, len(pcm), frame)])
    active = np.flatnonzero(levels > max(.003, float(levels.max()) * .08))
    first = max(0., (int(active[0]) * frame * 1000. / sr - 100)) if len(active) else 0.
    initial_pause_count = len(pauses)
    supplemental = first_pause_hypotheses(pcm, sr, first, policy.first_search_ms)
    supplemental = [p for p in supplemental if not any(
        p.start_ms < old.end_ms and old.start_ms < p.end_ms for old in pauses)]
    pauses.extend(supplemental[:policy.max_candidates - 1])
    geometry = [pause_geometry(pcm, sr, p) for p in pauses]
    trace["pause_geometry"] = [{"pause_id": i, "interval_ms": [p.start_ms, p.end_ms], **g}
                               for i, (p, g) in enumerate(zip(pauses, geometry))]
    trace["dsp_seconds"] = monotonic() - started
    trace["first_onset_hypotheses"] = {"speech_start_ms": first,
        "search_region_ms": [first, min(duration, first + policy.first_search_ms)],
        "supplemental_pauses": [{"pause_id": initial_pause_count + i, "start_ms": p.start_ms, "end_ms": p.end_ms,
                                  "center_ms": p.center_ms, "source": "prefix_silero_raw_rms"}
                                 for i, p in enumerate(pauses[initial_pause_count:])]}
    # Supplied DSP candidates are an input too (the comparison harness uses
    # this seam). Different pools must not authorize the same resume identity.
    import json
    snapshot["dsp_pauses_sha256"] = hashlib.sha256(json.dumps({
        "initial": [[float(p.start_ms), float(p.end_ms)] for p in pauses[:initial_pause_count]],
        "supplemental": [[float(p.start_ms), float(p.end_ms)] for p in pauses[initial_pause_count:]],
    }, sort_keys=True, allow_nan=False).encode()).hexdigest()
    trace["fast_run_key"] = fast_run_key(snapshot)
    reason = "no_local_evidence"
    tried = set()
    unresolved = {}
    reentry_failures = 0
    reentry_origin = None
    landmark_attempts = 0
    try:
        event("LOCATE_FIRST", ayahs[0])
        report("local", 0, len(ayahs))
        for ayah in ayahs:
            report("local", ayah - 1, len(ayahs))
            if confirmed and ayah <= confirmed[-1].ayah:
                continue  # already validated and atomically published by RESOLVE_BLOCK
            event("PLAN_PHRASE", ayah)
            plan = planned_opening(ayah)
            seam = plan_seam(groups, ayah, word_ends) if plan["ambiguous"] and confirmed else None
            if seam:
                plan = {**plan, **seam, "contextual_owner": True}
            phrase = plan["units"]
            event("PLAN_PHRASE", ayah, phrase=phrase,
                  canonical_word_indices=word_map[ayah][:len(phrase)], ambiguous=plan["ambiguous"])
            previous = confirmed[-1].start_ms if confirmed else -1
            candidates_here = [("pause", i, p.start_ms, p.end_ms, p)
                               for i, p in enumerate(pauses) if p.center_ms > previous
                               and (not confirmed or i < initial_pause_count)]
            if not confirmed:
                candidates_here.append(("first_onset", -1, first, first, None))
            candidates_here.sort(key=lambda item: item[2])
            forecast = None
            following_probe_id = None
            if len(confirmed) >= 2:
                prev = confirmed[-1].ayah
                elapsed_weight = sum(len(u) for a in ayahs
                                     if confirmed[0].ayah <= a < prev for u in groups[a])
                tempo = (confirmed[-1].start_ms - confirmed[0].start_ms) / max(1, elapsed_weight)
                # After an unknown ayah the next target is a distinct search:
                # account for the skipped canonical text, without inventing its time.
                predicted_span = tempo * sum(len(u) for a in ayahs if prev <= a < ayah for u in groups[a])
                forecast = previous + predicted_span
                left = previous + predicted_span * policy.tempo_lower_ratio
                right = previous + predicted_span * policy.tempo_upper_ratio
                candidate_pool = candidates_here
                candidates_here = [c for c in candidates_here if left <= c[2] <= right]
                event("SELECT_CANDIDATE", ayah, forecast_ms=forecast, region_ms=[left, right])
                if (not adapter.timing and (not candidates_here
                        or (adapter.model_id == "whisper-tarteel-base" and len(candidates_here) < policy.max_candidates))):
                    # One measured following gap is an optional bounded
                    # hypothesis after the ordinary candidates fail. Tempo
                    # never supplies the owner proof or a timestamp.
                    lookahead_right = right + predicted_span * policy.expansion_lookahead
                    following = next((c for c in candidate_pool if right < c[2] <= lookahead_right
                                      and geometry[c[1]]["accepted"]
                                      and (not candidates_here or geometry[c[1]]["midpoint_interval_radius_ms"]
                                           <= policy.short_word_pause_radius_ms)), None)
                    if following is not None:
                        if candidates_here:
                            following_probe_id = following[1]
                        candidates_here.append(following)
                        event("SELECT_CANDIDATE", ayah, candidate=following[1],
                              reason=("empty_tempo_region_next_dsp_pause" if following_probe_id is None
                                      else "optional_after_unproved_tempo_candidates"), region_ms=[left, right],
                              lookahead_region_ms=[right, lookahead_right])
            boundary, blocked_prefix, future = None, False, None
            for kind, index, start, speech_start, pause in candidates_here[:policy.max_candidates]:
                following_probe = index == following_probe_id
                # Geometry belongs to DSP. Acoustic identity needs speech at
                # the right edge, not the entire already measured silence.
                if pause:
                    start = max(pause.start_ms, pause.end_ms - 200.)
                    if not geometry[index]["accepted"]:
                        reason = geometry[index]["reason"]
                        event("SELECT_CANDIDATE", ayah, candidate=index, declined_geometry=reason,
                              identity_search=policy.independent_search)
                        if not policy.independent_search:
                            continue
                event("SELECT_CANDIDATE", ayah, candidate=index, kind=kind)
                had_prefix = False
                soft_short_word_route = False
                short_word_base_plan, short_word_base_phrase = plan, phrase
                speech_steps = list(policy.speech_ms)
                for step, speech in enumerate(speech_steps):
                    end = min(duration, speech_start + speech)
                    if end - start > 6000:
                        reason = "insufficient_local_context"
                        break
                    identity = (ayah, index, start, end, phrase, kind)
                    if identity in tried:
                        raise RuntimeError("repeated_decision_without_new_data")
                    tried.add(identity)
                    event("READ_LOCAL", ayah, candidate=index, window_ms=[start, end],
                          reason="minimal_phrase" if step == 0 else "incomplete_distinguishing_phrase")
                    obs = reads.read(start, end, reason=f"ayah={ayah}:{kind}:" + ("minimal_phrase" if step == 0 else "incomplete_distinguishing_phrase"))
                    if obs is None:
                        reason = reads.denied[-1]["reason"] if reads.denied else "insufficient_local_context"
                        break
                    opening_id = reads.last_read_id
                    beam_proof = {}
                    if _prefix(obs.units, phrase) != len(phrase):
                        obs, beam_proof = _beam_view(obs, phrase, threshold, policy.beam_log_margin)
                    posterior_expansion = False
                    if (_prefix(obs.units, phrase) != len(phrase) and pause and not plan["ambiguous"]):
                        obs, prefix_proof, posterior_expansion = _competing_prefix(
                            obs, phrase, groups, ayah, word_ends, adapter, reads.calls[opening_id], policy, threshold)
                        beam_proof.update(prefix_proof)
                    complete = _prefix(obs.units, phrase)
                    segmented_proof, segmented_expansion = (None, False)
                    if (pause and not adapter.timing and complete != len(phrase)
                            and not beam_proof and not plan.get("contextual_owner")):
                        segmented_proof, segmented_expansion = _split_opening(obs, groups, word_ends, ayah, threshold)
                    noisy_proof, noisy_expansion = (None, False)
                    if (pause and adapter.metric == "ctc_greedy_probability"
                            and complete != len(phrase) and not beam_proof and not plan.get("contextual_owner")):
                        noisy_proof, noisy_expansion = _noisy_opening(obs, groups, word_ends, ayah, threshold)
                    # A globally distinct longer opening can still be incomplete
                    # in this crop. Try the shorter adjoining context after that
                    # concrete failure; pause centers do not require word timing.
                    opening_alternative = None
                    alternative_context_read_id = None
                    alternative_context_quality = None
                    if (pause and confirmed and confirmed[-1].ayah == ayah - 1
                            and complete != len(phrase) and not plan.get("contextual_owner") and not obs.truncated):
                        contextual = plan_seam(groups, ayah, word_ends)
                        if (contextual and _prefix(obs.units, contextual["units"]) == len(contextual["units"])
                                and min(obs.unit_quality[:len(contextual["units"])], default=0.) >= threshold
                                and not _strong_continuation_conflict(obs.units, obs.unit_quality,
                                    groups[ayah], len(contextual["units"]), threshold)):
                            opening_alternative = (obs, plan, phrase, complete, beam_proof,
                                                   noisy_proof, noisy_expansion)
                            plan = {**plan, **contextual, "contextual_owner": True}
                            phrase = plan["units"]
                            complete = len(phrase)
                            beam_proof = {}
                            noisy_proof, noisy_expansion = None, False
                            event("EXPAND_LOCAL", ayah, reason="incomplete_opening_contextual_pause",
                                  phrase=phrase, previous_tail=plan["tail"])
                            for context_ms in policy.speech_ms:
                                context_start = max(previous, pause.start_ms - context_ms)
                                context_obs = reads.read(context_start, pause.start_ms,
                                    reason=f"ayah={ayah}:repeated_owner_previous_tail")
                                tail = plan["tail"]
                                if (context_obs and not context_obs.truncated
                                        and tuple(context_obs.units[-len(tail):]) == tail
                                        and min(context_obs.unit_quality[-len(tail):], default=0.) >= threshold):
                                    alternative_context_read_id = reads.last_read_id
                                    alternative_context_quality = min(context_obs.unit_quality[-len(tail):])
                                    break
                            if alternative_context_read_id is None:
                                # Retain the original opening/provenance. Its
                                # expansion and future-owner evaluations below
                                # still run on the independent opening read.
                                (obs, plan, phrase, complete, beam_proof,
                                 noisy_proof, noisy_expansion) = opening_alternative
                                event("EVALUATE", ayah,
                                      reason="alternative_context_unproved_resume_opening",
                                      phrase=phrase, read_id=opening_id)
                    alias_conflict = any("spoken_alias" in item and tuple(item["source_units"]) != rendered[word_map[ayah][0]]
                                         for item in obs.attribution[:len(phrase)])
                    if plan.get("contextual_owner"):
                        alias_conflict |= _strong_continuation_conflict(
                            obs.units, obs.unit_quality, groups[ayah], len(phrase), threshold)
                    spoken_owner = (rendered[word_map[ayah][0]] if len(proof_ends[word_map[ayah][0]]) > 1 else ())
                    if spoken_owner:
                        alias_conflict |= any(expected != actual and obs.unit_quality[i] >= threshold
                                              for i, (expected, actual) in enumerate(zip(spoken_owner, obs.units)))
                    phrase_quality = min(obs.unit_quality[:len(phrase)], default=0.)
                    if segmented_proof:
                        phrase_quality = min(obs.unit_quality[:segmented_proof["segmented_opening"]["source_count"]])
                    if noisy_proof:
                        recovery = noisy_proof["opening_recovery"]
                        i = recovery["additional_word_offset"]
                        phrase_quality = min(obs.unit_quality[:i + len(recovery["additional_word"])])
                    event("EVALUATE", ayah, candidate=index, complete_units=complete,
                          needed_units=len(phrase), quality=obs.quality, observed=obs.units,
                          identity_recovery=noisy_proof,
                          onset_supported=False if noisy_proof else None)
                    if ((complete == len(phrase) or noisy_proof or segmented_proof) and not plan["ambiguous"] and not obs.truncated
                            and phrase_quality >= threshold and not alias_conflict):
                        context_read_id = alternative_context_read_id
                        if alternative_context_quality is not None:
                            phrase_quality = min(phrase_quality, alternative_context_quality)
                        if plan.get("contextual_owner") and pause:
                            tail = plan["tail"]
                            for context_ms in (() if context_read_id is not None else policy.speech_ms):
                                context_start = max(previous, pause.start_ms - context_ms)
                                context_obs = reads.read(context_start, pause.start_ms,
                                    reason=f"ayah={ayah}:repeated_owner_previous_tail")
                                if (context_obs and not context_obs.truncated
                                        and tuple(context_obs.units[-len(tail):]) == tail
                                        and min(context_obs.unit_quality[-len(tail):], default=0.) >= threshold):
                                    context_read_id = reads.last_read_id
                                    phrase_quality = min(phrase_quality, min(context_obs.unit_quality[-len(tail):]))
                                    break
                            if context_read_id is None:
                                reason = "unproved_repeated_owner_context"
                                blocked_prefix = True
                                break
                        elif plan.get("contextual_owner"):
                            reason = "unproved_repeated_owner_context"
                            blocked_prefix = True
                            break
                        if kind == "first_onset" and obs.positions_ms is None and not policy.independent_search:
                            reason = "unsupported_first_onset_timing"
                            break
                        proof = {"phrase": phrase, "complete": True, "ambiguous": False,
                                 "competitors_checked": True, "competitor_count": plan["competitor_count"],
                                 "offset": 0, "quality_metric": adapter.metric, **beam_proof}
                        if following_probe:
                            proof["bounded_following_origin_ayah"] = ayah
                        if noisy_proof:
                            proof.update(noisy_proof)
                        if segmented_proof:
                            proof.update(segmented_proof)
                        if spoken_owner:
                            proof["spoken_owner_units"] = spoken_owner
                        if context_read_id is not None:
                            proof.update(contextual_owner=True, previous_tail=plan["tail"],
                                         previous_context_read_id=context_read_id)
                        if pause:
                            ms = round(pause.center_ms)
                            proof.update(pause_ms=[pause.start_ms, pause.end_ms], pause_id=index)
                        else:
                            ms = round(reads.calls[opening_id]["window_ms"][0] +
                                       (obs.positions_ms[0] if obs.positions_ms else 0))
                            if obs.positions_ms is None:
                                proof["identity_only"] = True
                        if ms <= previous:
                            reason = "quantized_boundary_collision"
                            break
                        # Context reads must not replace the opening's provenance.
                        boundary = Boundary(ayah, ms, kind, phrase_quality, read_id=opening_id, proof=proof)
                        break
                    if (confirmed and confirmed[-1].ayah == ayah - 1 and not obs.truncated
                            and len(canonical_ends[ayah]) > 1
                            and len(phrase) > canonical_ends[ayah][0]):
                        first_end = canonical_ends[ayah][0]
                        first_word = groups[ayah][:first_end]
                        previous_ayah = confirmed[-1].ayah
                        previous_ends = canonical_ends[previous_ayah]
                        rival = None
                        if _prefix(obs.units, first_word) == len(first_word):
                            for word_index in range(1, len(previous_ends) - 1):
                                start_unit = previous_ends[word_index - 1]
                                end_first, end_second = previous_ends[word_index:word_index + 2]
                                previous_first = groups[previous_ayah][start_unit:end_first]
                                previous_second = groups[previous_ayah][end_first:end_second]
                                current_second = groups[ayah][first_end:canonical_ends[ayah][1]]
                                alternative = previous_first + previous_second
                                quality = obs.unit_quality[:len(alternative)]
                                if (previous_first == first_word and previous_second != current_second
                                        and words[word_map[previous_ayah][start_unit]].text == words[word_map[ayah][0]].text
                                        and words[word_map[previous_ayah][end_first]].text != words[word_map[ayah][first_end]].text
                                        and arabic_units(words[word_map[previous_ayah][end_first]].text)
                                        != arabic_units(words[word_map[ayah][first_end]].text)
                                        and _prefix(obs.units, alternative) == len(alternative)
                                        and len(quality) == len(alternative)
                                        and all(math.isfinite(q) for q in quality)
                                        and min(quality, default=0.) >= threshold
                                        and _strong_continuation_conflict(obs.units, obs.unit_quality,
                                                                         groups[ayah], first_end, threshold)):
                                    rival = {"ayah": previous_ayah, "start_unit": start_unit,
                                             "units": alternative, "min_quality": min(quality)}
                                    break
                        if rival:
                            reason = "decoded_previous_ayah_body"
                            event("DECLINE_CANDIDATE", ayah, candidate=index, reason=reason, rival=rival,
                                  read_id=opening_id)
                            had_prefix = False
                            blocked_prefix = False
                            break
                    future_expansion = None
                    if pause is not None and obs.units:
                        for future_ayah in ayahs:
                            if future_ayah <= ayah:
                                continue
                            next_plan = planned_opening(future_ayah)
                            next_phrase = next_plan["units"]
                            next_obs, future_proof = _beam_view(obs, next_phrase, threshold, policy.beam_log_margin)
                            next_owner = rendered[word_map[future_ayah][0]]
                            next_spoken = len(proof_ends[word_map[future_ayah][0]]) > 1
                            conflict = any("spoken_alias" in item and tuple(item["source_units"]) != next_owner
                                           for item in next_obs.attribution[:len(next_phrase)])
                            if next_spoken:
                                conflict |= any(expected != actual and next_obs.unit_quality[i] >= threshold
                                                for i, (expected, actual) in enumerate(zip(next_owner, next_obs.units)))
                                future_proof["spoken_owner_units"] = next_owner
                            quality = min(next_obs.unit_quality[:len(next_phrase)], default=0.)
                            if (not next_plan["ambiguous"] and _prefix(next_obs.units, next_phrase) == len(next_phrase)
                                    and not next_obs.truncated and quality >= threshold and not conflict
                                    and (not following_probe or future_ayah - ayah <= policy.expansion_lookahead)):
                                if following_probe:
                                    future_proof["bounded_following_origin_ayah"] = ayah
                                future = (future_ayah, next_phrase, pause, index, opening_id, quality, future_proof)
                                reason = ("future_ayah_without_current" if future_ayah == ayah + 1
                                          else "future_outside_bounded_block")
                                break
                            if (policy.sparse_anchors and future_ayah - ayah <= policy.expansion_lookahead
                                    and not next_plan["ambiguous"] and not conflict):
                                anchor = _expansion_evidence(next_obs, next_phrase, word_ends[future_ayah], threshold,
                                                             groups, future_ayah, word_ends)
                                if (anchor and (anchor != "whole_spoken_prefix_incomplete_canonical_word"
                                        or (pause.end_ms - pause.start_ms) / 2 <= policy.short_word_pause_radius_ms)):
                                    future_expansion = (future_ayah, anchor)
                        if future:
                            break
                    expansion = _expansion_evidence(obs, phrase, word_ends[ayah], threshold, groups, ayah, word_ends)
                    if (expansion == "whole_spoken_prefix_incomplete_canonical_word" and pause
                            and (pause.end_ms - pause.start_ms) / 2 > policy.short_word_pause_radius_ms):
                        event("SEARCH_CUE_ONLY", ayah, read_id=opening_id,
                              reason="short_word_pause_time_uncertain", observed=obs.units,
                              pause_ms=[pause.start_ms, pause.end_ms])
                        expansion = None
                    if not expansion and segmented_expansion:
                        expansion = "split_head_requires_whole_continuation"
                    if not expansion and noisy_expansion:
                        expansion = "noisy_opening_additional_whole_word"
                    if noisy_expansion and step == 0:
                        # A damaged minimal opening already contains the next
                        # word's beginning. Small finite extensions preserve
                        # that independent observation; a jump to 2.5/4s can
                        # change the decoder output after adding unrelated text.
                        speech_steps[step + 1:] = [ms for ms in policy.noisy_speech_ms if ms > speech]
                    if not expansion and posterior_expansion:
                        expansion = "canonical_prefix_vs_unresolved_residual"
                    # Only the new permission is provisional. An earlier
                    # supported opening keeps its existing repetition block.
                    soft_short_word_route |= (expansion == "whole_spoken_prefix_incomplete_canonical_word"
                                              and not had_prefix)
                    current_expansion = bool(expansion) and not soft_short_word_route
                    if not expansion and future_expansion:
                        expansion = "provisional_unique_anchor_partial:" + future_expansion[1]
                    speech_edge = None
                    if not expansion and step + 1 < len(speech_steps):
                        speech_edge = _incomplete_speech_edge(pcm, sr, end, obs, phrase, groups,
                                                             word_ends, ayah, threshold)
                        if speech_edge:
                            expansion = "measured_speech_edge_incomplete_phrase"
                    if expansion:
                        had_prefix |= current_expansion
                        reason = "ambiguous_phrase" if plan["ambiguous"] else "incomplete_distinguishing_phrase"
                        if step + 1 < len(speech_steps):
                            event("EXPAND_LOCAL", ayah, reason=reason, acoustic_anchor=expansion,
                                  speech_edge=speech_edge)
                            continue
                        blocked_prefix |= current_expansion
                    # A previously plausible opening that became unclear cannot
                    # be silently replaced by a later repetition.
                    blocked_prefix = blocked_prefix or had_prefix
                    break
                if soft_short_word_route and boundary is None and future is None:
                    # This additional permission is provisional. Its failed
                    # reads cannot turn the original incomplete opening into
                    # a sticky block that hides a later independent anchor.
                    blocked_prefix = False
                    plan, phrase = short_word_base_plan, short_word_base_phrase
                    event("EVALUATE", ayah, candidate=index,
                          reason="short_word_extension_unresolved_resume_candidates")
                if boundary or blocked_prefix or future:
                    break
            if boundary is None and future:
                event("RESOLVE_BLOCK", ayah, future_ayah=future[0], reason=reason)
                # A future pause stays provisional. Publish a two-ayah block only
                # when one bounded independent reading proves ALL current text
                # and the next prefix, with a real current onset.
                if adapter.timing and not confirmed and future[0] == ayah + 1:
                    next_ayah, next_phrase, pause, index, future_id, next_quality, future_proof = future
                    end = min(duration, pause.end_ms + policy.speech_ms[0])
                    start = max(0., previous, end - 6000)
                    combined = groups[ayah] + next_phrase
                    obs = reads.read(start, end, reason=f"ayah={ayah}:resolve_adjacent_block")
                    if obs and obs.positions_ms is not None and not obs.truncated:
                        offsets = [i for i in range(len(obs.units)) if obs.units[i:i + len(combined)] == combined]
                        if len(offsets) == 1 and not plan["ambiguous"]:
                            offset = offsets[0]
                            quality = min(obs.unit_quality[offset:offset + len(combined)], default=0.)
                            ms = round(reads.calls[reads.last_read_id]["window_ms"][0] + obs.positions_ms[offset])
                            if quality >= threshold and previous < ms < round(pause.center_ms):
                                proof = {"phrase": phrase, "complete": True, "ambiguous": False,
                                         "competitors_checked": True, "offset": offset,
                                         "quality_metric": adapter.metric, "block_ayahs": [ayah, next_ayah],
                                         "following_phrase": next_phrase}
                                kind = "first_onset" if not confirmed else "wasl"
                                if confirmed:
                                    tail_plan = plan_tail(groups, confirmed[-1].ayah, word_ends)
                                    tail = tail_plan["units"]
                                    if tail_plan["ambiguous"] or offset < len(tail) or obs.units[offset-len(tail):offset] != tail:
                                        quality = 0.
                                    proof["previous_tail"] = tail
                                    quality = min(quality, min(obs.unit_quality[offset-len(tail):offset], default=0.))
                                if quality >= threshold:
                                    current = Boundary(ayah, ms, kind, quality, read_id=reads.last_read_id, proof=proof)
                                    next_proof = {"phrase": next_phrase, "complete": True, "ambiguous": False,
                                                  "competitors_checked": True, "offset": 0,
                                                  "pause_ms": [pause.start_ms, pause.end_ms], "pause_id": index,
                                                  "quality_metric": adapter.metric, **future_proof}
                                    following = Boundary(next_ayah, round(pause.center_ms), "pause", next_quality,
                                                         read_id=future_id, proof=next_proof)
                                    pair = [current, following]
                                    validate_boundaries(pair, [ayah, next_ayah], duration, reads.calls,
                                                        groups=groups, word_ends=word_ends, canonical_ends=canonical_ends, threshold=threshold, pauses=pauses, policy=policy, pause_geometry_data=geometry)
                                    confirmed.extend(pair)
                                    event("CONFIRMED", ayah, block_ayahs=[ayah, next_ayah])
                                    event("ADVANCE", next_ayah)
                                    if ayah_progress:
                                        ayah_progress(next_ayah)
                                    continue
                reason = "unresolved_adjacent_block" if future[0] == ayah + 1 else "future_outside_bounded_block"
            if boundary is None and confirmed:
                event("RESOLVE_BLOCK", ayah, reason=reason)
                if adapter.timing:
                    prior_ayah = confirmed[-1].ayah
                    consumed = sum(len(u) for a in ayahs if confirmed[0].ayah <= a <= prior_ayah for u in groups[a])
                    remaining_weight = sum(len(u) for a in ayahs if a >= ayah for u in groups[a])
                    tempo = (duration - confirmed[-1].start_ms) / max(1, remaining_weight + sum(map(len, groups[prior_ayah])))
                    if len(confirmed) > 1:
                        tempo = (confirmed[-1].start_ms - confirmed[0].start_ms) / max(1, consumed - sum(map(len, groups[prior_ayah])))
                    forecast = confirmed[-1].start_ms + tempo * sum(map(len, groups[prior_ayah]))
                    if future and future[0] == ayah + 1:
                        # The provisional future DSP location bounds a search;
                        # only the heard tail+opening can prove the current owner.
                        prior_weight = sum(map(len, groups[prior_ayah]))
                        current_weight = sum(map(len, groups[ayah]))
                        forecast = previous + (future[2].center_ms - previous) * prior_weight / max(1, prior_weight + current_weight)
                    start, end = max(previous, forecast - 2500), min(duration, forecast + 3000)
                    seam_plan = plan_seam(groups, ayah, word_ends)
                    tail_plan = seam_plan or plan_tail(groups, prior_ayah, word_ends)
                    tail = tail_plan["tail"] if seam_plan else tail_plan["units"]
                    wasl_phrase = seam_plan["units"] if seam_plan else phrase
                    combined = tail + wasl_phrase
                    if future:
                        end = min(end, future[2].start_ms)
                    event("READ_LOCAL", ayah, kind="wasl", window_ms=[start, end], reason="anchored_wasl_forecast")
                    obs = reads.read(start, end, reason=f"ayah={ayah}:anchored_wasl_forecast")
                    if obs and obs.positions_ms is not None and not obs.truncated and (seam_plan or not plan["ambiguous"]) and not tail_plan["ambiguous"]:
                        occurrences = [i for i in range(len(obs.units)) if obs.units[i:i + len(combined)] == combined]
                        if len(occurrences) == 1:
                            offset = occurrences[0] + len(tail)
                            ms = round(reads.calls[reads.last_read_id]["window_ms"][0] + obs.positions_ms[offset])
                            phrase_quality = min(obs.unit_quality[occurrences[0]:offset + len(wasl_phrase)], default=0.)
                            if (previous < ms < duration and phrase_quality >= threshold
                                    and (not seam_plan or not _strong_continuation_conflict(
                                        obs.units[offset:], obs.unit_quality[offset:], groups[ayah],
                                        len(wasl_phrase), threshold))):
                                proof = {"phrase": wasl_phrase, "complete": True, "ambiguous": False,
                                         "competitors_checked": True, "offset": offset,
                                         "previous_tail": tail, "quality_metric": adapter.metric,
                                         "contextual_owner": bool(seam_plan)}
                                boundary = Boundary(ayah, ms, "wasl", phrase_quality, read_id=reads.last_read_id, proof=proof)
                    if not boundary:
                        reason = "unresolved_local_block"
                else:
                    reason = "unsupported_wasl_timing"
            if (boundary is None and future is None and policy.sparse_anchors
                    and adapter.model_id in {"whisper-tarteel-tiny", "whisper-tarteel-base"}
                    and not adapter.timing and len(confirmed) >= 2 and not plan["ambiguous"]
                    and not plan.get("contextual_owner") and canonical_ends[ayah][0] == 1
                    and len(phrase) > 1 and len(phrase[0]) >= 3):
                current_words = [i for i, w in enumerate(words) if w.ayah == ayah]
                owners = {}
                for wi in range(len(rendered) - 1):
                    owners.setdefault(rendered[wi] + rendered[wi + 1], []).append(wi)
                current_reads = [r for r in reads.calls if r["reason"].startswith(f"ayah={ayah}:")
                                 and "previous_tail" not in r["reason"]]
                body_cue = None
                for r in current_reads:
                    observed = r.get("observed", {})
                    if observed.get("truncated"):
                        continue
                    units, qualities = observed.get("units", ()), observed.get("unit_quality", ())
                    for wi in current_words[1:-1]:
                        pair = rendered[wi] + rendered[wi + 1]
                        if owners[pair] != [wi]:
                            continue
                        for offset in range(len(units) - len(pair) + 1):
                            q = qualities[offset:offset + len(pair)]
                            if (tuple(units[offset:offset + len(pair)]) == pair
                                    and len(q) == len(pair) and all(math.isfinite(x) for x in q)
                                    and min(q) >= threshold and not any("spoken_alias" in a for a in
                                        observed.get("attribution", ())[offset:offset + len(pair)])):
                                body_cue = (r["id"], wi, pair)
                                break
                        if body_cue:
                            break
                    if body_cue:
                        break
                if body_cue:
                    tried_head_candidates = set()
                    for initial in current_reads:
                        observed = initial.get("observed", {})
                        units, qualities = observed.get("units", ()), observed.get("unit_quality", ())
                        if (observed.get("truncated") or not units or not qualities or qualities[0] <= 0
                                or not units[0].startswith(phrase[0]) or len(units[0]) <= len(phrase[0])
                                or any("unsupported" in a for a in observed.get("attribution", ()))):
                            continue
                        for kind, index, _, _, pause in candidates_here[:policy.max_candidates]:
                            if pause is None or index in tried_head_candidates or not geometry[index]["accepted"]:
                                continue
                            start = max(pause.start_ms, pause.end_ms - 200.)
                            if round(initial["window_ms"][0] * sr / 1000) != round(start * sr / 1000):
                                continue
                            tried_head_candidates.add(index)
                            for speech in policy.speech_ms[1:]:
                                event("HEAD_BOUNDARY_RECHECK", ayah, candidate=index,
                                      initial_read_id=initial["id"], body_read_id=body_cue[0],
                                      observed_word=units[0], expected_prefix=phrase[0],
                                      window_ms=[start, min(duration, pause.end_ms + speech)])
                                recheck = reads.read(start, min(duration, pause.end_ms + speech),
                                                     reason=f"ayah={ayah}:head_word_boundary_recheck")
                                if recheck is None:
                                    break
                                q = recheck.unit_quality[:len(phrase)]
                                conflict = any("spoken_alias" in a and tuple(a["source_units"]) != rendered[word_map[ayah][0]]
                                               for a in recheck.attribution[:len(phrase)])
                                if (_prefix(recheck.units, phrase) == len(phrase) and not recheck.truncated
                                        and len(q) == len(phrase) and min(q) >= threshold and not conflict):
                                    ms = round(pause.center_ms)
                                    if ms <= previous:
                                        break
                                    proof = {"phrase": phrase, "complete": True, "ambiguous": False,
                                             "competitors_checked": True, "competitor_count": plan["competitor_count"],
                                             "offset": 0, "quality_metric": adapter.metric,
                                             "pause_ms": [pause.start_ms, pause.end_ms], "pause_id": index,
                                             "head_boundary_recheck": True, "body_cue_read_id": body_cue[0],
                                             "initial_head_read_id": initial["id"]}
                                    boundary = Boundary(ayah, ms, "pause", min(q), read_id=reads.last_read_id, proof=proof)
                                    validate_boundaries([boundary], [ayah], duration, reads.calls,
                                                        groups=groups, word_ends=word_ends, canonical_ends=canonical_ends,
                                                        threshold=threshold, pauses=pauses, policy=policy, pause_geometry_data=geometry)
                                    break
                                if not _expansion_evidence(recheck, phrase, word_ends[ayah], threshold, groups, ayah, word_ends):
                                    break
                            if boundary:
                                break
                        if boundary:
                            break
            if boundary is None:
                if (future and policy.sparse_anchors and future[0] > ayah
                        and (policy.max_unresolved_ayahs is None or future[0] - ayah <= policy.max_unresolved_ayahs)):
                    future_ayah, future_phrase, pause, index, read_id, quality, future_proof = future
                    proof = {"phrase": future_phrase, "complete": True, "ambiguous": False,
                             "competitors_checked": True, "offset": 0, "quality_metric": adapter.metric,
                             "pause_ms": [pause.start_ms, pause.end_ms], "pause_id": index,
                             "sparse_anchor": True, "unresolved_ayahs": list(range(ayah, future_ayah)), **future_proof}
                    if reentry_failures:
                        proof.update(independent_target_origin_ayah=reentry_origin,
                                     independent_target_attempt=reentry_failures, search_target_ayah=ayah)
                    anchor = Boundary(future_ayah, round(pause.center_ms), "pause", quality,
                                      read_id=read_id, proof=proof)
                    validate_boundaries([anchor], [future_ayah], duration, reads.calls,
                                        groups=groups, word_ends=word_ends, canonical_ends=canonical_ends, threshold=threshold, pauses=pauses, policy=policy, pause_geometry_data=geometry,
                                        search_only=policy.independent_search)
                    confirmed.append(anchor)
                    reentry_failures = 0
                    reentry_origin = None
                    unresolved[ayah] = "unresolved_before_unique_anchor"
                    event("REVIEW_REQUIRED", ayah, reason=unresolved[ayah], block=proof["unresolved_ayahs"])
                    event("CONFIRMED", future_ayah, evidence="pause", sparse_anchor=True)
                    event("ADVANCE", future_ayah)
                    if ayah_progress:
                        ayah_progress(future_ayah)
                    continue
                unresolved[ayah] = reason
                event("REVIEW_REQUIRED", ayah, reason=reason)
                if (policy.independent_search and confirmed and ayah < ayahs[-1]
                        and reentry_failures < policy.max_block_ayahs):
                    if reentry_origin is None:
                        reentry_origin = ayah
                    reentry_failures += 1
                    event("REENTER_SEARCH", ayah + 1, unresolved_ayah=ayah,
                          search_anchor_ayah=confirmed[-1].ayah,
                          reason="independent_next_canonical_target", attempt=reentry_failures)
                    continue
                if (policy.independent_search and policy.sparse_anchors
                        and landmark_attempts < policy.landmark_probes):
                    # One deliberately separate unresolved region, not a scan
                    # of all pauses. Duration/text nominate a hypothesis only;
                    # the independently decoded complete phrase owns the anchor.
                    landmark_attempts += 1
                    remaining = [a for a in ayahs if a >= ayah]
                    weights = [sum(map(len, groups[a])) for a in remaining]
                    total_weight, cumulative = sum(weights), 0
                    target = remaining[-1]
                    for candidate_ayah, weight in zip(remaining, weights):
                        cumulative += weight
                        if cumulative >= total_weight / 2:
                            target = candidate_ayah
                            break
                    lower = confirmed[-1].start_ms if confirmed else first
                    ratio = sum(weights[:remaining.index(target)]) / max(1, total_weight)
                    nomination = lower + (duration - lower) * ratio
                    pool = [(i, p) for i, p in enumerate(pauses[:initial_pause_count])
                            if p.start_ms > lower and p.end_ms + policy.speech_ms[0] < duration]
                    selected = min(pool, key=lambda item: abs(item[1].center_ms - nomination), default=None)
                    if selected is not None:
                        index, pause = selected
                        start = max(pause.start_ms, pause.end_ms - 200.)
                        event("REACQUIRE_REGION", ayah, candidate=index, target_ayah=target,
                              nominated_ms=nomination, pause_ms=[pause.start_ms, pause.end_ms],
                              reason="unresolved_region_text_duration_hypothesis")
                        found = None
                        for speech in policy.speech_ms:
                            obs = reads.read(start, min(duration, pause.end_ms + speech),
                                             reason=f"ayah={ayah}:independent_region_identity")
                            if obs is None:
                                break
                            read_id = reads.last_read_id
                            eligible, found = False, None
                            for owner in remaining:
                                # A later repetition cannot replace the unresolved
                                # current opening; locate an independent later owner.
                                if owner <= ayah:
                                    continue
                                owner_plan = planned_opening(owner)
                                if owner_plan["ambiguous"]:
                                    continue
                                owner_phrase = owner_plan["units"]
                                view, beam = _beam_view(obs, owner_phrase, threshold, policy.beam_log_margin)
                                qualities = view.unit_quality[:len(owner_phrase)]
                                owner_word = rendered[word_map[owner][0]]
                                spoken = len(proof_ends[word_map[owner][0]]) > 1
                                conflict = any("spoken_alias" in a and tuple(a["source_units"]) != owner_word
                                               for a in view.attribution[:len(owner_phrase)])
                                if spoken:
                                    conflict |= any(wanted != actual and q >= threshold
                                                    for wanted, actual, q in zip(owner_word, view.units, view.unit_quality))
                                if (_prefix(view.units, owner_phrase) == len(owner_phrase) and not view.truncated
                                        and len(qualities) == len(owner_phrase) and all(math.isfinite(q) for q in qualities)
                                        and min(qualities) >= threshold and not conflict):
                                    proof = {"phrase": owner_phrase, "complete": True, "ambiguous": False,
                                             "competitors_checked": True, "offset": 0, "quality_metric": adapter.metric,
                                             "pause_ms": [pause.start_ms, pause.end_ms], "pause_id": index,
                                             "independent_region_origin_ayah": ayah,
                                             "independent_region_attempt": landmark_attempts}
                                    proof.update(beam)
                                    if spoken:
                                        proof["spoken_owner_units"] = owner_word
                                    found = Boundary(owner, round(pause.center_ms), "pause", min(qualities),
                                                     read_id=read_id, proof=proof)
                                    break
                                eligible |= bool(_expansion_evidence(view, owner_phrase, word_ends[owner],
                                                                   threshold, groups, owner, word_ends))
                            if found:
                                validate_boundaries([found], [found.ayah], duration, reads.calls,
                                    groups=groups, word_ends=word_ends, canonical_ends=canonical_ends,
                                    threshold=threshold, pauses=pauses, policy=policy,
                                    pause_geometry_data=geometry, search_only=True)
                                confirmed.append(found)
                                reentry_failures = 0
                                event("SEARCH_ANCHOR", found.ayah, candidate=index,
                                      origin_ayah=ayah, read_id=read_id, evidence="independent_region")
                                break
                            if not eligible:
                                break
                        if found:
                            continue
                break
            if reentry_failures:
                boundary = replace(boundary, proof={**boundary.proof,
                    "independent_target_origin_ayah": reentry_origin,
                    "independent_target_attempt": reentry_failures, "search_target_ayah": ayah})
            reentry_failures = 0
            reentry_origin = None
            if (following_probe_id is not None and boundary.evidence_kind == "pause"
                    and boundary.proof.get("pause_id") == following_probe_id):
                boundary = replace(boundary, proof={**boundary.proof, "bounded_following_origin_ayah": ayah})
            event("CONFIRMED", ayah, evidence=boundary.evidence_kind, start_ms=boundary.start_ms)
            validate_boundaries([boundary], [ayah], duration, reads.calls,
                                groups=groups, word_ends=word_ends, canonical_ends=canonical_ends,
                                threshold=threshold, pauses=pauses, policy=policy,
                                pause_geometry_data=geometry, search_only=policy.independent_search)
            confirmed.append(boundary)
            event("ADVANCE", ayah)
            if ayah_progress:
                ayah_progress(ayah)
        region_root = None
        for i, anchor in enumerate(confirmed):
            if ({"independent_region_origin_ayah", "independent_target_origin_ayah"} & (anchor.proof or {}).keys()):
                region_root = asdict(anchor)
            if region_root is not None:
                confirmed[i] = replace(anchor, proof={**anchor.proof, "search_region_root": region_root})
        trace["search_anchors"] = [asdict(b) for b in confirmed]
        reviewed, recovery_reviews = _review_addressed_recovery_onsets(confirmed, geometry, policy)
        reviewed, temporal_reviews = _review_weak_sparse_onsets(reviewed, reads.calls, geometry, threshold)
        temporal_reviews.extend(recovery_reviews)
        publishable = []
        for anchor in reviewed:
            index = (anchor.proof or {}).get("pause_id")
            missing_timing = ((anchor.proof or {}).get("identity_only") or
                              (index is not None and not geometry[index]["accepted"]))
            if missing_timing:
                unresolved[anchor.ayah] = (geometry[index]["reason"] if index is not None
                                           else "unsupported_first_onset_timing")
                event("IDENTITY_ONLY", anchor.ayah, reason=unresolved[anchor.ayah],
                      read_id=anchor.read_id, search_location_ms=anchor.start_ms)
            else:
                publishable.append(anchor)
        result = _complete_result(publishable, ayahs, reason, unresolved)
        report("verify")
        for review in temporal_reviews:
            event("IDENTITY_ONLY", **review)
            event("REVIEW_REQUIRED", review["ayah"], reason=review["reason"])
        event("VALIDATE")
        validate_boundaries(result, ayahs, duration, reads.calls,
                            groups=groups, word_ends=word_ends, canonical_ends=canonical_ends, threshold=threshold, pauses=pauses, policy=policy, pause_geometry_data=geometry)
        trace["boundaries"] = [asdict(b) for b in result]
        trace["confidence_semantics"] = {
            "recognition": "minimum_independently_supported_essential_unit_score",
            "calibrated_probability": False,
            "temporal_probability": None,
            "pause_timing": "midpoint_of_measured_DSP_interval_not_calibrated_probability",
            "noisy_opening": "identity_only_missing_onset_not_reconstructed",
        }
        trace["ayahs"] = [{"ayah": b.ayah, "onset_ms": b.start_ms, "conf": b.confidence,
                            "reliable": b.evidence_kind != "unresolved"} for b in result]
        trace["result_complete"] = all(b.start_ms is not None for b in result)
        trace["unresolved_ayahs"] = [b.ayah for b in result if b.start_ms is None]
        trace["review_required_ayahs"] = [b.ayah for b in result if b.evidence_kind == "unresolved"]
        trace["pause_matches"] = [{"ayah": b.ayah, "from_pause": b.evidence_kind == "pause",
                                   "onset_ms": b.start_ms, "conf": b.confidence,
                                   "boundary_evidence": b.evidence_kind,
                                   **({"pause_start_ms": b.proof["pause_ms"][0],
                                       "pause_end_ms": b.proof["pause_ms"][1]} if b.evidence_kind == "pause" else {})}
                                  for b in result]
        trace.update(n_pause=sum(b.evidence_kind == "pause" for b in result),
                     n_wasl=sum(b.evidence_kind == "wasl" for b in result), bounded_ctc_fallback=None)
        event("RESULT_READY")
        return [(b.ayah, b.start_ms, b.confidence) for b in result if b.start_ms is not None]
    except GenerationCancelled:
        events.append({"state": "CANCELLED"})
        raise
    except BaseException as error:
        events.append({"state": "FAILED", "error": str(error)})
        raise
    finally:
        trace.update(reads.diagnostics())
        trace["inference_seconds"] = sum(r.get("seconds", 0.) for r in reads.calls)
