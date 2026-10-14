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
from surah_cutter.core.fast_adapters import make_adapter, representation_sha, loaded_implementation_sha
from surah_cutter.core.fast_reads import LocalEvidenceReads

VERSION = "sequential-fast-v3-sparse-5"


@dataclass(frozen=True)
class Policy:
    speech_ms: tuple[int, ...] = (1300, 2500, 4000)
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
    start_ms: int
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
    if pause.end_ms - pause.start_ms <= 1000:
        return {"accepted": True, "reason": "short_dsp_interval"}
    left, right = int(pause.start_ms * sr / 1000), int(pause.end_ms * sr / 1000)
    local = pcm[left:right]
    frame = max(1, sr // 50)
    count = len(local) // frame
    if count < 8:
        return {"accepted": False, "reason": "unmeasurable_long_interval"}
    levels = np.sqrt(np.mean(local[:count * frame].reshape(count, frame) ** 2, axis=1))
    quarter = max(1, count // 4)
    beginning, ending = float(np.median(levels[:quarter])), float(np.median(levels[-quarter:]))
    decay = beginning > max(.003, ending * 4)
    return {"accepted": not decay, "reason": "decaying_tail_midpoint_unproved" if decay else "stable_long_interval",
            "first_quarter_rms": beginning, "last_quarter_rms": ending}


def validate_boundaries(boundaries, ayahs, duration_ms, reads, *, groups=None, word_ends=None,
                        canonical_ends=None, threshold=0., pauses=None, policy=Policy(), pause_geometry_data=None):
    if [b.ayah for b in boundaries] != list(ayahs):
        raise ValueError("canonical_ayah_set_mismatch")
    last = -1
    for boundary in boundaries:
        ms = boundary.start_ms
        if type(ms) is not int or not last < ms < duration_ms or ms < 0:
            raise ValueError("invalid_boundary_timeline")
        last = ms
        if not math.isfinite(boundary.confidence) or not 0 <= boundary.confidence <= 1:
            raise ValueError("invalid_boundary_quality")
        if boundary.evidence_kind == "estimate":
            if boundary.confidence != 0 or not boundary.review_reason or boundary.read_id is not None:
                raise ValueError("invalid_estimate")
            continue
        proof = boundary.proof or {}
        if boundary.review_reason or boundary.read_id is None or not 0 <= boundary.read_id < len(reads):
            raise ValueError("missing_acoustic_proof")
        read = reads[boundary.read_id]
        if read["outcome"] != "completed" or not proof.get("complete") or proof.get("ambiguous"):
            raise ValueError("incomplete_acoustic_proof")
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
        if not phrase or tuple(observed["units"][offset:offset + len(phrase)]) != phrase:
            raise ValueError("unheard_discriminating_phrase")
        quality = observed.get("unit_quality", ())[offset:offset + len(phrase)]
        if (len(quality) != len(phrase) or any(not math.isfinite(q) for q in quality)
                or min(quality) < threshold or boundary.confidence > min(quality)):
            raise ValueError("weak_discriminating_unit")
        if groups is not None:
            planned = plan_phrase(groups, boundary.ayah, word_ends)
            if proof.get("contextual_owner"):
                seam = plan_seam(groups, boundary.ayah, word_ends)
                if not seam or phrase != seam["units"] or tuple(proof.get("previous_tail", ())) != seam["tail"]:
                    raise ValueError("ambiguous_contextual_owner")
            elif phrase != planned["units"] or planned["ambiguous"]:
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
                if right - left > 1000 and (pause_geometry_data is None or not pause_geometry_data[index]["accepted"]):
                    raise ValueError("unproved_long_pause_geometry")
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


def _finish_estimates(confirmed, ayahs, groups, duration_ms, reason, unresolved=None):
    """Keep proven anchors exact; fill each unresolved block with honest conf0."""
    anchors = {b.ayah: b for b in confirmed}
    result, previous, cursor = [], None, 0
    while cursor < len(ayahs):
        ayah = ayahs[cursor]
        if ayah in anchors:
            previous = anchors[ayah]
            result.append(previous)
            cursor += 1
            continue
        end = cursor
        while end < len(ayahs) and ayahs[end] not in anchors:
            end += 1
        remaining = ayahs[cursor:end]
        following = anchors.get(ayahs[end]) if end < len(ayahs) else None
        lower = previous.start_ms + 1 if previous else 0
        upper = following.start_ms - 1 if following else math.ceil(duration_ms) - 1
        if len(remaining) > upper - lower + 1:
            raise ValueError("unrepresentable_timeline")
        weights = [max(1, sum(map(len, groups[a]))) for a in remaining]
        free = upper - lower + 1 - len(remaining)
        previous_weight = max(1, sum(map(len, groups[previous.ayah]))) if previous else 0
        total, acc = sum(weights) + previous_weight, previous_weight
        for i, (current, weight) in enumerate(zip(remaining, weights)):
            ms = lower + i + (free * acc // total if total else 0)
            result.append(Boundary(current, ms, "estimate", 0.,
                                   (unresolved or {}).get(current, reason) if i == 0 else "blocked_by_unresolved_ayah"))
            acc += weight
        cursor = end
    return result


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
    if policy.max_candidates < 1 or policy.max_block_ayahs != 2 or not policy.speech_ms:
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
    trace.update(align_lag_ms=adapter.lag_ms, backend=adapter.metric, fast_adapter=adapter.version,
                 fast_version=VERSION, fast_run_key=key, run_snapshot=snapshot,
                 model_source=f"{adapter.model_id}:revision={adapter.revision}:adapter={adapter.version}",
                 total_ms=duration, duration_ms=duration, transitions=events)
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
    try:
        event("LOCATE_FIRST", ayahs[0])
        for ayah in ayahs:
            if confirmed and ayah <= confirmed[-1].ayah:
                continue  # already validated and atomically published by RESOLVE_BLOCK
            event("PLAN_PHRASE", ayah)
            plan = plan_phrase(groups, ayah, word_ends)
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
            if len(confirmed) >= 2:
                prev = confirmed[-1].ayah
                elapsed_weight = sum(len(u) for a in ayahs
                                     if confirmed[0].ayah <= a < prev for u in groups[a])
                tempo = (confirmed[-1].start_ms - confirmed[0].start_ms) / max(1, elapsed_weight)
                predicted_span = tempo * sum(map(len, groups[prev]))
                forecast = previous + predicted_span
                left = previous + predicted_span * policy.tempo_lower_ratio
                right = previous + predicted_span * policy.tempo_upper_ratio
                candidates_here = [c for c in candidates_here if left <= c[2] <= right]
                event("SELECT_CANDIDATE", ayah, forecast_ms=forecast, region_ms=[left, right])
            boundary, blocked_prefix, future = None, False, None
            for kind, index, start, speech_start, pause in candidates_here[:policy.max_candidates]:
                # Geometry belongs to DSP. Acoustic identity needs speech at
                # the right edge, not the entire already measured silence.
                if pause:
                    start = max(pause.start_ms, pause.end_ms - 200.)
                    if not geometry[index]["accepted"]:
                        reason = geometry[index]["reason"]
                        event("SELECT_CANDIDATE", ayah, candidate=index, declined_geometry=reason)
                        continue
                event("SELECT_CANDIDATE", ayah, candidate=index, kind=kind)
                had_prefix = False
                for step, speech in enumerate(policy.speech_ms):
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
                    alias_conflict = any("spoken_alias" in item and tuple(item["source_units"]) != rendered[word_map[ayah][0]]
                                         for item in obs.attribution[:len(phrase)])
                    spoken_owner = (rendered[word_map[ayah][0]] if len(proof_ends[word_map[ayah][0]]) > 1 else ())
                    if spoken_owner:
                        alias_conflict |= any(expected != actual and obs.unit_quality[i] >= threshold
                                              for i, (expected, actual) in enumerate(zip(spoken_owner, obs.units)))
                    phrase_quality = min(obs.unit_quality[:len(phrase)], default=0.)
                    event("EVALUATE", ayah, candidate=index, complete_units=complete,
                          needed_units=len(phrase), quality=obs.quality, observed=obs.units)
                    if (complete == len(phrase) and not plan["ambiguous"] and not obs.truncated
                            and phrase_quality >= threshold and not alias_conflict):
                        context_read_id = None
                        if plan.get("contextual_owner") and pause:
                            tail = plan["tail"]
                            for context_ms in policy.speech_ms:
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
                        if kind == "first_onset" and obs.positions_ms is None:
                            reason = "unsupported_first_onset_timing"
                            break
                        proof = {"phrase": phrase, "complete": True, "ambiguous": False,
                                 "competitors_checked": True, "competitor_count": len(plan["competitors"]),
                                 "offset": 0, "quality_metric": adapter.metric, **beam_proof}
                        if spoken_owner:
                            proof["spoken_owner_units"] = spoken_owner
                        if context_read_id is not None:
                            proof.update(contextual_owner=True, previous_tail=plan["tail"],
                                         previous_context_read_id=context_read_id)
                        if pause:
                            ms = round(pause.center_ms)
                            proof.update(pause_ms=[pause.start_ms, pause.end_ms], pause_id=index)
                        else:
                            ms = round(start + obs.positions_ms[0])
                        if ms <= previous:
                            reason = "quantized_boundary_collision"
                            break
                        # Context reads must not replace the opening's provenance.
                        boundary = Boundary(ayah, ms, kind, phrase_quality, read_id=opening_id, proof=proof)
                        break
                    future_expansion = None
                    if pause is not None and obs.units:
                        for future_ayah in ayahs:
                            if future_ayah <= ayah:
                                continue
                            next_plan = plan_phrase(groups, future_ayah, word_ends)
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
                                    and not next_obs.truncated and quality >= threshold and not conflict):
                                future = (future_ayah, next_phrase, pause, index, reads.last_read_id, quality, future_proof)
                                reason = ("future_ayah_without_current" if future_ayah == ayah + 1
                                          else "future_outside_bounded_block")
                                break
                            if (policy.sparse_anchors and future_ayah - ayah <= policy.expansion_lookahead
                                    and not next_plan["ambiguous"] and not conflict):
                                anchor = _expansion_evidence(next_obs, next_phrase, word_ends[future_ayah], threshold,
                                                             groups, future_ayah, word_ends)
                                if anchor:
                                    future_expansion = (future_ayah, anchor)
                        if future:
                            break
                    expansion = _expansion_evidence(obs, phrase, word_ends[ayah], threshold, groups, ayah, word_ends)
                    if not expansion and posterior_expansion:
                        expansion = "canonical_prefix_vs_unresolved_residual"
                    current_expansion = bool(expansion)
                    if not expansion and future_expansion:
                        expansion = "provisional_unique_anchor_partial:" + future_expansion[1]
                    speech_edge = None
                    if not expansion and step + 1 < len(policy.speech_ms):
                        speech_edge = _incomplete_speech_edge(pcm, sr, end, obs, phrase, groups,
                                                             word_ends, ayah, threshold)
                        if speech_edge:
                            expansion = "measured_speech_edge_incomplete_phrase"
                    if expansion:
                        had_prefix |= current_expansion
                        reason = "ambiguous_phrase" if plan["ambiguous"] else "incomplete_distinguishing_phrase"
                        if step + 1 < len(policy.speech_ms):
                            event("EXPAND_LOCAL", ayah, reason=reason, acoustic_anchor=expansion,
                                  speech_edge=speech_edge)
                            continue
                        blocked_prefix |= current_expansion
                    # A previously plausible opening that became unclear cannot
                    # be silently replaced by a later repetition.
                    blocked_prefix = blocked_prefix or had_prefix
                    break
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
                            ms = round(start + obs.positions_ms[offset])
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
                            ms = round(start + obs.positions_ms[offset])
                            phrase_quality = min(obs.unit_quality[occurrences[0]:offset + len(wasl_phrase)], default=0.)
                            if previous < ms < duration and phrase_quality >= threshold:
                                proof = {"phrase": wasl_phrase, "complete": True, "ambiguous": False,
                                         "competitors_checked": True, "offset": offset,
                                         "previous_tail": tail, "quality_metric": adapter.metric,
                                         "contextual_owner": bool(seam_plan)}
                                boundary = Boundary(ayah, ms, "wasl", phrase_quality, read_id=reads.last_read_id, proof=proof)
                    if not boundary:
                        reason = "unresolved_local_block"
                else:
                    reason = "unsupported_wasl_timing"
            if boundary is None:
                if (future and policy.sparse_anchors and future[0] > ayah
                        and (policy.max_unresolved_ayahs is None or future[0] - ayah <= policy.max_unresolved_ayahs)):
                    future_ayah, future_phrase, pause, index, read_id, quality, future_proof = future
                    proof = {"phrase": future_phrase, "complete": True, "ambiguous": False,
                             "competitors_checked": True, "offset": 0, "quality_metric": adapter.metric,
                             "pause_ms": [pause.start_ms, pause.end_ms], "pause_id": index,
                             "sparse_anchor": True, "unresolved_ayahs": list(range(ayah, future_ayah)), **future_proof}
                    anchor = Boundary(future_ayah, round(pause.center_ms), "pause", quality,
                                      read_id=read_id, proof=proof)
                    validate_boundaries([anchor], [future_ayah], duration, reads.calls,
                                        groups=groups, word_ends=word_ends, canonical_ends=canonical_ends, threshold=threshold, pauses=pauses, policy=policy, pause_geometry_data=geometry)
                    confirmed.append(anchor)
                    unresolved[ayah] = "unresolved_before_unique_anchor"
                    event("REVIEW_REQUIRED", ayah, reason=unresolved[ayah], block=proof["unresolved_ayahs"])
                    event("CONFIRMED", future_ayah, evidence="pause", sparse_anchor=True)
                    event("ADVANCE", future_ayah)
                    if ayah_progress:
                        ayah_progress(future_ayah)
                    continue
                event("REVIEW_REQUIRED", ayah, reason=reason)
                break
            event("CONFIRMED", ayah, evidence=boundary.evidence_kind, start_ms=boundary.start_ms)
            confirmed.append(boundary)
            event("ADVANCE", ayah)
            if ayah_progress:
                ayah_progress(ayah)
        result = _finish_estimates(confirmed, ayahs, groups, duration, reason, unresolved)
        event("VALIDATE")
        validate_boundaries(result, ayahs, duration, reads.calls,
                            groups=groups, word_ends=word_ends, canonical_ends=canonical_ends, threshold=threshold, pauses=pauses, policy=policy, pause_geometry_data=geometry)
        trace["boundaries"] = [asdict(b) for b in result]
        trace["ayahs"] = [{"ayah": b.ayah, "onset_ms": b.start_ms, "conf": b.confidence,
                            "reliable": b.evidence_kind != "estimate"} for b in result]
        trace["review_required_ayahs"] = [b.ayah for b in result if b.evidence_kind == "estimate"]
        trace["pause_matches"] = [{"ayah": b.ayah, "from_pause": b.evidence_kind == "pause",
                                   "onset_ms": b.start_ms, "conf": b.confidence,
                                   "boundary_evidence": b.evidence_kind,
                                   **({"pause_start_ms": b.proof["pause_ms"][0],
                                       "pause_end_ms": b.proof["pause_ms"][1]} if b.evidence_kind == "pause" else {})}
                                  for b in result]
        trace.update(n_pause=sum(b.evidence_kind == "pause" for b in result),
                     n_wasl=sum(b.evidence_kind == "wasl" for b in result), bounded_ctc_fallback=None)
        event("RESULT_READY")
        return [(b.ayah, b.start_ms, b.confidence) for b in result]
    except GenerationCancelled:
        events.append({"state": "CANCELLED"})
        raise
    except BaseException as error:
        events.append({"state": "FAILED", "error": str(error)})
        raise
    finally:
        trace.update(reads.diagnostics())
        trace["inference_seconds"] = sum(r.get("seconds", 0.) for r in reads.calls)
