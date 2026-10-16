"""Conservative QPC FULL repair of a cut that misses an earlier quiet shelf."""

from __future__ import annotations

import numpy as np

from surah_cutter.core.cancellation import GenerationCancelled, check_cancelled


def refine_late_boundaries(bundle, pcm, sr, words, checks, *, trace=None, log_event=None):
    """Re-align both neighbours before moving a late cut into proven silence.

    Local and global CTC are correlated evidence, not independent recognition.
    Require matching speech anchors and argmax support in both phrases. A
    quiet shelf must pass DSP in both original contexts and the joint crop.
    Never move a repaired cut, move right, or promote acoustic confidence.
    """
    from surah_cutter.core import align, candidates
    from surah_cutter.core._ctc import aligned_segments, forced_align
    from surah_cutter.core.full_boundary import _consonants
    from surah_cutter.core.full_pause_center import record_edges
    from surah_cutter.data import transliteration

    policy = dict(max_crop_ms=12000, context_ms=400, anchor_tolerance_ms=200,
                  min_anchor_conf=.45, min_distinct_consonants=2, min_edge_sounds=1,
                  max_lead_ms=250, margin_ms=20, verifier="muaalem-if-installed")
    if trace is not None:
        trace["full_transition_policy"] = policy
    reads = trace.setdefault("full_boundary_reads", []) if trace is not None else []
    groups = {}
    for word in words:
        groups.setdefault(word.ayah, []).append(word)
    trs = None  # Do not load transliterations unless a candidate needs inference.
    for ayah in sorted(checks):
        check_cancelled()
        check = checks[ayah]
        previous = checks.get(ayah - 1)
        if (previous is None or check.get("repaired")
                or check.get("pause_center_reason") == "positioned"
                or check.get("first_token_conf", 0) < .25
                or previous.get("full_ctc_tail_conf", 0) < .25):
            continue
        tail, first = previous.get("full_ctc_end_ms"), check.get("full_ctc_first_ms")
        lo, hi = previous.get("full_ctc_tail_start_ms"), check.get("full_ctc_head_end_ms")
        raw = check.get("raw_onset_ms")
        if None in (tail, first, lo, hi, raw) or not 0 < first - tail <= 6000:
            continue
        margin = policy["margin_ms"]
        # Only an earlier shelf missed by the selected narrow context. Reject
        # intraword shelves after the compensated opening, even if very quiet.
        shelves = sorted({(p["start_ms"], p["end_ms"])
                          for key in ("pause_local_decisions", "pause_stability_decisions")
                          for p in check.get(key, [])
                          if p.get("accepted") and p.get("owned")
                          and tail <= p["start_ms"] < p["end_ms"] <= raw + margin
                          and p["end_ms"] - p["start_ms"] >= 120
                          and p["end_ms"] - margin < check["onset_ms"]})
        if len(shelves) != 1:
            continue
        start, end = shelves[0]
        lo = max(0, int((lo - policy["context_ms"]) // 20) * 20)
        hi = min(len(pcm) * 1000 / sr, int((hi + policy["context_ms"]) // 20) * 20)
        if not 1000 <= hi - lo <= policy["max_crop_ms"]:
            continue
        contexts = [check.get(k) for k in ("pause_local_crop_ms", "pause_stability_crop_ms")]
        contexts.append([lo, hi])
        if any(c is None or not c[0] <= start < end <= c[1] for c in contexts):
            continue
        metrics = [candidates._dsp_pause_metrics(
            pcm[round(a * sr / 1000):round(b * sr / 1000)], sr,
            candidates.Pause(start - a, end - a, end - start)) for a, b in contexts]
        if not all(candidates._dsp_accepts_pause(m) for m in metrics):
            continue
        selected = groups[ayah - 1][-2:] + groups[ayah][:2]
        if trs is None:
            trs = transliteration.surah_word_tr(words[0].surah)
        targets, word_spans = align._build_targets(selected, [trs[w.index] for w in selected], bundle, None)
        crop = pcm[round(lo * sr / 1000):round(hi * sr / 1000)]
        record = dict(ayah=ayah, role="two-sided-cut", start_ms=lo, end_ms=hi,
                      outcome="running", accepted=False, frames=0, normalization="local",
                      word_indices=[w.index for w in selected], target=[trs[w.index] for w in selected],
                      pause_start_ms=start, pause_end_ms=end, dsp_context_metrics=metrics)
        check_cancelled()
        reads.append(record)
        if log_event is not None:
            log_event(f"FULL: проверка перехода {ayah - 1}→{ayah}, "
                      f"QPC {lo / 1000:.3f}–{hi / 1000:.3f} сек.")
        try:
            matrix = bundle.emitter.emissions(crop)
            check_cancelled()
        except GenerationCancelled:
            record["outcome"] = "cancelled"
            raise
        except Exception:
            record["outcome"] = "failed"
            raise
        record.update(outcome="completed", frames=len(matrix))
        try:
            path, scores = forced_align(matrix[None], np.asarray(targets)[None], blank=bundle.blank_id)
            segments = aligned_segments(path.reshape(-1), targets, blank=bundle.blank_id)
        except ValueError:
            record["reason"] = "insufficient-transition-path"
            continue
        spans = [(s.start, s.end) for s in segments]
        scores = scores.reshape(-1)
        stride = align._stride_ms(len(crop), sr, len(matrix))
        local = align._check_ayah_onsets(selected, word_spans, spans, scores, stride,
                                       pauses=[], base_ms=lo, lag_ms=bundle.lag_ms)
        record_edges(local, selected, word_spans, spans, scores, stride, base_ms=lo)
        left, right = local[ayah - 1], local[ayah]
        inverse = {v: k for k, v in bundle.vocab.items()}
        greedy = matrix.argmax(axis=1)
        support = {a: set() for a in (ayah - 1, ayah)}
        edge_support = {a: set() for a in (ayah - 1, ayah)}
        edge_words = {groups[ayah - 1][-1].index, groups[ayah][0].index}
        for word, (ts, te) in zip(selected, word_spans):
            for token, (sf, ef) in zip(targets[ts:te], spans[ts:te]):
                consonant = _consonants(inverse[token])
                if (inverse[token].isalpha() and align._span_confidence(scores, sf, ef) >= policy["min_anchor_conf"]
                        and np.any(greedy[sf:ef + 1] == token)):
                    if consonant:
                        support[word.ayah].add(consonant)
                    if word.index in edge_words:
                        edge_support[word.ayah].add(inverse[token])
        record.update(local_checks=list(local.values()), distinct_consonants={a: len(v) for a, v in support.items()},
                      edge_sounds={a: len(v) for a, v in edge_support.items()})
        decision, reason, verified = record, "two-sided-pause-recheck", None
        conflict = (right["first_token_conf"] < policy["min_anchor_conf"]
                or left.get("full_ctc_tail_conf", 0) < policy["min_anchor_conf"]
                or min(map(len, support.values())) < policy["min_distinct_consonants"]
                or min(map(len, edge_support.values())) < policy["min_edge_sounds"]
                or abs(right["full_ctc_first_ms"] - first) > policy["anchor_tolerance_ms"]
                or abs(left["full_ctc_end_ms"] - tail) > policy["anchor_tolerance_ms"]
                or left["full_ctc_end_ms"] > start
                or right["full_ctc_first_ms"] < end)
        if conflict:
            record["reason"] = "conflicting-transition-evidence"
            from surah_cutter.core.full_transition_verifier import verify_transition

            verified = verify_transition(pcm, sr, words, ayah, lo, hi, checks, (start, end),
                                         reads=reads, trace=trace, log_event=log_event)
            if verified is None:
                continue
            decision, reason = verified["record"], "two-sided-muaalem-recheck"
            # An exact independent phrase may resolve a wrong GLOBAL tail,
            # but must agree with the locally reread speech on both sides.
            if (right["first_token_conf"] < policy["min_anchor_conf"]
                    or left.get("full_ctc_tail_conf", 0) < policy["min_anchor_conf"]
                    or abs(right["full_ctc_first_ms"] - first) > policy["anchor_tolerance_ms"]
                    or abs(verified["first_ms"] - right["full_ctc_first_ms"]) > policy["anchor_tolerance_ms"]
                    or abs(verified["tail_ms"] - left["full_ctc_end_ms"]) > policy["anchor_tolerance_ms"]
                    or max(left["full_ctc_end_ms"], verified["tail_ms"]) > start
                    or min(right["full_ctc_first_ms"], verified["first_ms"]) < end):
                decision["reason"] = "conflicting-independent-transition"
                continue
        # DSP measures the middle of the shelf, excluding smoothed speech edges.
        # Keep the cut inside that measured region, not just inside its bounds.
        guard = max(margin, min(80, (end - start) * .22))
        safe_start = max(start + guard, end - policy["max_lead_ms"], tail, left["full_ctc_end_ms"])
        safe_end = min(end - guard, raw - margin, right["onset_ms"] - margin)
        if verified is not None:
            safe_start = max(safe_start, verified["tail_ms"])
            safe_end = min(safe_end, verified["onset_ms"] - margin)
        desired = end - min((end - start) / 2, policy["max_lead_ms"])
        new_ms = round(max(safe_start, min(desired, safe_end)))
        following = checks.get(ayah + 1)
        if (safe_end - safe_start < 40 or not previous["onset_ms"] < new_ms < check["onset_ms"]
                or (following is not None and new_ms >= following["onset_ms"])):
            decision["reason"] = "no-safe-transition-cut"
            continue
        check_cancelled()
        decision.update(accepted=True, reason=reason, before_ms=check["onset_ms"],
                      after_ms=new_ms, safe_interval_ms=[safe_start, safe_end])
        check.update(onset_ms=new_ms, repaired=True, reliable=False, reason=reason,
                     transition_pause_start_ms=start, transition_pause_end_ms=end)
        if verified is not None:
            check["boundary_verifier"] = verified["source"]
