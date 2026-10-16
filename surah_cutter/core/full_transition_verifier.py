"""Optional installed Muaalem proof for an ambiguous local FULL transition."""

import numpy as np

from surah_cutter.core.cancellation import GenerationCancelled, check_cancelled


def _installed_verifier():
    from surah_cutter.core import model_store
    from surah_cutter.core.full_boundary import _muaalem_verifier

    if not model_store.is_installed(model_store.spec("muaalem")):
        return None
    try:
        return _muaalem_verifier()
    except ValueError as error:
        if str(error).startswith("Для проверки спорной границы FULL установите Muaalem"):
            return None  # Installed files without a ready runtime are not usable.
        raise


def verify_transition(pcm, sr, words, ayah, lo, hi, checks, pause, *, reads, trace=None, log_event=None):
    """Require an exact, unique greedy phrase and strong tokens on BOTH sides.

    Phonetize complete canonical ayahs, then slice the selected words. Passing
    a partial ayah to the phonetizer invents a start/waqf and changes its sounds.
    No text edits, duration normalization, model download or primary switch.
    """
    from surah_cutter.core import align
    from surah_cutter.core._ctc import aligned_segments, forced_align

    check_cancelled()
    groups = {}
    for word in words:
        groups.setdefault(word.ayah, []).append(word)
    left, right = groups[ayah - 1][-2:], groups[ayah][:2]
    pair = tuple(w.text for w in left + right)
    # Repeated adjacent phrases cannot establish an independent owner.
    if sum(tuple(w.text for w in groups[a - 1][-2:] + groups[a][:2]) == pair
           for a in groups if a - 1 in groups) != 1:
        return None
    # QPC can place the final phoneme of a short ayah too early. Include its
    # real end using the following ayah's opening, while keeping the 12 s cap.
    if len(groups[ayah]) <= 2 and ayah + 1 in checks:
        hi = max(hi, checks[ayah + 1]["onset_ms"] + 400)
    hi = min(hi, len(pcm) * 1000 / sr)
    if not 1000 <= hi - lo <= 12000:
        return None
    loaded = _installed_verifier()
    if trace is not None:
        trace["full_transition_verifier"] = "installed" if loaded else "unavailable"
    if loaded is None:
        return None
    bundle, source = loaded
    if trace is not None:
        trace.update(full_boundary_verifier_source=source,
                     full_boundary_verifier_identity=bundle.emitter.runtime_identity)
    complete = groups[ayah - 1] + groups[ayah]
    targets, ws = align._build_targets(complete, [], bundle, None)
    first_word, stop_word = len(groups[ayah - 1]) - len(left), len(groups[ayah - 1]) + len(right)
    start, stop = ws[first_word][0], ws[stop_word - 1][1]
    selected_spans = [(a - start, b - start) for a, b in ws[first_word:stop_word]]
    targets = targets[start:stop]
    crop = pcm[round(lo * sr / 1000):round(hi * sr / 1000)]
    record = dict(ayah=ayah, model="muaalem", role="two-sided-verifier", source=source,
                  start_ms=lo, end_ms=hi, target_words=[w.index for w in left + right],
                  targets=targets, target_context="complete-canonical-ayahs",
                  normalization="model-runtime",
                  pause_start_ms=pause[0], pause_end_ms=pause[1],
                  frames=0, outcome="running", accepted=False)
    check_cancelled()
    reads.append(record)
    if log_event is not None:
        log_event(f"FULL: независимая проверка перехода {ayah - 1}→{ayah}, "
                  f"Muaalem {lo / 1000:.3f}–{hi / 1000:.3f} сек.")
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
    ids = matrix.argmax(axis=1)
    greedy = [int(t) for i, t in enumerate(ids) if t != bundle.blank_id and (i == 0 or t != ids[i - 1])]
    matches = [i for i in range(len(greedy) - len(targets) + 1) if greedy[i:i + len(targets)] == targets]
    record.update(decoded_tokens=greedy, phrase_matches=matches)
    if len(matches) != 1 or not targets:
        record["reason"] = "verifier-phrase-mismatch"
        return None
    try:
        path, scores = forced_align(matrix[None], np.asarray(targets)[None], blank=bundle.blank_id)
        segments = aligned_segments(path.reshape(-1), targets, blank=bundle.blank_id)
    except ValueError:
        record["reason"] = "insufficient-verifier-path"
        return None
    quality = [align._span_confidence(scores.reshape(-1), s.start, s.end) for s in segments]
    record["token_conf"] = quality
    if min(quality, default=0) < .45:
        record["reason"] = "weak-verifier-phrase"
        return None
    stride = align._stride_ms(len(crop), sr, len(matrix))
    tail = lo + (segments[selected_spans[len(left) - 1][1] - 1].end + 1) * stride
    first = lo + segments[selected_spans[len(left)][0]].start * stride
    record.update(proof_supported=True, reason="verifier-phrase-confirmed", tail_ms=tail, first_ms=first)
    return dict(tail_ms=tail, first_ms=first, onset_ms=first - bundle.lag_ms, record=record, source=source)
