"""QPC FULL phrase recheck for first words split across a confirmed pause."""

from __future__ import annotations

import numpy as np

from surah_cutter.core.cancellation import GenerationCancelled, check_cancelled


def _consonants(text: str) -> str:
    letters = [c for c in text if c.isalpha() and c not in "aeiouāīū"]
    # Greedy CTC can merge geminated consonants. Targets keep their exact
    # spelling; this normalization is used only for independent phrase matching.
    return "".join(c for i, c in enumerate(letters) if i == 0 or c != letters[i - 1])


def _muaalem_verifier():
    """Load only the installed verifier; never select it or provision a runtime."""
    import hashlib
    import json
    import os
    from pathlib import Path

    from surah_cutter.core import align, model_runtime, model_store

    spec = model_store.spec("muaalem")
    env = model_runtime.environment("muaalem")
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = env / "ayah-runtime.json"
    lock = Path(model_runtime.__file__).with_name("runtime_locks") / "model-runtime.txt"
    try:
        ready = (python.is_file() and json.loads(marker.read_text()).get("requirements")
                 == hashlib.sha256(lock.read_bytes()).hexdigest())
    except (OSError, ValueError):
        ready = False
    if not model_store.is_installed(spec) or not ready:
        raise ValueError("Для проверки спорной границы FULL установите Muaalem во вкладке «Модели».")
    emitter = model_runtime.load_runtime("muaalem", model_store.installed_path(spec))
    return align._OnnxBundle(emitter), f"muaalem@{spec['revision']}"


def _recover_long_split(pcm, sr, selected_words, pause, lo, hi, checks, check, reads, trace,
                        log_event=None):
    """Prove an ambiguous opening with local phonemes, then cut inside silence."""
    from surah_cutter.core import align
    from surah_cutter.core._ctc import aligned_segments, forced_align
    from surah_cutter.data import quran_text

    # A short intraword interruption is not evidence of a wrong owner.
    if pause.depth_ms < 1200 or check["conf"] >= .25:
        return
    check_cancelled()
    if trace is not None:
        trace["full_boundary_verifier_requested"] = "muaalem"
    if log_event is not None:
        log_event(f"FULL: локальная проверка фразы аята {selected_words[0].ayah}, "
                  f"Muaalem {lo / 1000:.3f}–{hi / 1000:.3f} сек.")
    verifier, source = _muaalem_verifier()
    if trace is not None:
        trace["full_boundary_verifier_source"] = source
        trace["full_boundary_verifier_identity"] = verifier.emitter.runtime_identity
    crop = pcm[round(lo * sr / 1000):round(hi * sr / 1000)]
    record = {"ayah": selected_words[0].ayah, "model": "muaalem", "role": "opening-proof",
              "start_ms": lo, "end_ms": hi, "frames": 0, "accepted": False,
              "outcome": "running", "source": source, "normalization": "model-runtime"}
    check_cancelled()
    reads.append(record)
    try:
        matrix = verifier.emitter.emissions(crop)
        check_cancelled()
    except GenerationCancelled:
        record["outcome"] = "cancelled"
        raise
    except Exception:
        record["outcome"] = "failed"
        raise
    record.update(outcome="completed", frames=len(matrix))
    targets, word_spans = align._build_targets(selected_words, [], verifier, None)
    required = word_spans[1][1]  # At least two complete canonical words.
    # A shared opening needs the first differing phoneme, never just "inna...".
    all_words = quran_text.surah_words(selected_words[0].surah)
    first = {}
    for i, word in enumerate(all_words):
        first.setdefault(word.ayah, i)
    for ayah, start in first.items():
        if ayah == selected_words[0].ayah:
            continue
        rival_words = [w for w in all_words[start:start + 3] if w.ayah == ayah]
        rival, _ = align._build_targets(rival_words, [], verifier, None)
        common = 0
        for a, b in zip(targets, rival):
            if a != b:
                break
            common += 1
        if common == min(len(targets), len(rival)):
            record["reason"] = "indistinguishable-canonical-opening"
            return
        required = max(required, common + 1)
    ids = matrix.argmax(axis=1)
    greedy = [int(token) for i, token in enumerate(ids)
              if token != verifier.blank_id and (i == 0 or token != ids[i - 1])]
    record.update(required_tokens=required, targets=targets, decoded_tokens=greedy)
    if greedy[:required] != targets[:required] or len(greedy) < required:
        record["reason"] = "verifier-greedy-mismatch"
        return
    try:
        path, scores = forced_align(matrix[None], np.asarray(targets)[None], blank=verifier.blank_id)
        segments = aligned_segments(path.reshape(-1), targets, blank=verifier.blank_id)
    except ValueError:
        record["reason"] = "insufficient-verifier-path"
        return
    spans = [(s.start, s.end) for s in segments]
    scores = scores.reshape(-1)
    quality = [align._span_confidence(scores, sf, ef) for sf, ef in spans[:required]]
    stride = align._stride_ms(len(crop), sr, len(matrix))
    speech_start = lo + spans[0][0] * stride
    record.update(essential_token_conf=quality, speech_start_ms=speech_start)
    if min(quality, default=0) < .45 or not pause.end_ms - 100 <= speech_start <= pause.end_ms + 400:
        record["reason"] = "weak-verifier-opening"
        return
    start_index = next(i for i, w in enumerate(all_words) if w.index == selected_words[0].index)
    if start_index == 0:
        record["reason"] = "no-canonical-left-tail"
        return
    previous_ayah = all_words[start_index - 1].ayah
    previous_words = [w for w in all_words[:start_index] if w.ayah == previous_ayah][-2:]
    closing_targets, _ = align._build_targets(previous_words, [], verifier, None)
    current_tail = [w for w in all_words if w.ayah == selected_words[0].ayah][-2:]
    current_tail_targets, _ = align._build_targets(current_tail, [], verifier, None)
    if closing_targets == current_tail_targets:
        record["reason"] = "indistinguishable-left-owner"
        return
    left_lo = max(0, pause.start_ms - 5800)
    left_hi = pause.start_ms + 100
    left_crop = pcm[round(left_lo * sr / 1000):round(left_hi * sr / 1000)]
    closing = {"ayah": previous_ayah, "model": "muaalem", "role": "closing-proof",
               "start_ms": left_lo, "end_ms": left_hi, "frames": 0, "accepted": False,
               "outcome": "running", "source": source, "targets": closing_targets}
    if log_event is not None:
        log_event(f"FULL: проверка конца предыдущего аята, "
                  f"Muaalem {left_lo / 1000:.3f}–{left_hi / 1000:.3f} сек.")
    check_cancelled()
    reads.append(closing)
    try:
        left_matrix = verifier.emitter.emissions(left_crop)
        check_cancelled()
    except GenerationCancelled:
        closing["outcome"] = "cancelled"
        raise
    except Exception:
        closing["outcome"] = "failed"
        raise
    closing.update(outcome="completed", frames=len(left_matrix))
    left_ids = left_matrix.argmax(axis=1)
    left_greedy = [int(token) for i, token in enumerate(left_ids)
                   if token != verifier.blank_id and (i == 0 or token != left_ids[i - 1])]
    closing["decoded_tokens"] = left_greedy
    if left_greedy[-len(closing_targets):] != closing_targets:
        closing["reason"] = record["reason"] = "previous-tail-greedy-mismatch"
        return
    try:
        left_path, left_scores = forced_align(left_matrix[None], np.asarray(closing_targets)[None],
                                             blank=verifier.blank_id)
        left_segments = aligned_segments(left_path.reshape(-1), closing_targets, blank=verifier.blank_id)
    except ValueError:
        closing["reason"] = record["reason"] = "insufficient-left-path"
        return
    left_quality = [align._span_confidence(left_scores.reshape(-1), s.start, s.end) for s in left_segments]
    closing["essential_token_conf"] = left_quality
    if min(left_quality, default=0) < .45:
        closing["reason"] = record["reason"] = "weak-previous-tail"
        return
    closing.update(proof_supported=True, reason="previous-ayah-tail-confirmed")
    # The verifier identifies the phrase; DSP supplies a safe pre-speech cut.
    # Do not transplant another model's uncalibrated timing/lag into QPC.
    new_ms = round(pause.end_ms - 100)
    ayah = selected_words[0].ayah
    if any((a < ayah and c["onset_ms"] >= new_ms) or (a > ayah and c["onset_ms"] <= new_ms)
           for a, c in checks.items()):
        record["reason"] = "conflicting-ayah-order"
        return
    old_ms = check["onset_ms"]
    check.update(onset_ms=new_ms, repaired=True, reason="split-word-muaalem-proof",
                 original_conf=check["conf"], original_first_token_conf=check["first_token_conf"],
                 conf=min(quality + left_quality), first_token_conf=quality[0], token_confs=quality[:word_spans[0][1]],
                 reliable=False, pause_start_ms=pause.start_ms, pause_end_ms=pause.end_ms,
                 boundary_verifier=source)
    record.update(accepted=True, reason="split-word-muaalem-proof", before_ms=old_ms, after_ms=new_ms)


def recheck_split_words(
    bundle, pcm, sr, words, word_spans, spans, frame_scores, stride_ms, checks,
    pauses, *, base_ms=0, trace=None, log_event=None,
):
    from surah_cutter.core import align, candidates
    from surah_cutter.core._ctc import aligned_segments, forced_align
    from surah_cutter.data import transliteration

    reads = trace.setdefault("full_boundary_reads", []) if trace is not None else []
    trs = transliteration.surah_word_tr(words[0].surah)
    inverse = {v: k for k, v in bundle.vocab.items()}
    seen = set()
    for wi, word in enumerate(words):
        check_cancelled()
        if word.ayah in seen:
            continue
        seen.add(word.ayah)
        check = checks[word.ayah]
        if check["repaired"] or wi + 1 >= len(words) or words[wi + 1].ayah != word.ayah:
            continue
        ts, te = word_spans[wi]
        first = base_ms + spans[ts][0] * stride_ms
        last = base_ms + spans[te - 1][0] * stride_ms
        split = [p for p in pauses if first < p.start_ms < p.end_ms <= last and p.depth_ms >= 300]
        if len(split) != 1:
            continue
        pause = split[0]
        if not candidates._dsp_accepts_pause(candidates._dsp_pause_metrics(pcm, sr, pause)):
            continue
        # A supported consonant on the left may belong to an interrupted word.
        # Only vowels/formatting or unsupported consonants may be reassigned.
        first_targets, _ = align._build_targets([word], [trs[word.index]], bundle, None)
        if any(
            _consonants(inverse[token]) and align._span_confidence(frame_scores, sf, ef) >= .25
            for token, (sf, ef) in zip(first_targets, spans[ts:te])
            if base_ms + sf * stride_ms < pause.end_ms
        ):
            continue
        count = 2
        while (count < 3 and wi + count < len(words) and words[wi + count].ayah == word.ayah
               and len("".join(_consonants(trs[w.index]) for w in words[wi:wi + count])) < 6):
            count += 1
        selected_words = words[wi:wi + count]
        selected_trs = [trs[w.index] for w in selected_words]
        opening = "".join(_consonants(tr) for tr in selected_trs)
        if not _consonants(selected_trs[0]) or len(opening) < 6:
            continue
        context_end = base_ms + (spans[word_spans[wi + count - 1][1] - 1][1] + 1) * stride_ms + 100
        lo = round(max(pause.center_ms, pause.end_ms - 200) / 20) * 20
        hi = round(min(len(pcm) * 1000 / sr, context_end) / 20) * 20
        if not 500 <= hi - lo <= 6000:
            continue
        crop = pcm[round(lo * sr / 1000):round(hi * sr / 1000)]
        record = {"ayah": word.ayah, "start_ms": lo, "end_ms": hi,
                  "frames": 0, "accepted": False, "outcome": "running",
                  "target": selected_trs, "normalization": "local"}
        if log_event is not None:
            log_event(f"FULL: перепроверка начальной фразы аята {word.ayah}, "
                      f"QPC {lo / 1000:.3f}–{hi / 1000:.3f} сек.")
        check_cancelled()
        reads.append(record)
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
        decoded = "".join(inverse[int(token)] for i, token in enumerate(ids)
                          if token != bundle.blank_id and (i == 0 or token != ids[i - 1]))
        record["decoded"] = decoded
        if not _consonants(decoded).startswith(opening):
            record["reason"] = "greedy-opening-mismatch"
            _recover_long_split(pcm, sr, selected_words, pause, lo, hi, checks, check, reads, trace,
                                log_event=log_event)
            continue
        targets, local_words = align._build_targets(selected_words, selected_trs, bundle, None)
        try:
            path, scores = forced_align(matrix[None], np.asarray(targets)[None], blank=bundle.blank_id)
            segments = aligned_segments(path.reshape(-1), targets, blank=bundle.blank_id)
        except ValueError:
            record["reason"] = "insufficient-local-path"
            continue
        local_spans = [(s.start, s.end) for s in segments]
        local_stride = align._stride_ms(len(crop), sr, len(matrix))
        local = align._check_ayah_onsets(
            selected_words, local_words, local_spans, scores.reshape(-1), local_stride,
            pauses=[], base_ms=lo, lag_ms=bundle.lag_ms,
        )[word.ayah]
        chars = [inverse[targets[i]] for i in range(*local_words[0])]
        consonant_conf = [c for char, c in zip(chars, local["token_confs"]) if _consonants(char)]
        first_consonant = next(i for i, char in enumerate(chars) if _consonants(char))
        body_conf = [c for char, c in zip(chars[first_consonant:], local["token_confs"][first_consonant:])
                     if char.isalpha()]
        record.update(local_check=local, consonant_conf=consonant_conf, body_token_conf=body_conf)
        if (min(consonant_conf, default=0) < .45 or local["conf"] < .25
                or min(body_conf, default=0) < .25
                or not pause.center_ms <= local["onset_ms"] <= pause.end_ms + 200):
            record["reason"] = "insufficient-local-word-evidence"
            continue
        new_ms = local["onset_ms"]
        if any((a < word.ayah and c["onset_ms"] >= new_ms)
               or (a > word.ayah and c["onset_ms"] <= new_ms) for a, c in checks.items()):
            record["reason"] = "conflicting-ayah-order"
            continue
        old_ms = check["onset_ms"]
        check.update(
            onset_ms=new_ms, conf=local["conf"], first_token_conf=local["first_token_conf"],
            token_confs=local["token_confs"], reliable=check["reliable"] and local["reliable"],
            repaired=True, reason="split-word-local-recheck", original_conf=check["conf"],
            original_first_token_conf=check["first_token_conf"],
            pause_start_ms=pause.start_ms, pause_end_ms=pause.end_ms,
        )
        record.update(accepted=True, reason="split-word-local-recheck", before_ms=old_ms, after_ms=new_ms)
