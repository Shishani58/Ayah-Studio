"""Tiled forced alignment — exact alignment in small, self-re-anchoring chunks.

The full path (:func:`surah_cutter.core.align.align_surah`) is accurate but runs
ONE Viterbi over the whole sura: its backpointer matrix is ``frames × (2·tokens+1)``,
which for a 2-hour sura (al-Baqarah: ~354 k frames × ~82 k states) is ~233 GB — the
machine thrashes swap for ~an hour. The fast path
(:func:`surah_cutter.core.align_fast.align_surah_fast`) avoids that by anchoring
ayah boundaries on **pauses**, but on long *wasl*-heavy suras pauses stop being
reliable anchors (≈30 % of boundaries have no pause; the rest are ambiguous), so the
pause→ayah matching drifts and the sura falls back to full — i.e. back to the hour.

This module keeps full's accuracy without its memory blow-up by **tiling** the
exact forced alignment: the sura's known word sequence is cut into contiguous
chunks of a bounded token/duration budget, and each chunk is forced-aligned
*exactly* over a narrow audio window. Crucially the anchor is no longer a fragile
pause but **the previous chunk's exactly-found onset** plus a measured tempo
estimate that only has to place the chunk's words *inside* a padded window — at
chunk granularity the tempo prior is easily accurate enough (per the tempo probe
~0.47 s, while a window has seconds of slack). Each chunk re-locks onto real
acoustic content via Viterbi, so a small window error cannot cascade: there is no
relative placement and no pause guessing, the two sources of fast-mode drift.

Cost: the acoustic model is run over the audio ~once (chunks tile it with a little
overlap; emissions are the irreducible floor for accuracy), and every chunk's
backpointer matrix is bounded (~sura-78 size, ~270 MB, freed between chunks). So
al-Baqarah goes from 233 GB / ~1 h to a bounded few-hundred-MB / ~minutes, at
full-equivalent accuracy.

Output shape matches :func:`align_surah` / :func:`align_surah_fast`:
``[(ayah, start_ms, confidence)]`` with ``start_ms`` monotonic non-decreasing.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Callable

#: Per-chunk token budget (transliteration chars). Caps ``L`` per chunk, hence the
#: forced-align state count ``2·L+1`` and the backpointer matrix. ~1200 keeps a
#: chunk's DP about sura-78 sized (a few hundred MB), freed between chunks.
_MAX_CHUNK_TOKENS = int(os.environ.get("SC_TILED_MAX_TOKENS", "1200"))
#: Per-chunk trusted-span budget (ms). Caps the window length, hence the frame
#: count ``T`` in the chunk DP. Together with the token cap this bounds memory.
_MAX_CHUNK_MS = float(os.environ.get("SC_TILED_MAX_MS", "150000"))
#: Ayahs of already-trusted LEFT context re-included in each chunk (onsets
#: recomputed but discarded) so the first trusted ayah has real lead-in audio.
_CTX_AYAHS = int(os.environ.get("SC_TILED_CTX_AYAHS", "2"))
#: Ayahs of RIGHT context aligned past the trusted span (onsets discarded). The
#: last few slice ayahs drift toward the window's right edge; these absorb that
#: drift so the last TRUSTED ayah stays put. Without enough right buffer the last
#: trusted ayah is dragged late (observed ~12 s with a single right-context ayah).
_CTX_RIGHT = int(os.environ.get("SC_TILED_CTX_RIGHT", "3"))
#: Window right edge = expected onset of the first ayah AFTER the slice (tempo from
#: the chunk's left anchor) + this pad. Ending the window where the NEXT content
#: begins is critical: a window that runs far past the chunk's audio lets forced
#: align pull the chunk's TAIL forward onto later, similar-sounding ayahs (observed
#: 50 s tail drift with a 1.4× window). The pad just covers tempo error so the
#: right-context ayahs still fit.
_WIN_PAD_MS = float(os.environ.get("SC_TILED_WIN_PAD_MS", "5000"))
#: If the last TRUSTED ayah lands within this of the window's right edge, the window
#: was too short (tempo underestimate) → widen and retry. Keeps the trusted interior
#: comfortably inside the window.
_EDGE_GUARD_MS = float(os.environ.get("SC_TILED_EDGE_GUARD_MS", "3000"))
#: Banded CTC radius in extended-target states. Disabled by default because the
#: "tiled" mode is the exact long-sura path; enable only for measured experiments.
_BAND_RADIUS = int(os.environ.get("SC_TILED_BAND_RADIUS", "0"))


def align_surah_tiled(
    pcm,
    sr: int,
    words: list,
    *,
    surah: int | None = None,
    bundle: Any = None,
    trace: dict | None = None,
    log_event: Callable[[str], None] | None = None,
    check_onsets: bool = False,
    ayah_progress: Callable[[int], None] | None = None,
    parallel_full: bool = False,
) -> list[tuple[int, int, float]]:
    """Tiled exact forced alignment; see the module docstring.

    Args:
        pcm, sr, words: as :func:`surah_cutter.core.align.align_surah`.
        surah: sura number (inferred from ``words`` if omitted).
        bundle: pre-loaded emitter bundle to reuse (loaded lazily otherwise).
        trace: optional dict filled with per-chunk diagnostics and totals.
        log_event: optional callback for human-readable realtime diagnostics.

    Returns:
        ``[(ayah, start_ms, conf)]`` for every ayah, ``start_ms`` monotonic.
    """
    import numpy as np

    from surah_cutter.core import align as A
    from surah_cutter.core._ctc import forced_align, forced_align_banded, aligned_segments
    from surah_cutter.core.phonetics import unit_len as _unit_len
    from surah_cutter.data import reference_prior, transliteration

    if not words:
        return []
    if surah is None:
        surah = words[0].surah
    if bundle is None:
        bundle = A.default_bundle()
    if trace is not None:
        trace["model_source"] = A.bundle_source(bundle)
        trace["backend"] = ("torch" if isinstance(bundle, A._Bundle) else
                            getattr(getattr(bundle, "emitter", None), "model_id", "onnx"))
        trace["align_lag_ms"] = getattr(bundle, "lag_ms", A.ALIGN_LAG_MS)
    if getattr(bundle, "text_kind", "phonetic") == "whisper":
        return A.align_surah(pcm, sr, words, model=bundle, surah=surah,
                             trace=trace, log_event=log_event, ayah_progress=ayah_progress)

    def emit(message: str) -> None:
        if log_event is not None:
            log_event(message)

    word_trs = transliteration.surah_word_tr(surah)
    n_words = len(words)
    total_ms = len(pcm) * 1000.0 / sr if sr else 0.0
    lag = getattr(bundle, "lag_ms", A.ALIGN_LAG_MS)

    # ayah -> ordered global word indices.
    ayah_words: "OrderedDict[int, list[int]]" = OrderedDict()
    for i, w in enumerate(words):
        ayah_words.setdefault(w.ayah, []).append(i)
    ayah_list = list(ayah_words.keys())
    n_ayahs = len(ayah_list)

    # Tempo units (long vowels ×2 — recited longer; see phonetics.unit_len).
    # This remains the fallback prior when bundled reference timings are absent
    # or incomplete for a sura.
    text_wunit = {
        a: max(1.0, sum(_unit_len(word_trs[i]) for i in ayah_words[a] if i < len(word_trs)))
        for a in ayah_list
    }
    wunit, cum_u, total_units, prior_source = _position_prior_units(
        ayah_list,
        text_wunit,
        reference_prior.surah_start_fractions(surah),
    )
    # Initial tempo guess (ms per prior unit); refined from real onsets per chunk.
    measured_tempo = total_ms / max(1e-9, total_units)

    a_tokens = {
        a: max(1, sum(len(word_trs[i]) for i in ayah_words[a] if i < len(word_trs)))
        for a in ayah_list
    }
    basmala_default = surah not in (1, 9)

    # Compute emissions once for the whole sura and slice frames for each chunk.
    # ONNX generation normalises the whole waveform before windowing, so this is
    # both faster than overlapping per-chunk inference and closer to the full path.
    emit(
        f"Сура {surah}: tiled, чтение аудио всей суры {total_ms / 1000.0:.2f} сек."
    )
    audio = np.ascontiguousarray(pcm, dtype=np.float32)
    full_emissions = (A._full_emissions(bundle, audio, np, trace=trace) if parallel_full else
                      A._emissions(bundle, audio, np))
    full_n_frames = int(full_emissions.shape[0])
    full_stride = A._stride_ms(len(pcm), sr, full_n_frames)
    onset_pauses = []
    if check_onsets:
        from surah_cutter.core import candidates

        onset_pauses = candidates.detect_energy_pauses(pcm, sr, use_ten_vad=False)

    def sample_to_frame(sample: int) -> int:
        if len(pcm) <= 0 or full_n_frames <= 0:
            return 0
        frame = int(round(sample * full_n_frames / len(pcm)))
        return max(0, min(full_n_frames, frame))

    # --- exact forced-align of words[w0:w1] over audio [t0_ms, t1_ms]. --------
    def align_slice(w0: int, w1: int, t0_ms: float, t1_ms: float, basmala: bool):
        """Return per-word ``start_ms`` (lag-compensated) and ``conf`` for the
        sub-range, or ``None`` on a degenerate window."""
        s0 = max(0, int(round(t0_ms * sr / 1000.0)))
        s1 = min(len(pcm), int(round(t1_ms * sr / 1000.0)))
        if s1 - s0 < int(0.3 * sr):
            return None
        sub_words = words[w0:w1]
        sub_trs = word_trs[w0:w1]
        basmala_trs = A._basmala_trs(transliteration) if basmala else None
        targets, spans = A._build_targets(sub_words, sub_trs, bundle, basmala_trs)
        if not targets or any(end <= start for start, end in spans):
            return None
        f0 = sample_to_frame(s0)
        f1 = sample_to_frame(s1)
        if f1 - f0 < 2:
            return None
        emis = np.ascontiguousarray(full_emissions[f0:f1], dtype=np.float32)
        stride = full_stride
        base = f0 * full_stride
        log_probs = np.ascontiguousarray(emis[None, :, :], dtype=np.float32)
        target_arr = np.asarray(targets, dtype=np.int64)[None, :]
        try:
            if _BAND_RADIUS > 0:
                path, fscores = forced_align_banded(
                    log_probs, target_arr, blank=bundle.blank_id, radius=_BAND_RADIUS
                )
            else:
                raise ValueError("banded CTC disabled")
        except ValueError:
            try:
                path, fscores = forced_align(log_probs, target_arr, blank=bundle.blank_id)
            except ValueError:
                return None  # Grow the chunk rather than fabricate a partial tail.
        path = np.asarray(path).reshape(-1)
        fscores = np.asarray(fscores).reshape(-1)
        try:
            segs = aligned_segments(path, targets, blank=bundle.blank_id)
        except ValueError:
            return None
        tspans = [(s.start, s.end) for s in segs]
        starts: list[float] = []
        confs: list[float] = []
        last = base
        for (ts, te) in spans:
            if te > ts and ts < len(tspans):
                sf = tspans[ts][0]
                ef = tspans[te - 1][1] if te - 1 < len(tspans) else sf
                start = max(0.0, base + sf * stride - lag)
                seg = fscores[sf:max(sf + 1, ef + 1)]
                conf = float(np.exp(seg.mean())) if seg.size else 0.0
                last = start
            else:  # token-less word: carry previous start
                start, conf = last, 0.0
            starts.append(start)
            confs.append(conf)
        checks = {}
        if check_onsets:
            checks = A._check_ayah_onsets(
                sub_words, spans, tspans, fscores, stride, base_ms=base, pauses=onset_pauses,
                lag_ms=lag,
            )
            for idx, word in enumerate(sub_words):
                if idx == 0 or sub_words[idx - 1].ayah != word.ayah:
                    starts[idx] = float(checks[word.ayah]["onset_ms"])
                    confs[idx] = checks[word.ayah]["conf"]
        return starts, confs, checks

    starts_ms: dict[int, float] = {}
    confs: dict[int, float] = {}
    chunks_dbg: list[dict] = []
    trusted_checks: dict[int, dict] = {}

    i = 0  # index into ayah_list of the next ayah to TRUST
    while i < n_ayahs:
        if ayah_progress is not None:
            ayah_progress(ayah_list[i])
        # Trusted span [i, j): accumulate ayahs under the token / duration budget.
        j = i
        tok = 0
        dur = 0.0
        while j < n_ayahs:
            a = ayah_list[j]
            at = a_tokens[a]
            ad = wunit[a] * measured_tempo
            if j > i and (tok + at > _MAX_CHUNK_TOKENS or dur + ad > _MAX_CHUNK_MS):
                break
            tok += at
            dur += ad
            j += 1

        # Left context (already-trusted) and several ayahs of right context.
        lc = max(0, i - _CTX_AYAHS) if i > 0 else 0
        rc = min(n_ayahs, j + _CTX_RIGHT)
        w_lc = ayah_words[ayah_list[lc]][0]
        w_rc = ayah_words[ayah_list[rc - 1]][-1] + 1

        # Window anchored on the previous chunk's exact onset (or 0 for chunk 1).
        if i == 0:
            t0 = 0.0
            basmala = basmala_default
        else:
            t0 = max(0.0, starts_ms[ayah_list[lc]] - 100.0)
            basmala = False
        # Right edge ≈ expected onset of the first ayah AFTER the slice + pad, so
        # the window ends where the next content begins (no long foreign tail).
        if rc >= n_ayahs:
            t1 = total_ms
        else:
            # For the first chunk, empirical prior fractions are measured from
            # the file start and include intro-basmala/silence before ayah 1.
            # Later chunks are anchored on a trusted ayah onset, so use a local
            # delta from the left-context ayah.
            if i == 0:
                next_u = cum_u[ayah_list[rc]]
            else:
                next_u = cum_u[ayah_list[rc]] - cum_u[ayah_list[lc]]
            t1 = min(total_ms, t0 + next_u * measured_tempo + _WIN_PAD_MS)
        t1 = max(t1, t0 + 1000.0)

        res = align_slice(w_lc, w_rc, t0, t1, basmala)
        # Grow-on-edge: if the last TRUSTED ayah landed near the right edge, the
        # window was too short (tempo underestimate) — widen and retry.
        grow = 0
        while res is not None and t1 < total_ms and grow < 3:
            li = ayah_words[ayah_list[j - 1]][0] - w_lc
            last_onset = res[0][li] if 0 <= li < len(res[0]) else t0
            if last_onset < t1 - _EDGE_GUARD_MS:
                break  # comfortably inside → window long enough
            # Grow by half the CURRENT window (bounded), never by a fraction of the
            # distance to the sura end — for an early chunk that would blow the
            # window (and its T×n_st backpointer matrix) up to whole-sura size and
            # spike memory (observed ~9 GB on al-Baqarah).
            t1 = min(total_ms, t1 + max(15000.0, (t1 - t0) * 0.5))
            res = align_slice(w_lc, w_rc, t0, t1, basmala)
            grow += 1
        if res is None:
            # A whole-sura retry defeats the tile memory bound and can allocate
            # hundreds of GiB on long recordings.  Fail explicitly instead of
            # publishing guessed starts or recursively entering unbounded CTC.
            raise RuntimeError(
                "tiled CTC could not align bounded ayah chunk "
                f"{ayah_list[i]}-{ayah_list[j - 1]}"
            )

        slice_starts, slice_confs, slice_checks = res
        for k in range(i, j):  # trust [i, j) only; context onsets discarded
            a = ayah_list[k]
            fw = ayah_words[a][0]
            if ayah_progress is not None:
                ayah_progress(a)
            idx = fw - w_lc
            if 0 <= idx < len(slice_starts):
                starts_ms[a] = slice_starts[idx]
                confs[a] = slice_confs[idx]
                if a in slice_checks:
                    trusted_checks[a] = slice_checks[a]

        if trace is not None:
            chunks_dbg.append({
                "ayahs": [ayah_list[i], ayah_list[j - 1]],
                "window_ms": [round(t0, 1), round(t1, 1)],
                "n_tokens": tok,
                "tempo": round(measured_tempo, 2),
                "prior": prior_source,
            })
        first_word = words[ayah_words[ayah_list[i]][0]].text
        emit(
            f"Чтение аудио {(t1 - t0) / 1000.0:.2f} сек. "
            f"({t0 / 1000.0:.2f}-{t1 / 1000.0:.2f}), "
            f"прочитано слово «{first_word}» - детект аятов "
            f"{ayah_list[i]}-{ayah_list[j - 1]}"
        )

        # Refine tempo from the trusted onsets so far (median consecutive slope —
        # robust to the basmala / silence that skews the global rate).
        pts = [(cum_u[a], starts_ms[a]) for a in ayah_list[: j] if a in starts_ms]
        slopes = sorted(
            (t1b - t0b) / (u1 - u0)
            for (u0, t0b), (u1, t1b) in zip(pts, pts[1:]) if u1 > u0 and t1b >= t0b
        )
        if slopes:
            measured_tempo = slopes[len(slopes) // 2]

        i = j

    # Assemble rows, enforce monotonic non-decreasing starts (same as the others).
    rows: list[tuple[int, int, float]] = []
    prev = 0
    for a in ayah_list:
        start = int(round(starts_ms.get(a, prev)))
        if start < prev:
            start = prev
        prev = start
        rows.append((a, start, round(float(confs.get(a, 0.0)), 4)))

    if trace is not None:
        trace.update(
            mode="tiled",
            total_ms=total_ms,
            n_chunks=len(chunks_dbg),
            n_ayahs=n_ayahs,
            n_words=n_words,
            prior=prior_source,
            emissions_frames=full_n_frames,
            chunks=chunks_dbg,
            emission_passes=1,
            ayahs=[trusted_checks.get(a, {"ayah": a, "onset_ms": ms, "conf": conf})
                   for a, ms, conf in rows],
        )
    return rows


def _position_prior_units(
    ayah_list: list[int],
    text_wunit: dict[int, float],
    start_fracs: dict[int, float],
) -> tuple[dict[int, float], dict[int, float], float, str]:
    """Build duration/start units for tiled window sizing.

    Prefer bundled-reference median start fractions when they cover every ayah
    in the sura. Fall back to the text-length units used by the original tiled
    implementation. The returned units are only a window prior; CTC alignment
    still places the actual starts.
    """
    if ayah_list and all(a in start_fracs for a in ayah_list):
        cum_u = {a: float(start_fracs[a]) for a in ayah_list}
        wunit: dict[int, float] = {}
        gaps: list[float] = []
        for idx, a in enumerate(ayah_list):
            if idx + 1 < len(ayah_list):
                gap = cum_u[ayah_list[idx + 1]] - cum_u[a]
            else:
                gap = 1.0 - cum_u[a]
            if gap > 0:
                gaps.append(gap)
            wunit[a] = gap
        fallback_gap = sorted(gaps)[len(gaps) // 2] if gaps else 1.0 / len(ayah_list)
        floor = min(fallback_gap, 1.0 / len(ayah_list)) * 0.25
        for a in ayah_list:
            wunit[a] = max(floor, wunit[a])
        return wunit, cum_u, 1.0, "references"

    cum_u: dict[int, float] = {}
    acc = 0.0
    for a in ayah_list:
        cum_u[a] = acc
        acc += text_wunit[a]
    return dict(text_wunit), cum_u, max(1.0, acc), "text"
