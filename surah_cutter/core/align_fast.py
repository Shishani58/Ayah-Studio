"""Fast two-phase forced alignment — pause onsets + growing neural windows.

The accurate path (:func:`surah_cutter.core.align.align_surah`) runs the acoustic
model over the **whole** sura. This module saves inference time by running the
model only over **short windows**, sized up on demand. The boundary is still
placed by the neural aligner (not by a pause), so accuracy is preserved.

Two explicit phases (the UI exposes both so the user can steer them):

**Phase 1 — rough pass by pauses (cheap, no neural).**
Silero VAD gives silence segments. For each ayah boundary we take the *onset of
the wave* after the pause (``pause.end_ms``, where speech resumes) — NOT the
quiet centre — matched to the ayah's expected position (structural prior). Ayahs
with no nearby pause are *wasl* (joined) — their rough guess comes from the prior
plus the running tempo.

**Phase 2 — neural refinement with a growing window (from 1 s).**
For each boundary we read a **1-second** window around its rough position, run
the model on that slice only, forced-align the local words, and take the start of
the ayah's first word. If the alignment is confident and the onset sits
comfortably inside the window → keep it. If not → grow the window by 1 second and
re-read, up to ``max_win_s``. Most boundaries resolve at 1-2 s, so the total
audio fed to the model is far less than one full pass → genuinely faster.

No cascade: each ayah is anchored by *its own* pause, so one bad onset does not
poison the next. The only cross-ayah coupling is a final monotonic clamp.

Returns the same shape as :func:`align.align_surah`: ``[(ayah, start_ms, conf)]``.
"""

from __future__ import annotations

import difflib
from collections import OrderedDict
from typing import Any, Callable


def _collapse(s: str) -> str:
    """Collapse runs of the same character (melodic vowel elongation -> 1)."""
    out: list[str] = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


#: Chars of the (collapsed) decode/opening compared. Must span ~4 words so that
#: ayahs sharing an opening phrase (e.g. al-Bayyinah 6/7 "inna lladhīna kafarū"
#: vs "…āmanū", al-Falaq's repeated "wa-min sharri…") are told apart by a later,
#: differing word — the shared prefix alone is ambiguous.
_SIM_CHARS = 26


def _sim(dec: str, opening: str) -> float:
    """Fuzzy similarity of a greedy-decoded onset to an ayah's opening tokens.

    Both are collapsed (recitation stretches vowels) and capped at
    :data:`_SIM_CHARS` chars. Returns ``[0, 1]``.
    """
    return difflib.SequenceMatcher(
        None, _collapse(dec)[:_SIM_CHARS], _collapse(opening)[:_SIM_CHARS]
    ).ratio()


def _default_bundle() -> Any:
    from surah_cutter.core import align as A

    if A.DEFAULT_BACKEND == "onnx":
        return A._get_onnx_bundle()
    return A._get_bundle(A.DEFAULT_MODEL_ID)


def _match_boundaries(
    exp: list[float],
    ons: list[float],
    *,
    anchor: float,
    wasl_pen: float,
    extra_pen: float,
    match_cost: Callable[[int, int], float] | None = None,
    dur_floor: float = 0.4,
    dur_ceil: float = 6.0,
    abs_floor: float = 650.0,
    abs_ceil: float = 4500.0,
    max_skip: int = 6,
) -> list[float | None]:
    """Order-preserving match of boundaries ``exp`` to pause onsets ``ons``.

    Global DP. The per-pair *match quality* term is supplied by ``match_cost``
    (lower = better fit of pause ``j`` to boundary ``i``). When given, this is an
    **acoustic** cost — how well the words actually heard right after pause ``j``
    match ayah ``i``'s opening — which is bias-free (it does not depend on the
    char prior's absolute or relative position guesses, which drift by seconds in
    melodic recitation and caused the off-by-one runs). When ``match_cost`` is
    ``None`` the legacy positional cost is used: a boundary's predicted position
    is the *previously matched pause* plus the prior's *expected gap*
    (``exp[i] - exp[i']``); kept as a fallback.

    Either way a **duration floor/ceiling** rejects implausible matches: a matched
    ayah span shorter than ``dur_floor`` × its char-expected duration (an early
    grab of a within-ayah waqf / breath pause) or longer than ``dur_ceil`` × (a
    far jump past the real pause) is forbidden. ``exp`` is used only for these
    bounds when ``match_cost`` is given. ``anchor`` is the neural content start
    (ayah-1 onset), the floor reference for the first matched boundary.

    Each boundary is matched, left *wasl* (cost ``wasl_pen``), or skips an onset
    as an extra pause (cost ``extra_pen``). Skips are bounded to ``max_skip``
    consecutive. Returns, per boundary, the matched onset (ms) or ``None``.
    """
    m, p = len(exp), len(ons)
    if m == 0:
        return []
    if p == 0:
        return [None] * m
    inf = float("inf")
    dp = [[inf] * p for _ in range(m)]
    bk: list[list[tuple[int, int] | None]] = [[None] * p for _ in range(m)]
    def plausible(ogap: float, egap: float, span: int) -> bool:
        """Is a matched span of ``span`` ayahs covering ``ogap`` ms plausible?

        Floor: at least ``abs_floor`` per ayah (an ayah can't be recited in a
        blink), and at least ``dur_floor`` × the char estimate. Melodic stretch
        only lengthens ayahs, so a generous ceiling (max of relative / absolute)
        just rejects far jumps past the real pause.
        """
        lo = max(abs_floor * span, dur_floor * egap)
        hi = max(abs_ceil * span, dur_ceil * egap)
        return lo <= ogap <= hi

    for i in range(m):
        for j in range(p):
            best, bb = inf, None
            # Match-quality term: acoustic if given, else positional distance.
            mq = match_cost(i, j) if match_cost is not None else abs(ons[j] - exp[i])
            # Base: boundary i is the FIRST matched one (i wasls + j extras
            # before it); the span back to the neural anchor covers i+1 ayahs.
            if plausible(ons[j] - anchor, exp[i] - anchor, i + 1):
                best = wasl_pen * i + extra_pen * j + mq
            for ip in range(max(0, i - 1 - max_skip), i):
                base_ip = dp[ip]
                egap = exp[i] - exp[ip]
                for jp in range(max(0, j - 1 - max_skip), j):
                    if base_ip[jp] == inf:
                        continue
                    if not plausible(ons[j] - ons[jp], egap, i - ip):
                        continue  # implausible ayah span → forbid this match
                    step_mq = mq if match_cost is not None else abs(ons[j] - (ons[jp] + egap))
                    cost = (
                        base_ip[jp]
                        + wasl_pen * (i - ip - 1)
                        + extra_pen * (j - jp - 1)
                        + step_mq
                    )
                    if cost < best:
                        best, bb = cost, (ip, jp)
            dp[i][j] = best
            bk[i][j] = bb
    # Close the path: pick the last match, charging trailing wasl boundaries.
    best, end = inf, None
    for i in range(m):
        for j in range(p):
            if dp[i][j] == inf:
                continue
            c = dp[i][j] + wasl_pen * (m - 1 - i)
            if c < best:
                best, end = c, (i, j)
    out: list[float | None] = [None] * m
    cur = end
    while cur is not None:
        i, j = cur
        out[i] = ons[j]
        cur = bk[i][j]
    return out


def _interp_wasl(k: int, exp: list[float], matched: list[float | None]) -> float:
    """Position a wasl boundary by interpolating between bracketing matches.

    Uses the prior's relative spacing between the nearest matched onsets on each
    side (anchored both ways → no drift). Falls back to prior-offset
    extrapolation at the ends, or the raw prior if nothing is matched.
    """
    lo = next((i for i in range(k - 1, -1, -1) if matched[i] is not None), None)
    hi = next((i for i in range(k + 1, len(matched)) if matched[i] is not None), None)
    if lo is not None and hi is not None:
        o_lo, o_hi = matched[lo], matched[hi]
        span = exp[hi] - exp[lo]
        frac = (exp[k] - exp[lo]) / span if span > 0 else 0.5
        return o_lo + (o_hi - o_lo) * frac
    if lo is not None:
        return matched[lo] + (exp[k] - exp[lo])
    if hi is not None:
        return matched[hi] - (exp[hi] - exp[k])
    return exp[k]


def align_surah_fast(
    pcm,
    sr: int,
    words: list,
    *,
    surah: int | None = None,
    bundle: Any = None,
    min_win_s: float = 1.0,
    grow_s: float = 1.0,
    max_win_s: float = 6.0,
    conf_thresh: float = 0.25,
    conf_reliable: float = 0.45,
    edge_guard_ms: float = 280.0,
    trace: dict | None = None,
) -> list[tuple[int, int, float]]:
    """Two-phase windowed forced alignment; see module docstring.

    Args:
        pcm, sr, words: as :func:`align.align_surah`.
        surah: sura number (inferred from ``words`` if omitted).
        bundle: pre-loaded emitter bundle to reuse (loaded lazily otherwise).
        min_win_s: starting window width, seconds (the user's "read 1 second").
        grow_s: how much to widen the window on each retry, seconds.
        max_win_s: largest window before giving up and using the rough guess.
        conf_thresh: minimum first-word confidence to trust a window's onset.
        edge_guard_ms: an onset closer than this to a window edge is distrusted
            (the boundary may lie outside) → widen.
        trace: optional dict; if given, filled with per-ayah diagnostics and
            totals (used by the UI's explicit-phase view and by benchmarks).

    Returns:
        ``[(ayah, start_ms, conf)]`` for every ayah, ``start_ms`` monotonic.
    """
    import numpy as np

    from surah_cutter.core import align as A
    from surah_cutter.core import candidates
    from surah_cutter.core._ctc import forced_align, merge_repeats
    from surah_cutter.data import transliteration

    if surah is None:
        surah = words[0].surah
    if bundle is None:
        bundle = _default_bundle()

    word_trs = transliteration.surah_word_tr(surah)
    n_words = len(words)
    total_ms = len(pcm) * 1000.0 / sr

    # ayah -> ordered global word indices; chars per ayah (tempo unit).
    ayah_words: "OrderedDict[int, list[int]]" = OrderedDict()
    for i, w in enumerate(words):
        ayah_words.setdefault(w.ayah, []).append(i)
    ayah_list = list(ayah_words.keys())
    chars = {
        a: max(1, sum(len(word_trs[i]) for i in ayah_words[a] if i < len(word_trs)))
        for a in ayah_list
    }
    total_chars = max(1, sum(chars.values()))
    rate = total_ms / total_chars  # ms per transliteration char (window sizing)

    def wlen(i: int) -> float:
        return len(word_trs[i]) * rate if i < len(word_trs) else rate

    # wav2vec2's conv feature-extractor needs a minimum input length.
    min_win_ms = 800.0

    # --- low-level: align words[i0:i1] within audio [t0_ms, t1_ms]. ----------
    def align_slice(i0: int, i1: int, t0_ms: float, t1_ms: float, basmala: bool):
        if t1_ms - t0_ms < min_win_ms:
            mid = 0.5 * (t0_ms + t1_ms)
            t0_ms, t1_ms = mid - min_win_ms / 2, mid + min_win_ms / 2
        t0_ms = max(0.0, t0_ms)
        t1_ms = min(total_ms, t1_ms)
        if t1_ms - t0_ms < min_win_ms:  # near an edge: grow inward
            if t0_ms <= 0.0:
                t1_ms = min(total_ms, min_win_ms)
            else:
                t0_ms = max(0.0, total_ms - min_win_ms)
        s0 = max(0, int(round(t0_ms * sr / 1000.0)))
        s1 = min(len(pcm), int(round(t1_ms * sr / 1000.0)))
        if s1 - s0 < int(0.6 * sr) or i1 <= i0:  # < 0.6 s or no words
            return None
        sub_words = words[i0:i1]
        sub_trs = word_trs[i0:i1]
        basmala_trs = A._basmala_trs(transliteration) if basmala else None
        targets, spans = A._build_targets(sub_words, sub_trs, bundle, basmala_trs)
        if not targets:
            return None
        slice_pcm = np.ascontiguousarray(pcm[s0:s1], dtype=np.float32)
        emis = A._emissions(bundle, slice_pcm, np)
        nf = emis.shape[0]
        stride = A._stride_ms(s1 - s0, sr, nf)
        path, fscores = forced_align(
            np.ascontiguousarray(emis[None, :, :], dtype=np.float32),
            np.asarray(targets, dtype=np.int64)[None, :],
            blank=bundle.blank_id,
        )
        path = np.asarray(path).reshape(-1)
        fscores = np.asarray(fscores).reshape(-1)
        segs = [
            s for s in merge_repeats(path.tolist(), {i: i for i in range(emis.shape[1])})
            if s.label != bundle.blank_id
        ]
        tspans = [(s.start, s.end) for s in segs]
        while len(tspans) < len(targets):
            tspans.append(tspans[-1] if tspans else (nf - 1, nf - 1))
        base = s0 * 1000.0 / sr
        # CTC emission-lag compensation, same as the full path.
        lag = A.ALIGN_LAG_MS
        starts, confs = [], []
        for (ts, te) in spans:
            sf = tspans[ts][0] if ts < len(tspans) else nf - 1
            ef = tspans[te - 1][1] if te > ts and te - 1 < len(tspans) else sf
            starts.append(max(0.0, base + sf * stride - lag))
            seg = fscores[sf:max(sf + 1, ef + 1)]
            confs.append(float(np.exp(seg.mean())) if seg.size else 0.0)
        return {"starts": starts, "confs": confs,
                "base": base, "end": s1 * 1000.0 / sr,
                "win_ms": s1 * 1000.0 / sr - base}

    # --- low-level: greedy CTC decode of [t0_ms, t1_ms] -> transliteration. ---
    # Unlike forced alignment (which is *told* the words), greedy decode reports
    # what the model actually *hears* — the signal that says which ayah opens
    # right after a pause, immune to the char prior's drift.
    inv_vocab = {v: k for k, v in bundle.vocab.items()}

    def greedy_decode(t0_ms: float, t1_ms: float) -> str:
        s0 = max(0, int(round(t0_ms * sr / 1000.0)))
        s1 = min(len(pcm), int(round(t1_ms * sr / 1000.0)))
        if s1 - s0 < int(0.3 * sr):
            return ""
        emis = A._emissions(bundle, np.ascontiguousarray(pcm[s0:s1], dtype=np.float32), np)
        ids = emis.argmax(-1)
        out: list[str] = []
        prev = -1
        for i in ids.tolist():
            if i != prev and i != bundle.blank_id:
                out.append(inv_vocab.get(int(i), ""))
            prev = i
        return "".join(out)

    starts_ms: dict[int, float] = {}
    confs: dict[int, float] = {}
    tr_ayahs: list[dict] = []
    model_reads: list[dict] = []
    audio_ms = 0.0

    # ===================================================================== #
    # Phase 1a — detect pauses (cheap, no neural).                           #
    # ===================================================================== #
    pauses = candidates.detect_combined_pauses(pcm, sr, surah=surah)
    pause_onsets = sorted(p.center_ms for p in pauses)

    # ===================================================================== #
    # Phase 1b — anchor ayah 1 (absorbs the intro-basmala for non-Fatiha).   #
    # ===================================================================== #
    # This neural call also tells us where the CONTENT starts (after the ~6.5 s
    # basmala) — essential for the prior below, which otherwise places every
    # boundary ~6 s too early (it knows word fractions, not the basmala offset).
    basmala = surah not in (1, 9)
    fa1 = ayah_words[ayah_list[0]][0]
    intro = 6500.0 if basmala else 150.0
    head_end = min(n_words, fa1 + 4)
    head_dur = sum(wlen(j) for j in range(fa1, head_end))
    res = align_slice(0, head_end, 0.0, min(intro + head_dur + 900.0, total_ms), basmala)
    if res is None:  # degenerate: fall back to full accurate alignment
        return A.align_surah(pcm, sr, words, surah=surah, model=bundle)
    audio_ms += res["win_ms"]
    a1 = ayah_list[0]
    content_start = res["starts"][fa1]
    model_reads.append({
        "phase": "anchor",
        "ayah": a1,
        "window_ms": [round(res["base"], 1), round(res["end"], 1)],
        "words": " ".join(w.text for w in words[0:head_end]),
        "target": words[fa1].text,
        "onset_ms": round(content_start, 1),
        "conf": round(res["confs"][fa1], 3),
    })
    starts_ms[a1] = content_start
    confs[a1] = res["confs"][fa1]
    tr_ayahs.append({"ayah": a1, "rough_ms": 0.0, "from_pause": False,
                     "onset_ms": content_start, "conf": confs[a1],
                     "win_s": round(res["win_ms"] / 1000.0, 2), "reliable": True})

    # ===================================================================== #
    # Phase 1c — match pause onsets to boundaries (order-preserving DP).      #
    # ===================================================================== #
    # The DP assigns pauses to ayah boundaries by ACOUSTIC fit: for each pause we
    # greedy-decode the ~1.3 s after it and compare what was heard to each ayah's
    # opening tokens. This is bias-free — unlike the char prior, which drifts by
    # seconds under melodic stretch and made the DP shift whole runs onto the
    # wrong pause (the off-by-one bug). The prior is kept only for the duration
    # floor/ceiling. The DP still skips extra onsets (breaths) and marks
    # pause-less boundaries as wasl; a wrong match never cascades.
    fractions = candidates.expected_boundary_fractions(surah)
    content_dur = max(1.0, total_ms - content_start)
    inner = ayah_list[1:]  # boundaries we must place (ayah 1 is anchored)
    exp = [content_start + fractions.get(a, 0.0) * content_dur for a in inner]
    # Only pauses inside the content region can be ayah boundaries.
    pause_onsets = [p for p in pause_onsets if p > content_start - 200.0]
    # Opening tokens (first two words) per inner ayah, and a greedy decode per
    # pause — the two sides of the acoustic match.
    openings = [
        "".join(word_trs[i] for i in ayah_words[a][:4] if i < len(word_trs))
        for a in inner
    ]
    decode_win_ms = 2500.0  # long enough to reach a distinguishing word in slow melisma
    pause_decodes = [greedy_decode(p - 50.0, p + decode_win_ms) for p in pause_onsets]
    audio_ms += sum(min(total_ms, p + decode_win_ms) - max(0.0, p - 50.0) for p in pause_onsets)
    sim_cache: dict[tuple[int, int], float] = {}

    def acoustic_cost(i: int, j: int) -> float:
        key = (i, j)
        s = sim_cache.get(key)
        if s is None:
            s = _sim(pause_decodes[j], openings[i])
            sim_cache[key] = s
        return 3000.0 * (1.0 - s)  # lower cost = better-heard match

    # With a reliable acoustic match cost the char prior is no longer trusted for
    # position — so the plausibility *ceiling* (which used the prior's gap) must
    # be loose, or it would reject the genuinely long melodic ayahs (sura 98 runs
    # ~18 s/ayah) and force the DP onto an earlier within-ayah pause. The floor
    # still rejects grabbing a breath pause too soon; the acoustic cost does the
    # rest.
    matched = _match_boundaries(
        exp, pause_onsets, anchor=content_start, wasl_pen=2600.0,
        extra_pen=350.0, match_cost=acoustic_cost,
        dur_ceil=12.0, abs_ceil=22000.0,
    )
    # ayah -> (rough_ms, from_pause). Wasl boundaries: interpolate between the
    # matched onsets that bracket them (anchored both sides — no drift).
    place: dict[int, tuple[float, bool]] = {}
    for k, a in enumerate(inner):
        if matched[k] is not None:
            place[a] = (matched[k], True)
        else:
            place[a] = (_interp_wasl(k, exp, matched), False)

    # ===================================================================== #
    # Phase 2 — neural refinement with growing windows (from min_win_s).     #
    # ===================================================================== #
    n_steps = max(1, int(round((max_win_s - min_win_s) / max(grow_s, 0.1))) + 1)

    def refine(
        c: float,
        fa: int,
        prev_start: float,
        start_step: int = 0,
        *,
        ayah: int | None = None,
        phase: str = "probe",
    ):
        """Grow a window around ``c`` until the onset of word ``fa`` locks on.

        Returns ``(onset_ms, conf, win_s)`` — best onset seen; conf < thresh
        means it never locked on confidently. ``start_step`` skips the smallest
        window(s) for boundaries whose initial guess is known to be looser.
        """
        nonlocal audio_ms
        onset, conf, win_used = None, 0.0, 0.0
        for w in range(start_step, n_steps):
            win_ms = (min_win_s + w * grow_s) * 1000.0
            t0 = max(prev_start + 30.0, c - win_ms * 0.45)
            t1 = min(total_ms, c + win_ms * 0.55)
            # words filling [t0, c] (prev-ayah tail) and [c, t1] (this-ayah head),
            # sized to the audio so CTC never stretches a few words over a gap.
            i0 = fa
            back = 0.0
            while i0 > 0 and back < (c - t0):
                i0 -= 1
                back += wlen(i0)
            if fa > 0:
                i0 = min(i0, fa - 1)  # at least one tail word for the transition
            i1 = fa
            fwd = 0.0
            while i1 < n_words and fwd < (t1 - c):
                fwd += wlen(i1)
                i1 += 1
            i1 = max(i1, fa + 1)
            res = align_slice(i0, i1, t0, t1, basmala=False)
            if res is None:
                break
            audio_ms += res["win_ms"]
            win_used = res["win_ms"] / 1000.0
            cand = res["starts"][fa - i0]
            cconf = res["confs"][fa - i0]
            onset, conf = cand, cconf  # keep best-so-far
            guard = min(edge_guard_ms, 0.2 * res["win_ms"])
            near_edge = (cand - res["base"] < guard) or (res["end"] - cand < guard)
            model_reads.append({
                "phase": phase,
                "ayah": ayah,
                "candidate_ms": round(c, 1),
                "window_ms": [round(res["base"], 1), round(res["end"], 1)],
                "words": " ".join(w.text for w in words[i0:i1]),
                "target": words[fa].text if fa < len(words) else "",
                "onset_ms": round(cand, 1),
                "delta_ms": round(cand - c, 1),
                "conf": round(cconf, 3),
                "near_edge": near_edge,
            })
            if cconf >= conf_thresh and not near_edge:
                break  # boundary comfortably inside, confident → trust it
        return onset, conf, win_used

    for ai in range(1, len(ayah_list)):
        a = ayah_list[ai]
        prev_a = ayah_list[ai - 1]
        prev_start = starts_ms[prev_a]
        fa = ayah_words[a][0]

        # Position fixed by the global match (no cascade). Pause-backed onsets are
        # already ~200 ms accurate; a tiny window nails them. Wasl positions are
        # interpolated (looser) → start the window wider.
        c0, from_pause = place[a]
        c = min(max(c0, prev_start + 30.0), total_ms)
        start_step = 0 if from_pause else 1  # wasl: skip the 1 s window

        onset, conf, win_used = refine(c, fa, prev_start, start_step, ayah=a)
        reliable = onset is not None and conf >= conf_thresh
        if not reliable:  # never locked on → trust the matched/interpolated guess
            onset = c if onset is None else onset

        starts_ms[a] = onset
        confs[a] = conf
        tr_ayahs.append({"ayah": a, "rough_ms": c0, "from_pause": from_pause,
                         "onset_ms": onset, "conf": round(conf, 3),
                         "win_s": round(win_used, 2), "reliable": reliable})

    # --- Phase 2b — repair shifted runs using reliable neighbours. ----------
    # Char-prior compression can make the global match shift a run of boundaries
    # onto earlier pauses (detectable as low neural confidence). Re-anchor such
    # ayahs by interpolating between the nearest CONFIDENT onsets (real anchors
    # average out melodic drift), then re-run the neural window around that guess.
    cum: dict[int, int] = {}
    acc = 0
    for a in ayah_list:
        cum[a] = acc
        acc += chars[a]
    rel = {a: confs.get(a, 0.0) >= conf_reliable for a in ayah_list}
    rel[ayah_list[0]] = True
    for idx in range(1, len(ayah_list)):
        a = ayah_list[idx]
        if rel[a]:
            continue
        lo = next((ayah_list[k] for k in range(idx - 1, -1, -1) if rel[ayah_list[k]]), None)
        hi = next(
            (ayah_list[k] for k in range(idx + 1, len(ayah_list)) if rel[ayah_list[k]]),
            None,
        )
        if lo is None:
            continue
        if hi is not None and cum[hi] > cum[lo]:
            frac = (cum[a] - cum[lo]) / (cum[hi] - cum[lo])
            pred = starts_ms[lo] + (starts_ms[hi] - starts_ms[lo]) * frac
        else:
            pred = starts_ms[lo] + (cum[a] - cum[lo]) * rate
        if abs(starts_ms.get(a, pred) - pred) < 1200.0:
            continue  # already consistent with the reliable anchors → leave it
        fa = ayah_words[a][0]
        o2, c2, _w = refine(
            min(max(pred, starts_ms[lo] + 30.0), total_ms),
            fa,
            starts_ms[lo],
            1,
            ayah=a,
            phase="repair",
        )
        if o2 is not None and c2 >= confs.get(a, 0.0):
            starts_ms[a] = o2
            confs[a] = c2
            for t in tr_ayahs:
                if t["ayah"] == a:
                    t.update(onset_ms=o2, conf=round(c2, 3), repaired=True)
                    break

    # --- shape: monotonic non-decreasing starts, per-ayah rows. ------------
    rows: list[tuple[int, int, float]] = []
    last = 0
    for a in ayah_list:
        ms = int(round(starts_ms.get(a, last)))
        ms = max(ms, last)
        rows.append((a, ms, float(confs.get(a, 0.0))))
        last = ms

    if trace is not None:
        trace["phase1_pauses"] = [
            {
                "start_ms": round(p.start_ms, 1),
                "end_ms": round(p.end_ms, 1),
                "center_ms": round(p.center_ms, 1),
                "depth_ms": round(p.depth_ms, 1),
            }
            for p in pauses
        ]
        trace["model_reads"] = model_reads
        trace["ayahs"] = tr_ayahs
        trace["audio_ms"] = audio_ms
        trace["total_ms"] = total_ms
        trace["n_pause"] = sum(1 for t in tr_ayahs if t["from_pause"])
        trace["n_wasl"] = sum(
            1 for t in tr_ayahs if not t["from_pause"] and t["ayah"] != ayah_list[0]
        )
        trace["coverage"] = round(audio_ms / total_ms, 3) if total_ms else 0.0
    return rows
