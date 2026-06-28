"""Fast two-phase forced alignment — pause onsets + growing neural windows.

The accurate path (:func:`surah_cutter.core.align.align_surah`) runs the acoustic
model over the **whole** sura. This module saves inference time by running the
model only over **short windows**, sized up on demand. The boundary is still
placed by the neural aligner (not by a pause), so accuracy is preserved.

Two explicit phases (the UI exposes both so the user can steer them):

**Phase 1 — sequential pause confirmation.**
An energy/DSP detector gives all plausible silence segments. For each pause we
read only a short slice after speech resumes, greedy-decode what the model hears,
and compare it to the known opening of the current ayah. A pause is accepted only
when it confirms the next expected ayah. If it sounds like a later ayah, the
matcher first searches the skipped span for weaker/missed pauses; unresolved
gaps become *wasl* windows for local alignment instead of being forced onto a
weak pause.

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
import os
from collections import OrderedDict
from typing import Any, Callable

#: Lookahead "don't steal the next ayah's pause" gate (Phase 1c). If a near-next
#: ayah's opening is heard distinctly better after a pause (sim margin), penalise
#: matching that pause to the earlier ayah so the DP leaves it wasl instead of an
#: off-by-one grab. Env-tunable for benchmarking; ``_GATE_PEN=0`` disables it.
_GATE_PEN = float(os.environ.get("SC_FAST_GATE_PEN", "2000"))
_GATE_MARGIN = float(os.environ.get("SC_FAST_GATE_MARGIN", "0.25"))
#: Only gate a pause away from ayah i when i's OWN match is weak (s < this) — a
#: strong legitimate match is never penalised just because a neighbour is noisier.
#: (margin 0.25 + smax 0.50 fix Yasin's off-by-one without breaking sura 85.)
_GATE_SMAX = float(os.environ.get("SC_FAST_GATE_SMAX", "0.50"))
#: Post-DP wasl conversion: a matched pause whose opening is heard this poorly
#: (sim below) probably isn't this ayah's boundary at all — drop the match to wasl
#: so the continuous fill (≈ local full) places it, instead of trusting a breath
#: pause. Kept env-tunable because some reciters decode noisier than others.
_WASL_SIM = float(os.environ.get("SC_FAST_WASL_SIM", "0.36"))
#: Harder gate for accepting a pause as an ayah anchor. Values below this are
#: still logged, but are treated as too weak to be a real boundary match.
_MIN_MATCH_SIM = float(os.environ.get("SC_FAST_MIN_MATCH_SIM", "0.40"))
#: Recovery pass is only used after a later pause already proved we skipped over
#: an ayah. In that narrow context we can accept a slightly weaker similarity for
#: an intermediate pause instead of forcing the ayah into a long wasl window.
_RECOVER_MATCH_SIM = float(os.environ.get("SC_FAST_RECOVER_MATCH_SIM", "0.34"))
#: If the short decode after a pause already identifies the ayah this well, the
#: pause edge is trusted as the timing and Phase 2 does not run another
#: forced-align window for that boundary.
_PAUSE_TRUST_SIM = float(os.environ.get("SC_FAST_PAUSE_TRUST_SIM", "0.50"))
#: A weak match far from the structural/tempo prior is almost certainly a random
#: similar syllable elsewhere in the sura. Strong matches can still override it.
_MATCH_PRIOR_WINDOW_MS = float(os.environ.get("SC_FAST_MATCH_PRIOR_WINDOW_MS", "15000"))
_FAR_MATCH_SIM = float(os.environ.get("SC_FAST_FAR_MATCH_SIM", "0.72"))
#: Raw pause lists contain many within-ayah breaths. Let DP skip long runs of
#: them, otherwise it is forced onto early pauses just to keep the chain alive.
_MATCH_MAX_SKIP = int(os.environ.get("SC_FAST_MATCH_MAX_SKIP", "48"))
#: Sequential matcher only looks a few ayahs ahead when classifying what is heard
#: after a pause. Larger values read more text comparisons and can overfit repeated
#: openings; 3 catches the practical "we jumped over one/two ayahs" cases.
_MATCH_LOOKAHEAD = int(os.environ.get("SC_FAST_MATCH_LOOKAHEAD", "3"))
#: Future-ayah classification needs this similarity margin over the current ayah;
#: otherwise we keep the algorithm sequential and let the current ayah claim it.
_FUTURE_MATCH_MARGIN = float(os.environ.get("SC_FAST_FUTURE_MATCH_MARGIN", "0.12"))
#: Only openings that share this many initial collapsed transliteration
#: characters with a nearby ayah get a longer greedy decode; unique ayahs stay on
#: the short default read.
_AMBIG_PREFIX_CHARS = int(os.environ.get("SC_FAST_AMBIG_PREFIX_CHARS", "6"))
#: Greedy pause verification should start near speech resumption, not at the
#: centre of a long silence segment. This is only for reading; the emitted
#: pause-backed ayah start remains the pause centre.
_PAUSE_DECODE_PRE_ROLL_MS = float(os.environ.get("SC_FAST_PAUSE_DECODE_PRE_ROLL_MS", "50"))
_PAUSE_DECODE_WIN_MS = float(os.environ.get("SC_FAST_PAUSE_DECODE_WIN_MS", "2500"))
_PAUSE_DECODE_AMBIG_WIN_MS = float(os.environ.get("SC_FAST_PAUSE_DECODE_AMBIG_WIN_MS", "4500"))
_AMBIG_DECODE_PRIOR_MS = float(os.environ.get("SC_FAST_AMBIG_DECODE_PRIOR_MS", "7000"))
# A weak intra-ayah breath may fuzzy-match the next ayah because the short decode
# contains a tail from the previous ayah plus a little future text. Require the
# decode to actually begin like the target opening before accepting such pauses.
_PREFIX_WEAK_MIN = int(os.environ.get("SC_FAST_PREFIX_WEAK_MIN", "2"))
_PREFIX_STRONG_MIN = int(os.environ.get("SC_FAST_PREFIX_STRONG_MIN", "0"))
_PREFIX_BYPASS_SIM = float(os.environ.get("SC_FAST_PREFIX_BYPASS_SIM", "0.62"))
#: Small left context around speech resumption, cached with the pause decode so a
#: later 1s refine window can reuse the same emissions instead of rereading.
_PAUSE_CACHE_PRE_ROLL_MS = float(os.environ.get("SC_FAST_PAUSE_CACHE_PRE_ROLL_MS", "650"))
#: A wasl run longer than this is no longer a "short local fill"; aligning it
#: whole turns fast into full-ish ONNX inference. Keep rough points and let
#: review/repair flag them instead.
_MAX_WASL_RUN_ALIGN_MS = float(os.environ.get("SC_FAST_MAX_WASL_RUN_ALIGN_MS", "20000"))
# If local wasl alignment lands next to a strong detected pause, later repair may
# improve confidence but must not pull that boundary away from the visible pause.
_WASL_PAUSE_HINT_MS = float(os.environ.get("SC_FAST_WASL_PAUSE_HINT_MS", "900"))
_REPAIR_PAUSE_PULL_MS = float(os.environ.get("SC_FAST_REPAIR_PAUSE_PULL_MS", "250"))
#: Self-diagnostic threshold. A drifted sura (off-by-one cascade with no recovery
#: anchor) shows up as low MEAN boundary confidence. Fast mode no longer falls
#: back to full automatically; this threshold marks the result as suspicious so
#: the review UI can highlight weak auto rows. 0 disables.
_FALLBACK_CONF = float(os.environ.get("SC_FAST_FALLBACK_CONF", "0.61"))
#: Independent placement verification threshold. Unlike refine confidence (high
#: even when forced-aligned to the wrong spot), a greedy decode of what is actually
#: heard at each placed boundary is low across the board when the sura has drifted.
#: Below this mean verify-sim the result is marked suspicious. 0 disables.
_VERIFY_SIM = float(os.environ.get("SC_FAST_VERIFY_SIM", "0.57"))
_VERIFY_WINDOW_MS = float(os.environ.get("SC_FAST_VERIFY_WINDOW_MS", "1600"))
_DRIFT_MIN_PAUSE_COVERAGE = float(os.environ.get("SC_FAST_DRIFT_MIN_PAUSE_COVERAGE", "0.80"))

_WAQF_STOP_SIGNS = frozenset("ۖۗۘۙۚۛۜ")


def _collapse(s: str) -> str:
    """Collapse runs of the same character (melodic vowel elongation -> 1)."""
    out: list[str] = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the common collapsed prefix of two transliteration strings."""
    common = 0
    for ca, cb in zip(_collapse(a), _collapse(b)):
        if ca != cb:
            break
        common += 1
    return common


#: Latin long-vowel (madd-carrier) chars. As a *tempo unit* a long vowel is
#: recited longer, so it weighs double. The tempo probe (tools/bench_fast) showed
#: this proxy predicts where ayah boundaries land in audio better than raw char
#: count and than the mushaf render-width prior used before.
_LONG_VOWELS = frozenset("āīū")


def _unit_len(s: str) -> float:
    """Tempo-unit length of a transliteration token (long vowels count double)."""
    return float(sum(2 if c in _LONG_VOWELS else 1 for c in s))


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


def _count_waqf_marks(words: list) -> int:
    """Count explicit stop signs available in the bundled mushaf text."""
    return sum(
        1
        for w in words
        for ch in str(getattr(w, "text", ""))
        if ch in _WAQF_STOP_SIGNS
    )


def _ambiguous_openings(
    openings: list[str],
    *,
    lookahead: int = _MATCH_LOOKAHEAD,
    prefix_chars: int = _AMBIG_PREFIX_CHARS,
) -> set[int]:
    """Indices whose opening text shares an actual prefix with a nearby ayah."""
    ambiguous: set[int] = set()
    for i, opening in enumerate(openings):
        hi = min(len(openings), i + max(1, lookahead) + 1)
        ci = _collapse(opening)
        for k in range(i + 1, hi):
            ck = _collapse(openings[k])
            common = 0
            for a, b in zip(ci, ck):
                if a != b:
                    break
                common += 1
            if common >= prefix_chars:
                ambiguous.add(i)
                ambiguous.add(k)
    return ambiguous


def _sequential_pause_match(
    exp: list[float],
    ons: list[float],
    sim: Callable[[int, int], float],
    *,
    min_sim: float,
    prior_window_ms: float,
    recover_min_sim: float | None = None,
    lookahead: int = _MATCH_LOOKAHEAD,
    future_margin: float = _FUTURE_MATCH_MARGIN,
    max_candidates: int = _MATCH_MAX_SKIP,
    reject: Callable[[int, int, float], str | None] | None = None,
) -> tuple[list[float | None], list[dict]]:
    """Sequentially confirm ayah boundaries from pause decodes.

    ``exp`` and ``ons`` are boundary/pause times in milliseconds. ``sim(i, j)``
    says how well the short decode after pause ``j`` matches boundary ``i``'s
    ayah opening. The matcher walks forward: a pause can confirm the current
    boundary, be skipped as an internal/weak pause, or reveal that we jumped to a
    future ayah. In the latter case, skipped boundaries are recovered from the
    pauses before the future anchor when possible.
    """
    n = len(exp)
    if n == 0:
        return [], []
    if not ons:
        return [None] * n, [
            {"boundary": i, "action": "wasl", "reason": "no_pauses"}
            for i in range(n)
        ]

    matched: list[float | None] = [None] * n
    events: list[dict] = []
    j_cursor = 0
    k = 0
    recover_pool: list[int] = []
    if recover_min_sim is None:
        recover_min_sim = min_sim

    def reject_reason(i: int, j: int, s: float) -> str | None:
        if s < min_sim:
            return "low_sim"
        return reject(i, j, s) if reject is not None else None

    def score(i: int, j: int) -> float:
        dist_pen = min(1.0, abs(ons[j] - exp[i]) / max(1.0, prior_window_ms))
        return sim(i, j) - 0.18 * dist_pen

    def label_pause(i0: int, j: int) -> tuple[int, float]:
        current_s = sim(i0, j)
        best_i, best_s, best_score = i0, current_s, score(i0, j)
        hi = min(n - 1, i0 + max(0, lookahead))
        for i in range(i0 + 1, hi + 1):
            s = sim(i, j)
            reason = reject_reason(i, j, s)
            if reason is not None:
                continue
            sc = score(i, j)
            if sc > best_score:
                best_i, best_s, best_score = i, s, sc
        if best_i > i0 and best_s >= current_s + future_margin:
            return best_i, best_s
        return i0, current_s

    def accept(i: int, j: int, *, action: str) -> None:
        s = sim(i, j)
        matched[i] = ons[j]
        events.append({
            "boundary": i,
            "pause": j,
            "action": action,
            "sim": round(s, 3),
            "expected_ms": round(exp[i], 1),
            "pause_ms": round(ons[j], 1),
            "expected_delta_ms": round(ons[j] - exp[i], 1),
        })

    def recover_missing(i0: int, i1: int, js: list[int]) -> None:
        last_recovered = -1
        recover_js = sorted(dict.fromkeys(js))
        for i in range(i0, i1):
            best_j, best_s, best_score = -1, 0.0, -1e9
            for j in recover_js:
                if j <= last_recovered or matched.count(ons[j]) > 0:
                    continue
                s = sim(i, j)
                if s < recover_min_sim:
                    continue
                hard_reason = reject(i, j, s) if reject is not None else None
                if hard_reason is not None and hard_reason != "future_better":
                    continue
                sc = score(i, j)
                if sc > best_score:
                    best_j, best_s, best_score = j, s, sc
            if best_j >= 0:
                accept(i, best_j, action="recovered")
                last_recovered = best_j
            else:
                events.append({
                    "boundary": i,
                    "action": "wasl",
                    "reason": "missing_before_future",
                    "best_sim": round(best_s, 3),
                    "expected_ms": round(exp[i], 1),
                })

    while k < n:
        upper_i = min(n - 1, k + max(0, lookahead))
        lower = exp[k] - prior_window_ms
        upper = exp[upper_i] + prior_window_ms
        while j_cursor < len(ons) and ons[j_cursor] < lower:
            events.append({
                "boundary": k,
                "pause": j_cursor,
                "action": "early_pool",
                "reason": "before_expected_window",
                "pause_ms": round(ons[j_cursor], 1),
            })
            recover_pool.append(j_cursor)
            j_cursor += 1

        candidates: list[int] = []
        j = j_cursor
        while j < len(ons) and ons[j] <= upper and len(candidates) < max_candidates:
            candidates.append(j)
            j += 1
        if not candidates:
            events.append({
                "boundary": k,
                "action": "wasl",
                "reason": "no_pause_in_window",
                "expected_ms": round(exp[k], 1),
            })
            k += 1
            continue

        pending: list[int] = []
        advanced = False
        for j in candidates:
            heard_i, heard_s = label_pause(k, j)
            reason = reject_reason(heard_i, j, heard_s)
            if heard_i == k and reason is None:
                for extra_j in recover_pool + pending:
                    events.append({
                        "boundary": k,
                        "pause": extra_j,
                        "action": "extra",
                        "reason": "low_or_internal_before_confirmed",
                        "sim": round(sim(k, extra_j), 3),
                        "pause_ms": round(ons[extra_j], 1),
                    })
                accept(k, j, action="confirmed")
                j_cursor = j + 1
                recover_pool = []
                k += 1
                advanced = True
                break
            if heard_i > k and reason is None:
                events.append({
                    "boundary": k,
                    "pause": j,
                    "action": "future_gap",
                    "heard_boundary": heard_i,
                    "heard_sim": round(heard_s, 3),
                    "current_sim": round(sim(k, j), 3),
                    "pause_ms": round(ons[j], 1),
                })
                recover_missing(k, heard_i, recover_pool + pending)
                accept(heard_i, j, action="future_anchor")
                j_cursor = j + 1
                recover_pool = []
                k = heard_i + 1
                advanced = True
                break
            pending.append(j)

        if not advanced:
            best_s = max((sim(k, j) for j in candidates), default=0.0)
            for extra_j in pending:
                events.append({
                    "boundary": k,
                    "pause": extra_j,
                    "action": "extra",
                    "reason": "no_confirmed_match",
                    "sim": round(sim(k, extra_j), 3),
                    "pause_ms": round(ons[extra_j], 1),
                })
                recover_pool.append(extra_j)
            events.append({
                "boundary": k,
                "action": "wasl",
                "reason": "no_confirmed_match",
                "best_sim": round(best_s, 3),
                "expected_ms": round(exp[k], 1),
            })
            j_cursor = max(j_cursor, candidates[-1] + 1)
            k += 1

    return matched, events


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
    log_event: Callable[[str], None] | None = None,
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
        log_event: optional callback that receives human-readable realtime
            diagnostics while the sura is being processed.

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

    def emit(message: str) -> None:
        if log_event is not None:
            log_event(message)

    def sec(ms: float) -> str:
        return f"{ms / 1000.0:.2f}"

    def short(text: str, limit: int = 80) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def log_read(read: dict) -> None:
        w0, w1 = read.get("window_ms", [None, None])
        if w0 is None or w1 is None:
            return
        dur = max(0.0, float(w1) - float(w0)) / 1000.0
        phase = read.get("phase")
        ayah = read.get("ayah")
        target = short(str(read.get("target") or ""))
        if phase == "wasl-run":
            detect = f"детект wasl-серии с аята {ayah}"
        else:
            detect = f"детект аята {ayah}"
        extra = ""
        if read.get("conf") is not None:
            extra += f", conf={read.get('conf')}"
        if read.get("delta_ms") is not None:
            extra += f", delta={read.get('delta_ms'):+.0f}мс"
        if read.get("candidate_ms") is not None:
            c = float(read["candidate_ms"])
            left = c - float(w0)
            right = float(w1) - c
            extra += f", окно от кандидата -{left:.0f}/+{right:.0f}мс"
        if read.get("near_edge") is not None:
            extra += ", край окна" if read["near_edge"] else ", внутри окна"
        emit(
            f"Чтение аудио {dur:.2f} сек. ({sec(float(w0))}-{sec(float(w1))}), "
            f"прочитано слово «{target}» - {detect}{extra}"
        )

    word_trs = transliteration.surah_word_tr(surah)
    n_words = len(words)
    total_ms = len(pcm) * 1000.0 / sr

    # ayah -> ordered global word indices; raw chars per ayah for WINDOW SIZING
    # only (a rough ms/char is all the windows need; weighting it just perturbs
    # window widths and the emissions cache for no gain — see weighted units below).
    ayah_words: "OrderedDict[int, list[int]]" = OrderedDict()
    for i, w in enumerate(words):
        ayah_words.setdefault(w.ayah, []).append(i)
    ayah_list = list(ayah_words.keys())
    chars = {
        a: max(1, sum(len(word_trs[i]) for i in ayah_words[a] if i < len(word_trs)))
        for a in ayah_list
    }
    total_chars = max(1, sum(chars.values()))
    rate = total_ms / total_chars  # ms per char (window sizing only)

    def wlen(i: int) -> float:
        return len(word_trs[i]) * rate if i < len(word_trs) else rate

    # wav2vec2's conv feature-extractor needs a minimum input length.
    min_win_ms = 800.0

    # Transparent emissions cache: every computed window is kept, so a later read
    # that falls *inside* one (a refine window around a pause we already decoded)
    # reuses the emissions instead of re-running the model. ``audio_ms`` counts
    # only freshly computed audio, so it reflects the true model load (coverage).
    inv_vocab = {v: k for k, v in bundle.vocab.items()}
    emis_cache: list[tuple[int, int, Any, float]] = []  # (s0, s1, emis, stride_ms)
    audio_ms = 0.0

    def emissions_for(t0_ms: float, t1_ms: float):
        """Emissions over [t0,t1] as ``(emis, base_ms, stride_ms, end_ms)``.

        Returns a covering cached window's sub-slice when one exists (no model
        run), else computes, caches and accounts it. ``None`` if too short.
        """
        nonlocal audio_ms
        s0 = max(0, int(round(t0_ms * sr / 1000.0)))
        s1 = min(len(pcm), int(round(t1_ms * sr / 1000.0)))
        if s1 - s0 < int(0.3 * sr):
            return None
        for cs0, cs1, cemis, cstride in emis_cache:
            if cs0 <= s0 and s1 <= cs1:
                nf = cemis.shape[0]
                spf = (cs1 - cs0) / nf  # samples per frame
                f0 = max(0, int(round((s0 - cs0) / spf)))
                f1 = min(nf, int(round((s1 - cs0) / spf)))
                if f1 - f0 >= 2:
                    base = (cs0 + f0 * spf) * 1000.0 / sr
                    end = (cs0 + f1 * spf) * 1000.0 / sr
                    return cemis[f0:f1], base, cstride, end
        emis = A._emissions(bundle, np.ascontiguousarray(pcm[s0:s1], dtype=np.float32), np)
        stride = A._stride_ms(s1 - s0, sr, emis.shape[0])
        emis_cache.append((s0, s1, emis, stride))
        audio_ms += (s1 - s0) * 1000.0 / sr
        return emis, s0 * 1000.0 / sr, stride, s1 * 1000.0 / sr

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
        if i1 <= i0:  # no words
            return None
        sub_words = words[i0:i1]
        sub_trs = word_trs[i0:i1]
        basmala_trs = A._basmala_trs(transliteration) if basmala else None
        targets, spans = A._build_targets(sub_words, sub_trs, bundle, basmala_trs)
        if not targets:
            return None
        em = emissions_for(t0_ms, t1_ms)
        if em is None:  # < 0.3 s of audio
            return None
        emis, base, stride, end = em
        nf = emis.shape[0]
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
                "base": base, "end": end, "win_ms": end - base}

    # --- low-level: greedy CTC decode of [t0_ms, t1_ms] -> transliteration. ---
    # Unlike forced alignment (which is *told* the words), greedy decode reports
    # what the model actually *hears* — the signal that says which ayah opens
    # right after a pause, immune to the char prior's drift.
    def greedy_decode(t0_ms: float, t1_ms: float) -> str:
        em = emissions_for(t0_ms, t1_ms)
        if em is None:
            return ""
        ids = em[0].argmax(-1)
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
    pause_match_debug: list[dict] = []

    # ===================================================================== #
    # Phase 1a — detect pauses (cheap, no neural).                           #
    # ===================================================================== #
    pause_decisions = candidates.pause_pipeline_decisions(pcm, sr, surah=surah)
    pauses = [d.pause for d in pause_decisions if d.accepted]
    pause_diag = {d.pause: d for d in pause_decisions}
    pause_onsets = sorted(p.center_ms for p in pauses)
    ayah_boundary_count = max(0, len(ayah_list) - 1)
    waqf_mark_count = _count_waqf_marks(words)
    expected_pause_targets = ayah_boundary_count + waqf_mark_count
    accepted_pause_count = len(pauses)
    rejected_pause_count = len(pause_decisions) - accepted_pause_count
    emit(
        f"Сура {surah}: старт fast-разметки, длительность {sec(total_ms)} сек., "
        f"найдено пауз: {len(pauses)}"
    )
    emit(
        f"Бюджет пауз: границ аятов {ayah_boundary_count}, "
        f"waqf-знаков {waqf_mark_count}, ожидаемых целей ~{expected_pause_targets}; "
        f"принято пауз {accepted_pause_count}, отклонено {rejected_pause_count}, "
        f"отношение {accepted_pause_count / max(1, expected_pause_targets):.2f}"
    )
    for idx, p in enumerate(pauses, 1):
        d = pause_diag.get(p)
        dsp = (
            f", reason={d.reason}, rms={d.rms_ratio:.3f}, "
            f"flat={d.flatness:.3f}, zcr={d.zcr:.3f}, vad={d.vad_ok}"
            if d is not None else ""
        )
        emit(
            f"Пауза {idx}: {sec(p.start_ms)}-{sec(p.end_ms)} сек., "
            f"центр {sec(p.center_ms)} сек., глубина {p.depth_ms:.0f} мс{dsp}"
        )

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
    if res is None:
        emit("Fast не смог заякорить первый аят - ставлю грубый prior с conf=0")
        rows: list[tuple[int, int, float]] = []
        acc = 0
        for a in ayah_list:
            ms = int(round((acc / total_chars) * total_ms)) if total_chars else 0
            rows.append((a, ms, 0.0))
            acc += chars[a]
        if trace is not None:
            trace.update({
                "fast_anchor_failed": True,
                "drift_detected": True,
                "total_ms": total_ms,
                "audio_ms": audio_ms,
                "coverage": round(audio_ms / total_ms, 3) if total_ms else 0.0,
                "ayahs": [
                    {
                        "ayah": ayah,
                        "rough_ms": ms,
                        "from_pause": False,
                        "onset_ms": ms,
                        "conf": 0.0,
                        "win_s": 0.0,
                        "reliable": False,
                    }
                    for ayah, ms, _conf in rows
                ],
                "model_reads": model_reads,
                "phase1_pauses": [],
                "pause_matches": [],
                "n_pause": 0,
                "n_wasl": max(0, len(rows) - 1),
            })
        return rows
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
    log_read(model_reads[-1])
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
    # Prior position = content duration distributed by cumulative WEIGHTED units
    # (long vowels ×2 — a long vowel is recited longer). This beat both raw chars
    # and the mushaf render-width prior on the tempo probe. Used only for the
    # duration floor/ceiling and the repair interpolation, never for placement.
    wunit = {
        a: max(1.0, sum(_unit_len(word_trs[i]) for i in ayah_words[a] if i < len(word_trs)))
        for a in ayah_list
    }
    total_w = max(1.0, sum(wunit.values()))
    cum_u: dict[int, float] = {}
    _acc = 0.0
    for a in ayah_list:
        cum_u[a] = _acc
        _acc += wunit[a]
    content_dur = max(1.0, total_ms - content_start)
    inner = ayah_list[1:]  # boundaries we must place (ayah 1 is anchored)
    exp = [content_start + (cum_u[a] / total_w) * content_dur for a in inner]
    # Only pauses inside the content region can be ayah boundaries.
    pause_onsets = [p for p in pause_onsets if p > content_start - 200.0]
    pause_ord = {p: i + 1 for i, p in enumerate(pause_onsets)}
    pause_by_center = {p.center_ms: p for p in pauses}
    # Opening tokens (first four words) per inner ayah, and a greedy decode per
    # pause — the two sides of the acoustic match. Decode starts at the right edge
    # of the pause (with a tiny pre-roll), because the centre of a long pause is
    # still silence and makes the greedy string noisier.
    openings = [
        "".join(word_trs[i] for i in ayah_words[a][:4] if i < len(word_trs))
        for a in inner
    ]
    ambiguous_opening_idx = _ambiguous_openings(openings)
    decode_win_ms = _PAUSE_DECODE_WIN_MS
    decode_ambig_win_ms = max(decode_win_ms, _PAUSE_DECODE_AMBIG_WIN_MS)
    pause_decodes: dict[tuple[int, bool], str] = {}
    pause_decode_reasons: dict[float, str] = {}
    pause_decode_starts: dict[float, float] = {}

    def decode_pause(j: int, *, extended: bool = False) -> str:
        key = (j, extended)
        heard = pause_decodes.get(key)
        if heard is not None:
            return heard
        p = pause_onsets[j]
        pause = pause_by_center.get(p)
        decode_start = (
            max(0.0, pause.end_ms - _PAUSE_DECODE_PRE_ROLL_MS)
            if pause is not None else max(0.0, p - 50.0)
        )
        win_ms = decode_ambig_win_ms if extended else decode_win_ms
        decode_end = min(total_ms, decode_start + win_ms)
        edge_delta = decode_start - pause.end_ms if pause is not None else 0.0
        pause_decode_starts[p] = decode_start
        pause_decode_reasons[p] = "ambiguous-opening" if extended else "default"
        emissions_for(max(0.0, decode_start - _PAUSE_CACHE_PRE_ROLL_MS), decode_end)
        heard = greedy_decode(decode_start, decode_end)
        pause_decodes[key] = heard
        emit(
            f"Пауза {pause_ord[p]}: чтение аудио {(decode_end - decode_start) / 1000.0:.2f} сек. "
            f"после паузы {sec(pause.end_ms if pause is not None else p)} сек. "
            f"({sec(decode_start)}-{sec(decode_end)}), "
            f"старт относительно конца паузы {edge_delta:+.0f}мс, "
            f"reason={pause_decode_reasons[p]}, услышано «{short(heard)}»"
        )
        return heard

    sim_cache: dict[tuple[int, int, bool], float] = {}

    def needs_extended_decode(i: int, j: int) -> bool:
        return i in ambiguous_opening_idx and prior_delta(i, j) <= _AMBIG_DECODE_PRIOR_MS

    def _sim_ij(i: int, j: int) -> float:
        extended = needs_extended_decode(i, j)
        key = (i, j, extended)
        s = sim_cache.get(key)
        if s is None:
            s = _sim(decode_pause(j, extended=extended), openings[i])
            sim_cache[key] = s
        return s

    def prior_delta(i: int, j: int) -> float:
        return abs(pause_onsets[j] - exp[i])

    def far_weak_match(i: int, j: int, s: float) -> bool:
        return prior_delta(i, j) > _MATCH_PRIOR_WINDOW_MS and s < _FAR_MATCH_SIM

    def weak_pause(j: int) -> bool:
        pause = pause_by_center.get(pause_onsets[j])
        diag = pause_diag.get(pause) if pause is not None else None
        return bool(diag is not None and diag.reason == "weak-dsp")

    def prefix_mismatch(i: int, j: int, s: float) -> bool:
        heard = decode_pause(j, extended=needs_extended_decode(i, j))
        common = _common_prefix_len(heard, openings[i])
        if s >= _PREFIX_BYPASS_SIM:
            return False
        min_prefix = _PREFIX_WEAK_MIN if weak_pause(j) else _PREFIX_STRONG_MIN
        return common < min_prefix

    def reject_match(i: int, j: int, s: float) -> str | None:
        if far_weak_match(i, j, s):
            return "far_from_expected"
        if prefix_mismatch(i, j, s):
            return "prefix_mismatch"
        # If a pause clearly opens a later ayah, do not let the current ayah take
        # it as a weak anchor; the sequential matcher will turn it into a
        # future-gap event and recover/skips the missing span explicitly.
        if _GATE_PEN and s < _GATE_SMAX:
            for kk in (i + 1, i + 2):
                if kk < len(openings) and _sim_ij(kk, j) > s + _GATE_MARGIN:
                    return "future_better"
        return None

    def acoustic_cost(i: int, j: int) -> float:
        s = _sim_ij(i, j)
        if far_weak_match(i, j, s):
            return 1_000_000_000.0 + prior_delta(i, j)
        if _MIN_MATCH_SIM and s < _MIN_MATCH_SIM:
            return 50_000.0 + 3000.0 * (1.0 - s)
        cost = 3000.0 * (1.0 - s)  # lower cost = better-heard match
        # Don't let ayah i "steal" a pause that clearly opens a LATER ayah: if a
        # near-next ayah's opening is heard distinctly better after pause j, this
        # pause belongs to it — penalise so the DP leaves ayah i wasl instead of
        # an off-by-one grab (Yasin 29 took ayah 30's pause; 30 then fell on a
        # breath). Bounded lookahead keeps it local.
        if _GATE_PEN and s < _GATE_SMAX:
            for k in (i + 1, i + 2):
                if k < len(openings) and _sim_ij(k, j) > s + _GATE_MARGIN:
                    cost += _GATE_PEN
                    break
        return cost

    matched, sequence_events = _sequential_pause_match(
        exp,
        pause_onsets,
        _sim_ij,
        min_sim=_MIN_MATCH_SIM,
        prior_window_ms=_MATCH_PRIOR_WINDOW_MS,
        recover_min_sim=_RECOVER_MATCH_SIM,
        reject=reject_match,
    )
    for ev in sequence_events:
        action = ev.get("action")
        if action == "future_gap":
            j = ev.get("pause")
            bi = ev.get("boundary")
            hi = ev.get("heard_boundary")
            if isinstance(j, int) and isinstance(bi, int) and isinstance(hi, int):
                emit(
                    f"Пауза {pause_ord.get(pause_onsets[j], '?')} звучит как аят "
                    f"{inner[hi]}, а ожидался {inner[bi]}: sim={ev.get('heard_sim')}, "
                    f"current_sim={ev.get('current_sim')} - ищу пропущенный участок"
                )
        elif action == "recovered":
            j = ev.get("pause")
            bi = ev.get("boundary")
            if isinstance(j, int) and isinstance(bi, int):
                emit(
                    f"Пропущенный аят {inner[bi]} найден на паузе "
                    f"{pause_ord.get(pause_onsets[j], '?')}: sim={ev.get('sim')}"
                )
        elif action == "wasl":
            bi = ev.get("boundary")
            if isinstance(bi, int):
                emit(
                    f"Аят {inner[bi]} без подтверждённой паузы: "
                    f"reason={ev.get('reason')}, best_sim={ev.get('best_sim')} - "
                    "передаю в локальный wasl-align"
                )
    # ayah -> (rough_ms, from_pause). Wasl boundaries: interpolate between the
    # matched onsets that bracket them (anchored both sides — no drift).
    # First drop matches whose pause opening barely resembles the ayah — a likely
    # breath/wrong pause grab; leaving it wasl lets the continuous fill place it.
    dropped_match_reason: dict[int, str] = {}
    drop_sim = max(_WASL_SIM, _MIN_MATCH_SIM)
    if drop_sim:
        pidx = {ms: j for j, ms in enumerate(pause_onsets)}
        for k in range(len(matched)):
            mk = matched[k]
            if mk is None:
                continue
            j = pidx[mk]
            sim = _sim_ij(k, j)
            action = next(
                (
                    str(ev.get("action"))
                    for ev in sequence_events
                    if ev.get("boundary") == k
                    and ev.get("action") in ("confirmed", "recovered", "future_anchor")
                ),
                "",
            )
            min_keep_sim = _RECOVER_MATCH_SIM if action == "recovered" else drop_sim
            reason = None
            if sim < min_keep_sim:
                reason = "low_sim"
            elif far_weak_match(k, j, sim):
                reason = "far_from_expected"
            if reason is not None:
                matched[k] = None
                dropped_match_reason[k] = reason
    sequence_reason = {
        int(ev["boundary"]): str(ev.get("reason") or ev.get("action"))
        for ev in sequence_events
        if ev.get("action") == "wasl" and ev.get("boundary") is not None
    }
    sequence_action = {
        int(ev["boundary"]): str(ev.get("action"))
        for ev in sequence_events
        if ev.get("action") in ("confirmed", "recovered", "future_anchor")
        and ev.get("boundary") is not None
    }
    matched_center_for_ayah: dict[int, float] = {}

    def pause_timing_point(center_ms: float) -> float:
        return center_ms

    def strong_pause(center_ms: float) -> bool:
        pause = pause_by_center.get(center_ms)
        diag = pause_diag.get(pause) if pause is not None else None
        return bool(diag is not None and diag.reason != "weak-dsp")

    def nearest_pause_point(t: float, *, strong_only: bool = False) -> float | None:
        points: list[float] = []
        for center in pause_onsets:
            if strong_only and not strong_pause(center):
                continue
            points.append(center)
            points.append(pause_timing_point(center))
        if not points:
            return None
        return min(points, key=lambda p: abs(t - p))

    place: dict[int, tuple[float, bool]] = {}
    pause_match_by_ayah: dict[int, dict] = {}
    for k, a in enumerate(inner):
        near_js = [
            j for j, _p in enumerate(pause_onsets)
            if prior_delta(k, j) <= _MATCH_PRIOR_WINDOW_MS
        ]
        best_j, best_sim = max(
            ((j, _sim_ij(k, j)) for j in near_js),
            key=lambda item: item[1],
            default=(-1, 0.0),
        )
        best_near_j, best_near_sim = max(
            ((j, _sim_ij(k, j)) for j in near_js),
            key=lambda item: item[1],
            default=(-1, 0.0),
        )
        expected = exp[k]
        if matched[k] is not None:
            j = pause_onsets.index(matched[k])
            sim = _sim_ij(k, j)
            rough = pause_timing_point(matched[k])
            matched_center_for_ayah[a] = matched[k]
            place[a] = (rough, True)
            pause = pause_by_center.get(matched[k])
            pause_match_by_ayah[a] = {
                "pause": pause_ord.get(matched[k]),
                "sim": sim,
                "center_ms": matched[k],
                "rough_ms": rough,
                "action": sequence_action.get(k),
            }
            pause_match_debug.append({
                "ayah": a,
                "expected_ms": round(expected, 1),
                "rough_ms": round(rough, 1),
                "from_pause": True,
                "pause": pause_ord.get(matched[k]),
                "pause_center_ms": round(matched[k], 1),
                "pause_end_ms": round(pause.end_ms, 1) if pause is not None else None,
                "pause_sim": round(sim, 3),
                "best_pause": pause_ord.get(pause_onsets[best_j]) if best_j >= 0 else None,
                "best_sim": round(best_sim, 3),
                "best_near_pause": (
                    pause_ord.get(pause_onsets[best_near_j]) if best_near_j >= 0 else None
                ),
                "best_near_sim": round(best_near_sim, 3),
                "prior_window_ms": round(_MATCH_PRIOR_WINDOW_MS, 1),
                "expected_delta_ms": round(rough - expected, 1),
                "drop_reason": None,
                "match_action": sequence_action.get(k),
            })
            emit(
                f"Пауза {pause_ord.get(matched[k], '?')} -> детект аята {a}, "
                f"черновой старт {sec(rough)} сек., sim={sim:.3f}, "
                f"ожидание {sec(expected)} сек., delta_prior={rough - expected:+.0f}мс, "
                f"best_near={pause_ord.get(pause_onsets[best_near_j], '?') if best_near_j >= 0 else '?'}"
                f"/{best_near_sim:.3f}"
            )
        else:
            drop_reason = dropped_match_reason.get(k) or sequence_reason.get(k)
            rough = _interp_wasl(k, exp, matched)
            place[a] = (_interp_wasl(k, exp, matched), False)
            pause_match_debug.append({
                "ayah": a,
                "expected_ms": round(expected, 1),
                "rough_ms": round(rough, 1),
                "from_pause": False,
                "pause": None,
                "pause_sim": None,
                "best_pause": pause_ord.get(pause_onsets[best_j]) if best_j >= 0 else None,
                "best_sim": round(best_sim, 3),
                "best_near_pause": (
                    pause_ord.get(pause_onsets[best_near_j]) if best_near_j >= 0 else None
                ),
                "best_near_sim": round(best_near_sim, 3),
                "prior_window_ms": round(_MATCH_PRIOR_WINDOW_MS, 1),
                "expected_delta_ms": round(rough - expected, 1),
                "drop_reason": drop_reason or "dp_wasl",
                "match_action": "wasl",
            })
            emit(
                f"Без паузы -> детект середины/wasl аята {a}, "
                f"черновой старт {sec(place[a][0])} сек., "
                f"best_pause={pause_ord.get(pause_onsets[best_j], '?') if best_j >= 0 else '?'}, "
                f"best_sim={best_sim:.3f}, "
                f"best_near={pause_ord.get(pause_onsets[best_near_j], '?') if best_near_j >= 0 else '?'}/"
                f"{best_near_sim:.3f}, reason={drop_reason or 'dp_wasl'}"
            )
    used_pauses = {item["pause"] for item in pause_match_debug if item["pause"] is not None}
    skipped_pauses = [
        pause_ord[p] for p in pause_onsets if pause_ord[p] not in used_pauses
    ]
    emit(
        f"Сопоставление пауз: использовано {len(used_pauses)}, "
        f"пропущено {len(skipped_pauses)}; skipped={skipped_pauses}"
    )

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
        onset, conf, win_used = None, 0.0, 0.0
        best_score: tuple[int, float] | None = None  # (inside_window, conf)
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
            cand = res["starts"][fa - i0]
            cconf = res["confs"][fa - i0]
            guard = min(edge_guard_ms, 0.2 * res["win_ms"])
            near_edge = (cand - res["base"] < guard) or (res["end"] - cand < guard)
            # Keep the genuinely best window seen, not merely the last one: a
            # comfortably-inside onset is preferred over a near-edge one (its
            # boundary may lie outside that window), then higher confidence.
            score = (0 if near_edge else 1, cconf)
            if best_score is None or score > best_score:
                best_score = score
                onset, conf, win_used = cand, cconf, res["win_ms"] / 1000.0
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
            log_read(model_reads[-1])
            if cconf >= conf_thresh and not near_edge:
                break  # boundary comfortably inside, confident → trust it
        return onset, conf, win_used

    # A run of consecutive *wasl* ayahs (no pause) is not refined per-ayah: the
    # char prior interpolates each one's centre too early and a narrow window can
    # miss the true onset entirely (its right edge falls before the boundary). We
    # instead align the WHOLE run in one window between its bracketing anchors —
    # the continuous forced alignment that the full path uses locally — and read
    # off each ayah's first-word onset. Accurate and one read per run.
    wasl_onsets: dict[int, tuple[float, float]] = {}
    wasl_pause_hints: dict[int, float] = {}

    def ayah_opening(a: int) -> str:
        return "".join(word_trs[i] for i in ayah_words[a][:4] if i < len(word_trs))

    def fill_wasl_run(ai: int) -> None:
        left_a = ayah_list[ai - 1]  # always an anchor (run starts after one)
        i0 = ayah_words[left_a][0]
        t0 = starts_ms[left_a]
        j = ai
        run: list[int] = []
        while j < len(ayah_list) and not place[ayah_list[j]][1]:
            run.append(ayah_list[j])
            j += 1
        if j < len(ayah_list):  # bracketed on the right by the next anchor
            right_a = ayah_list[j]
            i1 = ayah_words[right_a][0] + 1  # include its first word as the closer
            t1 = place[right_a][0]
        else:  # trailing wasl run: to the end of the sura
            i1 = n_words
            t1 = total_ms
        if t1 - t0 > _MAX_WASL_RUN_ALIGN_MS:
            for a in run:
                wasl_onsets[a] = (place[a][0], 0.0)
            emit(
                f"Wasl-серия {run[0]}-{run[-1]}: окно {(t1 - t0) / 1000.0:.2f} сек. "
                f"слишком длинное для fast, оставляю rough без цельного align"
            )
            return
        res = align_slice(i0, i1, t0 - 60.0, t1 + 60.0, basmala=False)
        for a in run:
            fa = ayah_words[a][0]
            if res is None:
                wasl_onsets[a] = (place[a][0], 0.0)
            else:
                k = fa - i0
                onset = res["starts"][k]
                verify = _sim(
                    greedy_decode(onset - 50.0, onset + _VERIFY_WINDOW_MS),
                    ayah_opening(a),
                )
                wasl_onsets[a] = (onset, max(res["confs"][k], verify))
                hint = nearest_pause_point(onset, strong_only=True)
                if hint is not None and abs(onset - hint) <= _WASL_PAUSE_HINT_MS:
                    wasl_pause_hints[a] = hint
                    wasl_onsets[a] = (hint, wasl_onsets[a][1])
        if res is not None:
            model_reads.append({
                "phase": "wasl-run",
                "ayah": run[0],
                "window_ms": [round(res["base"], 1), round(res["end"], 1)],
                "words": " ".join(w.text for w in words[i0:i1]),
                "target": " ".join(words[ayah_words[a][0]].text for a in run),
                "onset_ms": round(wasl_onsets[run[0]][0], 1),
                "conf": round(wasl_onsets[run[0]][1], 3),
            })
            log_read(model_reads[-1])

    for ai in range(1, len(ayah_list)):
        a = ayah_list[ai]
        fa = ayah_words[a][0]
        c0, from_pause = place[a]

        if from_pause:
            prev_start = starts_ms[ayah_list[ai - 1]]
            c = min(max(c0, prev_start + 30.0), total_ms)
            match = pause_match_by_ayah.get(a, {})
            pause_sim = float(match.get("sim") or 0.0)
            if pause_sim >= _PAUSE_TRUST_SIM:
                onset, conf, win_used = c, max(conf_thresh, pause_sim), 0.0
                reliable = True
                emit(
                    f"Пауза {match.get('pause', '?')} подтвердила аят {a}: "
                    f"sim={pause_sim:.3f}, старт {sec(onset)} сек., refine не нужен"
                )
            else:
                # Weak pause matches still get a tiny local forced-align window.
                c = min(max(c0, prev_start + 30.0), total_ms)
                onset, conf, win_used = refine(c, fa, prev_start, 0, ayah=a)
                reliable = onset is not None and conf >= conf_thresh
                if not reliable:  # never locked on → trust the matched guess
                    onset = c if onset is None else onset
        else:
            if a not in wasl_onsets:
                fill_wasl_run(ai)
            onset, conf = wasl_onsets[a]
            win_used = 0.0
            reliable = conf >= conf_thresh
            if onset is None:
                onset = c0

        starts_ms[a] = onset
        confs[a] = conf
        tr_ayahs.append({
            "ayah": a,
            "rough_ms": c0,
            "from_pause": from_pause,
            "onset_ms": onset,
            "conf": round(conf, 3),
            "win_s": round(win_used, 2),
            "reliable": reliable,
            "refine_skipped": bool(from_pause and win_used == 0.0 and reliable),
        })

    # --- Phase 2b — repair shifted runs using reliable neighbours. ----------
    # Char-prior compression can make the global match shift a run of boundaries
    # onto earlier pauses (detectable as low neural confidence). Re-anchor such
    # ayahs by interpolating between the nearest CONFIDENT onsets (real anchors
    # average out melodic drift), then re-run the neural window around that guess.
    cum = cum_u  # cumulative weighted units (already built for the prior above)
    rel = {a: confs.get(a, 0.0) >= conf_reliable for a in ayah_list}
    rel[ayah_list[0]] = True
    # A real boundary sits at a pause; a repair that moves a pause-backed onset
    # (which the model was at least somewhat sure of) to a non-pause spot is a
    # spurious refine match — keep the original (sura 80 a3). Garbage-conf onsets
    # (< conf_thresh, never locked on) are exempt: geometry may move them freely
    # (sura 89 a23). ``pause_onsets`` is the detected pause grid from Phase 1c.
    pause_grid = pause_onsets + [pause_timing_point(p) for p in pause_onsets]

    def _dist_pause(t: float) -> float:
        return min((abs(t - p) for p in pause_grid), default=1e9)

    # Measured tempo (ms per weighted unit) from reliable anchors — robust to the
    # basmala / trailing silence that inflates the global ``rate``; used to
    # extrapolate a repair guess past the last reliable anchor.
    _pts = [(cum[a], starts_ms[a]) for a in ayah_list if rel[a] and a in starts_ms]
    _slopes = sorted(
        (t1 - t0) / (u1 - u0)
        for (u0, t0), (u1, t1) in zip(_pts, _pts[1:]) if u1 > u0
    )
    tempo_meas = _slopes[len(_slopes) // 2] if _slopes else rate
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
            pred = starts_ms[lo] + (cum[a] - cum[lo]) * tempo_meas
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
            # A refine that still never locks on (conf below thresh) returns a
            # noisy onset — on a long melodic ayah the model may not find the
            # boundary at all. Geometry between two reliable anchors beats model
            # noise, so fall back to the interpolated guess in that case.
            chosen = o2 if c2 >= conf_thresh else pred
            old_onset = starts_ms.get(a, pred)
            pause_hint = wasl_pause_hints.get(a)
            if (
                pause_hint is not None
                and abs(old_onset - pause_hint) <= _WASL_PAUSE_HINT_MS
                and abs(chosen - pause_hint)
                > abs(old_onset - pause_hint) + _REPAIR_PAUSE_PULL_MS
            ):
                emit(
                    f"Repair аята {a} отклонён: wasl-старт {sec(old_onset)} сек. "
                    f"рядом с сильной паузой {sec(pause_hint)} сек., "
                    f"новый старт {sec(chosen)} сек. уводит от неё"
                )
                continue
            # Don't pull a decent pause-backed onset off the pause grid (spurious).
            if (place[a][1] and confs.get(a, 0.0) >= conf_thresh
                    and _dist_pause(chosen) > 350.0):
                continue
            starts_ms[a] = chosen
            confs[a] = c2
            for t in tr_ayahs:
                if t["ayah"] == a:
                    t.update(onset_ms=chosen, conf=round(c2, 3), repaired=True)
                    break

    def write_fast_trace(extra: dict | None = None) -> None:
        if trace is None:
            return
        pause_decode_by_ord: dict[int, str] = {}
        for (i, extended), heard in pause_decodes.items():
            p = pause_onsets[i]
            if extended or pause_ord[p] not in pause_decode_by_ord:
                pause_decode_by_ord[pause_ord[p]] = heard
        used_pause_ord = {
            int(item["pause"])
            for item in pause_match_debug
            if item.get("pause") is not None
        }

        def sequence_trace_event(ev: dict) -> dict:
            out = dict(ev)
            boundary = out.pop("boundary", None)
            if isinstance(boundary, int) and 0 <= boundary < len(inner):
                out["ayah"] = inner[boundary]
            heard_boundary = out.pop("heard_boundary", None)
            if isinstance(heard_boundary, int) and 0 <= heard_boundary < len(inner):
                out["heard_ayah"] = inner[heard_boundary]
            pause_idx = out.get("pause")
            if isinstance(pause_idx, int) and 0 <= pause_idx < len(pause_onsets):
                out["pause_idx"] = pause_ord.get(pause_onsets[pause_idx])
                out["pause_center_ms"] = round(pause_onsets[pause_idx], 1)
                del out["pause"]
            return out

        trace["phase1_pauses"] = [
            {
                "idx": pause_ord.get(p.center_ms),
                "start_ms": round(p.start_ms, 1),
                "end_ms": round(p.end_ms, 1),
                "center_ms": round(p.center_ms, 1),
                "depth_ms": round(p.depth_ms, 1),
                "reason": pause_diag[p].reason if p in pause_diag else None,
                "quality": (
                    "weak"
                    if p in pause_diag and pause_diag[p].reason == "weak-dsp"
                    else "strong"
                ),
                "rms_ratio": round(pause_diag[p].rms_ratio, 4) if p in pause_diag else None,
                "flatness": round(pause_diag[p].flatness, 4) if p in pause_diag else None,
                "zcr": round(pause_diag[p].zcr, 4) if p in pause_diag else None,
                "vad_ok": pause_diag[p].vad_ok if p in pause_diag else None,
                "decode_reason": pause_decode_reasons.get(p.center_ms),
                "decode_start_vs_pause_end_ms": round(
                    (
                        pause_ord.get(p.center_ms)
                        and (pause_decode_starts.get(p.center_ms, p.center_ms) - p.end_ms)
                    ) or 0.0,
                    1,
                ),
                "decode": short(pause_decode_by_ord.get(pause_ord.get(p.center_ms), ""), 120),
            }
            for p in pauses
        ]
        trace["pause_matches"] = pause_match_debug
        trace["pause_sequence_events"] = [
            sequence_trace_event(ev) for ev in sequence_events
        ]
        trace["pause_budget"] = {
            "ayah_boundaries": ayah_boundary_count,
            "waqf_marks": waqf_mark_count,
            "expected_pause_targets": expected_pause_targets,
            "candidate_pauses": len(pause_decisions),
            "accepted_pauses": accepted_pause_count,
            "rejected_pauses": rejected_pause_count,
            "used_pauses": len(used_pause_ord),
            "decoded_pauses": len({i for i, _extended in pause_decodes}),
            "decode_reads": len(pause_decodes),
            "ambiguous_openings": [
                inner[i] for i in sorted(ambiguous_opening_idx)
            ],
            "pause_target_ratio": round(
                accepted_pause_count / max(1, expected_pause_targets), 3
            ),
            "used_accept_ratio": round(
                len(used_pause_ord) / max(1, accepted_pause_count), 3
            ),
        }
        trace["model_reads"] = model_reads
        trace["ayahs"] = tr_ayahs
        trace["audio_ms"] = audio_ms
        trace["total_ms"] = total_ms
        trace["n_pause"] = sum(1 for t in tr_ayahs if t["from_pause"])
        trace["n_wasl"] = sum(
            1 for t in tr_ayahs if not t["from_pause"] and t["ayah"] != ayah_list[0]
        )
        trace["n_refine_skipped"] = sum(1 for t in tr_ayahs if t.get("refine_skipped"))
        trace["coverage"] = round(audio_ms / total_ms, 3) if total_ms else 0.0
        if extra:
            trace.update(extra)

    # --- self-diagnostic: drifted sura → mark weak, keep fast result. --------
    # A cascaded off-by-one with no recovery anchor drives the mean boundary
    # confidence down. Fast mode keeps its result and lets review highlight weak
    # rows; the user can explicitly choose full/tiled when needed.
    inner_confs = [confs.get(a, 0.0) for a in ayah_list[1:]]
    mean_conf = sum(inner_confs) / len(inner_confs) if inner_confs else 1.0
    # Independent check: does the audio at each placed boundary actually sound like
    # that ayah's opening? Low across the board ⇒ drift (mean-conf can miss this).
    pidx_by_onset = {ms: j for j, ms in enumerate(pause_onsets)}
    verify_extra_reads = 0
    vsims = []
    for k, a in enumerate(inner):
        if a not in starts_ms:
            continue
        heard = ""
        rough, from_pause = place.get(a, (None, False))
        if from_pause and rough is not None:
            center = matched_center_for_ayah.get(a, rough)
            j = pidx_by_onset.get(center)
            if j is not None:
                heard = decode_pause(j, extended=needs_extended_decode(k, j))
        if not heard:
            verify_extra_reads += 1
            heard = greedy_decode(starts_ms[a] - 50.0, starts_ms[a] + _VERIFY_WINDOW_MS)
        vsims.append(_sim(heard, openings[k]))
    verify_sim = sum(vsims) / len(vsims) if vsims else 1.0
    if trace is not None:
        trace["verify_sim"] = round(verify_sim, 3)
        trace["mean_conf"] = round(mean_conf, 3)
        trace["verify_extra_reads"] = verify_extra_reads
        trace["pause_coverage"] = round(
            sum(1 for t in tr_ayahs if t["from_pause"]) / max(1, len(inner)),
            3,
        )
    pause_coverage = sum(1 for t in tr_ayahs if t["from_pause"]) / max(1, len(inner))
    drifted = bool(_VERIFY_SIM and verify_sim < _VERIFY_SIM)
    if (
        not drifted
        and _FALLBACK_CONF
        and mean_conf < _FALLBACK_CONF
        and pause_coverage < _DRIFT_MIN_PAUSE_COVERAGE
    ):
        drifted = True
    if drifted:
        emit(
            f"Сура {surah}: fast заподозрил дрейф "
            f"(mean_conf={mean_conf:.3f}, verify_sim={verify_sim:.3f}) - "
            "оставляю fast и помечаю слабые места"
        )

    # --- shape: monotonic non-decreasing starts, per-ayah rows. ------------
    rows: list[tuple[int, int, float]] = []
    last = 0
    for a in ayah_list:
        ms = int(round(starts_ms.get(a, last)))
        ms = max(ms, last)
        rows.append((a, ms, float(confs.get(a, 0.0))))
        last = ms

    write_fast_trace({
        "drift_detected": bool(drifted),
        "mean_conf": round(mean_conf, 3),
        "verify_sim": round(verify_sim, 3),
    })
    return rows
