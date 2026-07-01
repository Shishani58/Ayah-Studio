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
import math
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
#: Hard gate for accepting a pause as an ayah anchor. Short greedy decodes can
#: be noisy on brief openings, so this matches the recovery floor; stricter
#: confirmation still uses ``_CONFIRM_MATCH_SIM``.
_MIN_MATCH_SIM = float(os.environ.get("SC_FAST_MIN_MATCH_SIM", "0.31"))
_CONFIRM_MATCH_SIM = float(os.environ.get("SC_FAST_CONFIRM_MATCH_SIM", "0.50"))
#: Recovery pass is only used after a later pause already proved we skipped over
#: an ayah. In that narrow context we can accept a slightly weaker similarity for
#: an intermediate pause instead of forcing the ayah into a long wasl window.
_RECOVER_MATCH_SIM = float(os.environ.get("SC_FAST_RECOVER_MATCH_SIM", "0.34"))
_RECOVER_ECHO_SIM_SLACK = float(os.environ.get("SC_FAST_RECOVER_ECHO_SIM_SLACK", "0.07"))
_FINAL_WEAK_MATCH_SIM = float(os.environ.get("SC_FAST_FINAL_WEAK_MATCH_SIM", "0.18"))
_FINAL_WEAK_TEMPO_FLOOR = float(os.environ.get("SC_FAST_FINAL_WEAK_TEMPO_FLOOR", "-0.02"))
_CONFIRMED_EARLY_CURRENT_SLACK = float(
    os.environ.get("SC_FAST_CONFIRMED_EARLY_CURRENT_SLACK", "0.08")
)
_BASMALA_MIN_CONTENT_START_MS = float(
    os.environ.get("SC_FAST_BASMALA_MIN_CONTENT_START_MS", "6000")
)
_SHORT_PAUSE_RIGHT_BIAS = float(os.environ.get("SC_FAST_SHORT_PAUSE_RIGHT_BIAS", "0.25"))
_LONG_PAUSE_CENTER_MS = float(os.environ.get("SC_FAST_LONG_PAUSE_CENTER_MS", "1300"))
_TRUSTED_PAUSE_CENTER_MS = float(os.environ.get("SC_FAST_TRUSTED_PAUSE_CENTER_MS", "1300"))
_COMPACT_PAUSE_CENTER_MS = float(os.environ.get("SC_FAST_COMPACT_PAUSE_CENTER_MS", "1100"))
_MICRO_BACKTRACK_MS = float(os.environ.get("SC_FAST_MICRO_BACKTRACK_MS", "3500"))
_FAR_WEAK_CURRENT_GRACE_MS = float(os.environ.get("SC_FAST_FAR_WEAK_CURRENT_GRACE_MS", "9000"))
_FAR_WEAK_CURRENT_SIM = float(os.environ.get("SC_FAST_FAR_WEAK_CURRENT_SIM", "0.42"))
_FAR_CURRENT_MAX_OVERRIDE_MS = float(os.environ.get("SC_FAST_FAR_CURRENT_MAX_OVERRIDE_MS", "30000"))
#: If the short decode after a pause already identifies the ayah this well, the
#: pause edge is trusted as the timing and Phase 2 does not run another
#: forced-align window for that boundary.
_PAUSE_TRUST_SIM = float(os.environ.get("SC_FAST_PAUSE_TRUST_SIM", "0.50"))
#: A weak match far from the structural/tempo prior is almost certainly a random
#: similar syllable elsewhere in the sura. Strong matches can still override it.
_MATCH_PRIOR_WINDOW_MS = float(os.environ.get("SC_FAST_MATCH_PRIOR_WINDOW_MS", "15000"))
_MATCH_PRIOR_GRACE_MS = float(os.environ.get("SC_FAST_MATCH_PRIOR_GRACE_MS", "1000"))
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
_PAUSE_DECODE_TEXT_MAX_WIN_MS = float(os.environ.get("SC_FAST_PAUSE_DECODE_TEXT_MAX_WIN_MS", "8500"))
_PAUSE_DECODE_TEXT_MS_PER_CHAR = float(os.environ.get("SC_FAST_PAUSE_DECODE_TEXT_MS_PER_CHAR", "140"))
_AMBIG_DECODE_PRIOR_MS = float(os.environ.get("SC_FAST_AMBIG_DECODE_PRIOR_MS", "7000"))
# A weak intra-ayah breath may fuzzy-match the next ayah because the short decode
# contains a tail from the previous ayah plus a little future text. Require the
# decode to actually begin like the target opening before accepting such pauses.
_PREFIX_WEAK_MIN = int(os.environ.get("SC_FAST_PREFIX_WEAK_MIN", "2"))
_PREFIX_STRONG_MIN = int(os.environ.get("SC_FAST_PREFIX_STRONG_MIN", "0"))
_PREFIX_BORDERLINE_MIN = int(os.environ.get("SC_FAST_PREFIX_BORDERLINE_MIN", "4"))
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
_PAUSE_REFINE_CONFIRM_MS = float(os.environ.get("SC_FAST_PAUSE_REFINE_CONFIRM_MS", "900"))
_PAUSE_REFINE_CONFIRM_CONF = float(os.environ.get("SC_FAST_PAUSE_REFINE_CONFIRM_CONF", "0.05"))
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
# Soft wave/tempo prior. It compares speech spans around a pause with expected
# ayah duration from text units. Kept deliberately small: greedy decode remains
# the primary proof, this only breaks ties and filters weak internal pauses.
_TEMPO_SCORE_WEIGHT = float(os.environ.get("SC_FAST_TEMPO_SCORE_WEIGHT", "0.08"))
_TEMPO_BAD_SCORE = float(os.environ.get("SC_FAST_TEMPO_BAD_SCORE", "-0.06"))
_TEMPO_WEAK_BYPASS_SIM = float(os.environ.get("SC_FAST_TEMPO_WEAK_BYPASS_SIM", "0.55"))
# When text/tempo says a boundary is missing inside a speech span, scan around
# the predicted point for a local RMS valley and inject it as a synthetic weak
# pause candidate. This is intentionally conservative and only runs after a
# future-gap proves that an ayah was skipped.
_MICRO_PAUSE_SEARCH_MS = float(os.environ.get("SC_FAST_MICRO_PAUSE_SEARCH_MS", "2500"))
_MICRO_PAUSE_MIN_SEP_MS = float(os.environ.get("SC_FAST_MICRO_PAUSE_MIN_SEP_MS", "280"))
_MICRO_PAUSE_FRAME_MS = float(os.environ.get("SC_FAST_MICRO_PAUSE_FRAME_MS", "24"))
_MICRO_PAUSE_HOP_MS = float(os.environ.get("SC_FAST_MICRO_PAUSE_HOP_MS", "8"))
_MICRO_PAUSE_MAX_MS = float(os.environ.get("SC_FAST_MICRO_PAUSE_MAX_MS", "180"))
_MICRO_PAUSE_SHELF_MAX_MS = float(os.environ.get("SC_FAST_MICRO_PAUSE_SHELF_MAX_MS", "900"))
_MICRO_PAUSE_SHELF_RATIO = float(os.environ.get("SC_FAST_MICRO_PAUSE_SHELF_RATIO", "1.65"))
_MICRO_PAUSE_RMS_RATIO = float(os.environ.get("SC_FAST_MICRO_PAUSE_RMS_RATIO", "0.72"))

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
    return _common_prefix_len_collapsed(_collapse(a), _collapse(b))


def _common_prefix_len_collapsed(a: str, b: str) -> int:
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    return common


def _prefix_key(s: str) -> str:
    return _collapse("".join(c for c in s if c.isalnum()))


def _onset_prefix_len(a: str, b: str) -> int:
    """Common prefix for onset checks, tolerant of separator/noisy article loss."""
    ak = _prefix_key(a)
    bk = _prefix_key(b)
    common = _common_prefix_len_collapsed(ak, bk)
    variants: list[tuple[str, str]] = []
    if ak.startswith("a") and bk.startswith("l"):
        variants.append((ak[1:], bk))
    if bk.startswith("a") and ak.startswith("l"):
        variants.append((ak, bk[1:]))
    for va, vb in variants:
        common = max(common, _common_prefix_len_collapsed(va, vb))
    return common


#: Latin long-vowel (madd-carrier) chars. As a *tempo unit* a long vowel is
#: recited longer, so it weighs double. The tempo probe (tools/bench_fast) showed
#: this proxy predicts where ayah boundaries land in audio better than raw char
#: count and than the mushaf render-width prior used before.
_LONG_VOWELS = frozenset("āīū")


def _unit_len(s: str) -> float:
    """Tempo-unit length of a transliteration token (long vowels count double)."""
    return float(sum(2 if c in _LONG_VOWELS else 1 for c in s))


def _span_ratio_quality(actual_ms: float, expected_ms: float) -> float:
    """Return [0,1] quality for how well an audio span fits an expected duration."""
    if actual_ms <= 0.0 or expected_ms <= 0.0:
        return 0.0
    ratio = actual_ms / expected_ms
    # A factor of 2.5 mismatch is already very suspicious, but not impossible in
    # melodic recitation; clamp softly so this remains a prior, not a hard rule.
    quality = max(0.0, 1.0 - abs(math.log(ratio)) / math.log(2.5))
    return 0.0 if quality < 1e-9 else quality


def _find_micro_pause_valley(
    pcm,
    sr: int,
    *,
    interval_start_ms: float,
    interval_end_ms: float,
    expected_ms: float,
    search_ms: float = _MICRO_PAUSE_SEARCH_MS,
    frame_ms: float = _MICRO_PAUSE_FRAME_MS,
    hop_ms: float = _MICRO_PAUSE_HOP_MS,
    max_width_ms: float = _MICRO_PAUSE_MAX_MS,
    shelf_max_width_ms: float = _MICRO_PAUSE_SHELF_MAX_MS,
    shelf_ratio: float = _MICRO_PAUSE_SHELF_RATIO,
    rms_ratio: float = _MICRO_PAUSE_RMS_RATIO,
) -> tuple[float, float, float, float, float] | None:
    """Find a local low-energy valley near ``expected_ms`` inside a speech span.

    Returns ``(start_ms, end_ms, ratio, seed_start_ms, seed_end_ms)`` where the
    first pair is an expanded quiet shelf around the seed valley, and ratio is
    seed RMS / local median RMS. ``None`` means the dip is not convincing. This
    intentionally does not classify the candidate as a Quran boundary; it only
    proposes a pause-like point for the normal greedy-decode verifier.
    """
    import numpy as np

    lo_ms = max(interval_start_ms, expected_ms - search_ms)
    hi_ms = min(interval_end_ms, expected_ms + search_ms)
    if hi_ms - lo_ms < max(160.0, frame_ms * 3):
        return None
    s0 = max(0, int(round(lo_ms * sr / 1000.0)))
    s1 = min(len(pcm), int(round(hi_ms * sr / 1000.0)))
    frame = max(8, int(round(frame_ms * sr / 1000.0)))
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    if s1 - s0 < frame * 3:
        return None
    rms: list[float] = []
    centers: list[float] = []
    arr = np.asarray(pcm, dtype=np.float32)
    for start in range(s0, s1 - frame + 1, hop):
        chunk = arr[start:start + frame]
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        centers.append((start + frame / 2) * 1000.0 / sr)
    if len(rms) < 3:
        return None
    values = np.asarray(rms, dtype=np.float32)
    median = float(np.median(values))
    if median <= 1e-8:
        return None
    min_i = int(np.argmin(values))
    ratio = float(values[min_i] / median)
    if ratio > rms_ratio:
        return None
    threshold = min(median * rms_ratio, float(np.percentile(values, 30)))
    left = min_i
    while left > 0 and values[left - 1] <= threshold:
        left -= 1
    right = min_i
    while right + 1 < len(values) and values[right + 1] <= threshold:
        right += 1
    seed_start_ms = centers[left] - frame_ms / 2.0
    seed_end_ms = centers[right] + frame_ms / 2.0
    center = centers[min_i]
    if seed_end_ms - seed_start_ms > max_width_ms:
        seed_start_ms = center - max_width_ms / 2.0
        seed_end_ms = center + max_width_ms / 2.0
    shelf_threshold = min(
        median * min(0.85, shelf_ratio * rms_ratio),
        float(np.percentile(values, 45)),
    )
    shelf_left = min_i
    while shelf_left > 0 and values[shelf_left - 1] <= shelf_threshold:
        shelf_left -= 1
    shelf_right = min_i
    while shelf_right + 1 < len(values) and values[shelf_right + 1] <= shelf_threshold:
        shelf_right += 1
    start_ms = centers[shelf_left] - frame_ms / 2.0
    end_ms = centers[shelf_right] + frame_ms / 2.0
    if end_ms - start_ms > shelf_max_width_ms:
        start_ms = center - shelf_max_width_ms / 2.0
        end_ms = center + shelf_max_width_ms / 2.0
    start_ms = max(interval_start_ms, start_ms)
    end_ms = min(interval_end_ms, end_ms)
    seed_start_ms = max(interval_start_ms, seed_start_ms)
    seed_end_ms = min(interval_end_ms, seed_end_ms)
    if end_ms <= start_ms:
        return None
    return start_ms, end_ms, ratio, seed_start_ms, seed_end_ms


#: Chars of the (collapsed) decode/opening compared. Must span ~4 words so that
#: ayahs sharing an opening phrase (e.g. al-Bayyinah 6/7 "inna lladhīna kafarū"
#: vs "…āmanū", al-Falaq's repeated "wa-min sharri…") are told apart by a later,
#: differing word — the shared prefix alone is ambiguous.
_SIM_CHARS = 26
_SIM_TEXT_MARGIN_CHARS = int(os.environ.get("SC_FAST_SIM_TEXT_MARGIN_CHARS", "18"))
_SIM_TEXT_MAX_CHARS = int(os.environ.get("SC_FAST_SIM_TEXT_MAX_CHARS", "96"))
_SIM_PREFIX_WEIGHT = float(os.environ.get("SC_FAST_SIM_PREFIX_WEIGHT", "0.35"))
_INTERNAL_REPEAT_MATCH_MARGIN = float(os.environ.get("SC_FAST_INTERNAL_REPEAT_MATCH_MARGIN", "0.08"))


def _sim(dec: str, opening: str, *, chars: int = _SIM_CHARS) -> float:
    """Fuzzy similarity of a greedy-decoded onset to an ayah's opening tokens.

    Both are collapsed (recitation stretches vowels) and capped at ``chars``
    chars. The score is intentionally onset-aware: a decode that contains the
    target later after a previous-ayah tail is worse than one that starts with
    the target. Returns ``[0, 1]``.
    """
    chars = max(1, int(chars))
    dec_c = _collapse(dec)[:chars]
    opening_c = _collapse(opening)[:chars]
    fuzzy = difflib.SequenceMatcher(None, dec_c, opening_c).ratio()
    common = _common_prefix_len(dec_c, opening_c)
    prefix = common / max(1, min(chars, len(opening_c)))
    weight = min(1.0, max(0.0, _SIM_PREFIX_WEIGHT))
    return fuzzy * (1.0 - weight) + prefix * weight


def _pause_refine_confirms_pause(
    pause_ms: float,
    onset_ms: float | None,
    conf: float,
    *,
    max_delta_ms: float = _PAUSE_REFINE_CONFIRM_MS,
    min_conf: float = _PAUSE_REFINE_CONFIRM_CONF,
) -> bool:
    """Whether a local refine found the target word close enough to trust pause."""
    return (
        onset_ms is not None
        and conf >= min_conf
        and abs(onset_ms - pause_ms) <= max_delta_ms
    )


def _pause_center_with_refined_onset(
    pause_start_ms: float,
    pause_center_ms: float,
    refined_onset_ms: float | None,
) -> float:
    """Use word onset as the right edge when a synthetic pause ended too early."""
    if refined_onset_ms is None or refined_onset_ms <= pause_center_ms:
        return pause_center_ms
    return 0.5 * (pause_start_ms + refined_onset_ms)


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


def _ayah_waqf_counts(words: list) -> dict[int, int]:
    """Count explicit stop signs per ayah in the bundled mushaf text."""
    counts: dict[int, int] = {}
    for w in words:
        ayah = int(getattr(w, "ayah", 0) or 0)
        counts[ayah] = counts.get(ayah, 0) + sum(
            1 for ch in str(getattr(w, "text", "")) if ch in _WAQF_STOP_SIGNS
        )
    return counts


def _ambiguous_openings(
    openings: list[str],
    *,
    lookahead: int = _MATCH_LOOKAHEAD,
    prefix_chars: int = _AMBIG_PREFIX_CHARS,
) -> set[int]:
    """Indices whose opening text shares an actual prefix with a nearby ayah."""
    shared = _opening_shared_prefix_chars(
        openings,
        lookahead=lookahead,
        prefix_chars=prefix_chars,
    )
    return {i for i, n in shared.items() if n >= prefix_chars}


def _opening_shared_prefix_chars(
    openings: list[str],
    *,
    lookahead: int = _MATCH_LOOKAHEAD,
    prefix_chars: int = _AMBIG_PREFIX_CHARS,
) -> dict[int, int]:
    """Max collapsed prefix shared with nearby ayah openings."""
    shared = {i: 0 for i in range(len(openings))}
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
            shared[i] = max(shared[i], common)
            shared[k] = max(shared[k], common)
    return shared


def _opening_internal_repeats(
    openings: list[str],
    bodies: list[str],
    *,
    prefix_chars: int = _AMBIG_PREFIX_CHARS,
    max_chars: int = _SIM_TEXT_MAX_CHARS,
) -> dict[int, tuple[int, str]]:
    """Find target openings repeated inside the matching previous ayah body.

    ``openings[i]`` is the real opening of the ayah whose boundary is being
    placed. ``bodies[i]`` is the text before that boundary, usually the previous
    ayah. A returned suffix means an internal pause in that body may sound like
    the target ayah's start and needs a longer/read stricter comparison.
    """
    repeats: dict[int, tuple[int, str]] = {}
    for i, opening in enumerate(openings):
        if i >= len(bodies):
            break
        target = _collapse(opening)
        body = _collapse(bodies[i])
        if not target or len(body) <= prefix_chars:
            continue
        best_common = 0
        best_suffix = ""
        # Offset 0 is the previous ayah's own start; inter-ayah repeated starts
        # are handled by _opening_shared_prefix_chars. Here only internal repeats
        # can create a false boundary inside the previous ayah.
        for pos in range(1, len(body)):
            common = 0
            for a, b in zip(target, body[pos:]):
                if a != b:
                    break
                common += 1
            if common > best_common:
                best_common = common
                best_suffix = body[pos:pos + max_chars]
        if best_common >= prefix_chars:
            repeats[i] = (best_common, best_suffix)
    return repeats


def _sequential_pause_match(
    exp: list[float],
    ons: list[float],
    sim: Callable[[int, int], float],
    *,
    min_sim: float,
    prior_window_ms: float,
    confirm_min_sim: float = _CONFIRM_MATCH_SIM,
    recover_min_sim: float | None = None,
    lookahead: int = _MATCH_LOOKAHEAD,
    future_margin: float = _FUTURE_MATCH_MARGIN,
    max_candidates: int = _MATCH_MAX_SKIP,
    prior_grace_ms: float = _MATCH_PRIOR_GRACE_MS,
    reject: Callable[[int, int, float], str | None] | None = None,
    score_bonus: Callable[[int, int], float] | None = None,
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
    deferred_future: dict[int, int] = {}
    if recover_min_sim is None:
        recover_min_sim = min_sim
    observed_cache: dict[int, tuple[int, float]] = {}
    observed_local_cache: dict[tuple[int, int], tuple[int, float]] = {}

    def reject_reason(i: int, j: int, s: float) -> str | None:
        if s < min_sim:
            return "low_sim"
        return reject(i, j, s) if reject is not None else None

    def bonus(i: int, j: int) -> float:
        return score_bonus(i, j) if score_bonus is not None else 0.0

    def expected_ms(i: int) -> float:
        for prev_i in range(i - 1, -1, -1):
            if matched[prev_i] is not None:
                return matched[prev_i] + (exp[i] - exp[prev_i])
        return exp[i]

    def score(i: int, j: int) -> float:
        dist_pen = min(1.0, abs(ons[j] - expected_ms(i)) / max(1.0, prior_window_ms))
        return sim(i, j) - 0.18 * dist_pen + bonus(i, j)

    def observe_pause(j: int) -> tuple[int, float]:
        cached = observed_cache.get(j)
        if cached is not None:
            return cached
        best_i, best_s, best_score = 0, sim(0, j), -1e9
        for i in range(n):
            s = sim(i, j)
            reason = reject_reason(i, j, s)
            if reason is not None:
                continue
            sc = score(i, j)
            if sc > best_score:
                best_i, best_s, best_score = i, s, sc
        if best_score == -1e9:
            best_i, best_s = max(((i, sim(i, j)) for i in range(n)), key=lambda x: x[1])
        observed_cache[j] = (best_i, best_s)
        return best_i, best_s

    def observe_pause_local(i0: int, j: int) -> tuple[int, float]:
        key = (i0, j)
        cached = observed_local_cache.get(key)
        if cached is not None:
            return cached
        hi = min(n - 1, i0 + max(0, lookahead))
        best_i, best_s, best_score = i0, sim(i0, j), -1e9
        for i in range(i0, hi + 1):
            s = sim(i, j)
            reason = reject_reason(i, j, s)
            if reason is not None:
                continue
            sc = score(i, j)
            if sc > best_score:
                best_i, best_s, best_score = i, s, sc
        if best_score == -1e9:
            best_i, best_s = max(((i, sim(i, j)) for i in range(i0, hi + 1)), key=lambda x: x[1])
        observed_local_cache[key] = (best_i, best_s)
        return best_i, best_s

    def label_pause(i0: int, j: int) -> tuple[int, float]:
        current_s = sim(i0, j)
        local_future_margin = future_margin
        if min_sim <= current_s < recover_min_sim:
            local_future_margin = min(local_future_margin, 0.06)
        global_i, global_s = observe_pause(j)
        if global_i < i0 and global_s >= current_s + future_margin:
            return global_i, global_s
        if min_sim <= current_s < recover_min_sim and i0 + 1 < n:
            next_s = sim(i0 + 1, j)
            if next_s >= current_s + 0.05:
                return i0 + 1, next_s
        observed_i, observed_s = observe_pause_local(i0, j)
        if (
            observed_i > i0 + 1
            and current_s >= min_sim
            and observed_s <= current_s + (2.0 * local_future_margin)
            and abs(ons[j] - expected_ms(i0)) <= min(3000.0, 0.25 * prior_window_ms)
        ):
            return i0, current_s
        if (
            observed_i > i0
            and observed_s >= current_s + local_future_margin
            and ons[j] >= expected_ms(observed_i) - 0.5 * prior_window_ms
        ):
            return observed_i, observed_s
        return i0, current_s

    def observation_fields(j: int, i0: int | None = None) -> dict[str, float | int]:
        if i0 is None:
            observed_i, observed_s = observe_pause(j)
        else:
            observed_i, observed_s = observe_pause_local(i0, j)
        global_i, global_s = observe_pause(j)
        fields = {
            "observed_boundary": observed_i,
            "observed_sim": round(observed_s, 3),
        }
        if global_i != observed_i or abs(global_s - observed_s) > 1e-9:
            fields["observed_global_boundary"] = global_i
            fields["observed_global_sim"] = round(global_s, 3)
        return fields

    def accept(i: int, j: int, *, action: str) -> None:
        s = sim(i, j)
        matched[i] = ons[j]
        events.append({
            "boundary": i,
            "pause": j,
            "action": action,
            "sim": round(s, 3),
            "tempo_score": round(bonus(i, j), 3),
            "expected_ms": round(expected_ms(i), 1),
            "pause_ms": round(ons[j], 1),
            "expected_delta_ms": round(ons[j] - expected_ms(i), 1),
            **observation_fields(j, i),
        })

    def choose_current(i: int, js: list[int]) -> int:
        best_s = max(sim(i, jj) for jj in js)
        near_best = [jj for jj in js if sim(i, jj) >= best_s - 0.05]
        return min(near_best)

    def current_with_recover_pool(i: int, js: list[int], pool: list[int]) -> list[int]:
        if not js:
            return js
        best_s = max(sim(i, jj) for jj in js)
        early_js = [
            jj for jj in pool
            if matched.count(ons[jj]) == 0
            and observe_pause_local(i, jj)[0] == i
            and (s := sim(i, jj)) >= best_s - _CONFIRMED_EARLY_CURRENT_SLACK
            and not far_weak_current(i, jj, s)
        ]
        return sorted(dict.fromkeys(early_js + js))

    def pause_order_current_before_late_confirmed(i: int, j_confirmed: int, skipped: list[int]) -> int | None:
        """Prefer the local pause-order candidate over a far repeated opening.

        In repeated openings a later ayah may decode with a stronger textual
        similarity than the real current pause.  If the strong match is far from
        the tempo-predicted current boundary, choose an earlier skipped pause
        that is near the expected position and locally labels as the current
        boundary.
        """
        confirmed_dist = ons[j_confirmed] - expected_ms(i)
        if confirmed_dist <= _FAR_WEAK_CURRENT_GRACE_MS:
            return None
        floor = max(0.0, min_sim - _RECOVER_ECHO_SIM_SLACK - 0.02)
        near_limit = min(
            12_000.0,
            max(3500.0, 0.30 * confirmed_dist, 0.25 * prior_window_ms),
        )
        choices: list[tuple[float, int]] = []
        for jj in sorted(dict.fromkeys(skipped)):
            if jj >= j_confirmed or matched.count(ons[jj]) > 0:
                continue
            s = sim(i, jj)
            if s < floor:
                continue
            observed_i, observed_s = observe_pause_local(i, jj)
            if observed_i != i:
                continue
            dist = abs(ons[jj] - expected_ms(i))
            if dist > near_limit:
                continue
            hard_reason = reject(i, jj, s) if reject is not None else None
            if hard_reason is not None and hard_reason != "future_better":
                continue
            dist_pen = min(1.0, dist / max(1.0, near_limit))
            sc = max(s, observed_s) + max(0.0, bonus(i, jj)) - 0.05 * dist_pen
            choices.append((sc, jj))
        if not choices:
            return None
        return max(choices, key=lambda item: (item[0], -abs(ons[item[1]] - expected_ms(i))))[1]

    def far_weak_current(i: int, j: int, s: float) -> bool:
        dist = abs(ons[j] - expected_ms(i))
        return (
            (
                dist > _FAR_WEAK_CURRENT_GRACE_MS
                and s < _FAR_WEAK_CURRENT_SIM
            )
            or (
                dist > _FAR_CURRENT_MAX_OVERRIDE_MS
                and s < confirm_min_sim
            )
        )

    def recover_one(i: int, js: list[int]) -> int | None:
        recover_js = sorted(dict.fromkeys(js))
        best_j, best_s, best_score = -1, 0.0, -1e9
        early_observed: list[int] = []
        for j in recover_js:
            if matched.count(ons[j]) > 0:
                continue
            observed_i, observed_s = observe_pause_local(i, j)
            global_i, global_s = observe_pause(j)
            s = sim(i, j)
            effective_s = observed_s if observed_i == i else s
            current_dist = abs(ons[j] - expected_ms(i))
            future_dist = abs(ons[j] - expected_ms(global_i)) if 0 <= global_i < n else prior_window_ms
            near_current_observed = (
                observed_i == i
                and effective_s >= max(0.0, min_sim - _RECOVER_ECHO_SIM_SLACK)
                and current_dist <= min(3000.0, 0.25 * prior_window_ms)
            )
            if near_current_observed:
                early_observed.append(j)
            near_current_prior = (
                s >= max(0.0, min_sim - _RECOVER_ECHO_SIM_SLACK)
                and current_dist <= 1500.0
            )
            near_current_future_echo = (
                global_i > i
                and s >= max(0.0, min_sim - _RECOVER_ECHO_SIM_SLACK)
                and global_s <= s + (2.0 * future_margin)
                and current_dist <= min(3000.0, 0.25 * prior_window_ms)
                and current_dist + 1500.0 < future_dist
            )
            if (
                not near_current_observed
                and not near_current_prior
                and not near_current_future_echo
                and (
                    effective_s + max(0.0, bonus(i, j)) < recover_min_sim
                    or (
                        current_dist > _FAR_WEAK_CURRENT_GRACE_MS
                        and effective_s < _FAR_WEAK_CURRENT_SIM
                    )
                )
            ):
                events.append({
                    "boundary": i,
                    "pause": j,
                    "action": "skip_recovery_low_sim",
                    "sim": round(s, 3),
                    "effective_sim": round(effective_s, 3),
                    "pause_ms": round(ons[j], 1),
                    **observation_fields(j, i),
                })
                continue
            if (
                global_i != i
                and global_s > effective_s
                and (observed_i != i or global_i > i)
                and not near_current_future_echo
            ):
                events.append({
                    "boundary": i,
                    "pause": j,
                    "action": "skip_recovery_observed_mismatch",
                    "sim": round(s, 3),
                    "pause_ms": round(ons[j], 1),
                    **observation_fields(j, i),
                })
                continue
            hard_reason = reject(i, j, s) if reject is not None else None
            if (
                hard_reason is not None
                and hard_reason != "future_better"
                and observed_i != i
                and not near_current_prior
                and not near_current_future_echo
            ):
                events.append({
                    "boundary": i,
                    "pause": j,
                    "action": "skip_recovery_reject",
                    "reason": hard_reason,
                    "sim": round(s, 3),
                    "pause_ms": round(ons[j], 1),
                    **observation_fields(j, i),
                })
                continue
            dist_pen = min(1.0, abs(ons[j] - expected_ms(i)) / max(1.0, prior_window_ms))
            sc = effective_s - 0.03 * dist_pen + min(0.03, max(0.0, bonus(i, j)))
            if observed_i != i:
                sc -= future_margin
            if sc > best_score:
                best_j, best_s, best_score = j, effective_s, sc
        if best_j >= 0:
            early_observed = [
                j for j in early_observed
                if ons[j] + 1500.0 < ons[best_j]
                and abs(ons[j] - expected_ms(i)) <= abs(ons[best_j] - expected_ms(i)) + 500.0
            ]
            if early_observed:
                best_j = min(early_observed)
                best_s = sim(i, best_j)
            accept(i, best_j, action="recovered")
            return best_j
        events.append({
            "boundary": i,
            "action": "wasl",
            "reason": "missing_before_future",
            "best_sim": round(best_s, 3),
            "expected_ms": round(expected_ms(i), 1),
        })
        return None

    while k < n:
        upper_i = min(n - 1, k + max(0, lookahead))
        lower = expected_ms(k) - prior_window_ms - prior_grace_ms
        upper = expected_ms(upper_i) + prior_window_ms + prior_grace_ms
        while j_cursor < len(ons) and ons[j_cursor] < lower:
            events.append({
                "boundary": k,
                "pause": j_cursor,
                "action": "early_pool",
                "reason": "before_expected_window",
                "pause_ms": round(ons[j_cursor], 1),
                **observation_fields(j_cursor, k),
            })
            recover_pool.append(j_cursor)
            j_cursor += 1

        candidates: list[int] = []
        j = j_cursor
        while j < len(ons) and ons[j] <= upper and len(candidates) < max_candidates:
            candidates.append(j)
            j += 1
        future_j = deferred_future.get(k)
        if future_j is not None and future_j >= j_cursor and future_j not in candidates:
            candidates.append(future_j)
            candidates.sort()
        if not candidates:
            events.append({
                "boundary": k,
                "action": "wasl",
                "reason": "no_pause_in_window",
                "expected_ms": round(expected_ms(k), 1),
            })
            k += 1
            continue

        pending: list[int] = []
        current_candidates: list[int] = []
        stale_candidates: list[int] = []
        advanced = False
        for j in candidates:
            heard_i, heard_s = label_pause(k, j)
            reason = reject_reason(heard_i, j, heard_s)
            local_i, _local_s = observe_pause_local(k, j)
            current_reason_ok = reason is None or (
                reason == "far_from_expected"
                and local_i == k
                and heard_s >= min_sim
            )
            found_current = (
                heard_i == k
                and current_reason_ok
                and heard_s >= min_sim
                and not far_weak_current(k, j, heard_s)
                and (
                    local_i == k
                    or abs(ons[j] - expected_ms(k)) <= min(3000.0, 0.25 * prior_window_ms)
                )
            )
            if heard_i == k and reason is None and heard_s >= confirm_min_sim:
                order_j = pause_order_current_before_late_confirmed(
                    k,
                    j,
                    recover_pool + current_candidates + pending,
                )
                if order_j is not None:
                    for extra_j in recover_pool + [
                        jj for jj in pending + [j] if jj != order_j
                    ]:
                        events.append({
                            "boundary": k,
                            "pause": extra_j,
                            "action": "extra",
                            "reason": "late_confirmed_after_pause_order_current",
                            "sim": round(sim(k, extra_j), 3),
                            "pause_ms": round(ons[extra_j], 1),
                            **observation_fields(extra_j, k),
                        })
                    accept(k, order_j, action="observed_current")
                    j_cursor = order_j + 1
                    recover_pool = []
                    k += 1
                    advanced = True
                    break
                early_current_js = [
                    jj for jj in sorted(dict.fromkeys(current_candidates + pending))
                    if observe_pause_local(k, jj)[0] == k
                    and sim(k, jj) >= heard_s - _CONFIRMED_EARLY_CURRENT_SLACK
                ]
                if not early_current_js and heard_s < _PREFIX_BYPASS_SIM:
                    early_current_js = [
                        jj for jj in sorted(dict.fromkeys(recover_pool + current_candidates + pending))
                        if matched.count(ons[jj]) == 0
                        and ons[jj] < ons[j]
                        and observe_pause_local(k, jj)[0] == k
                        and sim(k, jj) >= min_sim
                        and not far_weak_current(k, jj, sim(k, jj))
                    ]
                if early_current_js:
                    best_current_j = min(early_current_js)
                    for extra_j in recover_pool + [
                        jj for jj in pending + [j] if jj != best_current_j
                    ]:
                        events.append({
                            "boundary": k,
                            "pause": extra_j,
                            "action": "extra",
                            "reason": "low_or_internal_before_observed_current",
                            "sim": round(sim(k, extra_j), 3),
                            "pause_ms": round(ons[extra_j], 1),
                            **observation_fields(extra_j, k),
                        })
                    accept(k, best_current_j, action="observed_current")
                    j_cursor = best_current_j + 1
                    recover_pool = []
                    k += 1
                    advanced = True
                    break
                for extra_j in recover_pool + pending:
                    events.append({
                        "boundary": k,
                        "pause": extra_j,
                        "action": "extra",
                        "reason": "low_or_internal_before_confirmed",
                        "sim": round(sim(k, extra_j), 3),
                        "pause_ms": round(ons[extra_j], 1),
                        **observation_fields(extra_j, k),
                    })
                accept(k, j, action="confirmed")
                j_cursor = j + 1
                recover_pool = []
                k += 1
                advanced = True
                break
            if found_current:
                events.append({
                    "boundary": k,
                    "pause": j,
                    "action": "observed_current_candidate",
                    "sim": round(heard_s, 3),
                    "min_confirm_sim": round(confirm_min_sim, 3),
                    "pause_ms": round(ons[j], 1),
                    **observation_fields(j, k),
                })
                current_candidates.append(j)
                pending.append(j)
                continue
            if heard_i == k and reason == "low_sim":
                weak_floor = max(0.0, min_sim - _RECOVER_ECHO_SIM_SLACK - 0.02)
                if (
                    heard_s >= weak_floor
                    and observe_pause_local(k, j)[0] == k
                    and abs(ons[j] - expected_ms(k))
                    <= min(3500.0, max(1800.0, 0.25 * prior_window_ms))
                ):
                    events.append({
                        "boundary": k,
                        "pause": j,
                        "action": "pause_order_current_candidate",
                        "sim": round(heard_s, 3),
                        "min_confirm_sim": round(confirm_min_sim, 3),
                        "pause_ms": round(ons[j], 1),
                        **observation_fields(j, k),
                    })
                    pending.append(j)
                    continue
            if heard_i == k and reason is None:
                events.append({
                    "boundary": k,
                    "pause": j,
                    "action": "weak_current",
                    "sim": round(heard_s, 3),
                    "min_confirm_sim": round(confirm_min_sim, 3),
                    "pause_ms": round(ons[j], 1),
                    **observation_fields(j, k),
                })
                continue
            if heard_i < k and heard_s >= min_sim:
                local_i, _local_s = observe_pause_local(k, j)
                current_s = sim(k, j)
                if (
                    local_i == k
                    and current_s >= max(0.0, min_sim - _RECOVER_ECHO_SIM_SLACK)
                    and abs(ons[j] - expected_ms(k)) <= min(3000.0, 0.25 * prior_window_ms)
                ):
                    events.append({
                        "boundary": k,
                        "pause": j,
                        "action": "observed_current_candidate",
                        "reason": "local_current_over_global_stale",
                        "sim": round(current_s, 3),
                        "min_confirm_sim": round(confirm_min_sim, 3),
                        "pause_ms": round(ons[j], 1),
                        **observation_fields(j, k),
                    })
                    current_candidates.append(j)
                    pending.append(j)
                    continue
                stale_candidates.append(j)
                if current_candidates:
                    current_choices = current_with_recover_pool(k, current_candidates, recover_pool)
                    best_current_j = choose_current(k, current_choices)
                    for extra_j in recover_pool + [jj for jj in pending if jj != best_current_j]:
                        events.append({
                            "boundary": k,
                            "pause": extra_j,
                            "action": "extra",
                            "reason": "low_or_internal_before_stale",
                            "sim": round(sim(k, extra_j), 3),
                            "pause_ms": round(ons[extra_j], 1),
                            **observation_fields(extra_j, k),
                        })
                    events.append({
                        "boundary": k,
                        "pause": j,
                        "action": "forward_after_stale",
                        "chosen_pause": best_current_j,
                        "sim": round(sim(k, best_current_j), 3),
                        "stale_sim": round(heard_s, 3),
                        "pause_ms": round(ons[j], 1),
                        **observation_fields(j, k),
                    })
                    accept(k, best_current_j, action="observed_current")
                    j_cursor = best_current_j + 1
                    recover_pool = []
                    k += 1
                    advanced = True
                    break
                events.append({
                    "boundary": k,
                    "pause": j,
                    "action": "stale_observation",
                    "sim": round(sim(k, j), 3),
                    "pause_ms": round(ons[j], 1),
                    **observation_fields(j, k),
                })
                continue
            if heard_i > k and reason is None:
                deferred_future[heard_i] = j
                events.append({
                    "boundary": k,
                    "pause": j,
                    "action": "future_gap",
                    "heard_boundary": heard_i,
                    "heard_sim": round(heard_s, 3),
                    "current_sim": round(sim(k, j), 3),
                    "tempo_score": round(bonus(heard_i, j), 3),
                    "pause_ms": round(ons[j], 1),
                    **observation_fields(j, k),
                })
                events.append({
                    "boundary": k,
                    "pause": j,
                    "action": "backtrack_before_future",
                    "heard_boundary": heard_i,
                    "heard_sim": round(heard_s, 3),
                    "current_sim": round(sim(k, j), 3),
                    "pending_pauses": list(recover_pool + pending),
                    "pause_ms": round(ons[j], 1),
                })
                recover_js = recover_pool + pending
                recovered_j = None
                recovered_until = k - 1
                for missing_i in range(k, heard_i):
                    next_j = recover_one(missing_i, recover_js)
                    if next_j is None:
                        break
                    recovered_j = next_j
                    recovered_until = missing_i
                    recover_js = [jj for jj in recover_js if jj > next_j]
                j_cursor = recovered_j + 1 if recovered_j is not None else j_cursor
                recover_pool = []
                k = recovered_until + 1 if recovered_until >= k else k + 1
                advanced = True
                break
            pending.append(j)

        if not advanced:
            if current_candidates:
                current_choices = current_with_recover_pool(k, current_candidates, recover_pool)
                best_current_j = choose_current(k, current_choices)
                for extra_j in recover_pool + [jj for jj in pending if jj < best_current_j]:
                    events.append({
                        "boundary": k,
                        "pause": extra_j,
                        "action": "extra",
                        "reason": "low_or_internal_before_observed_current",
                        "sim": round(sim(k, extra_j), 3),
                        "pause_ms": round(ons[extra_j], 1),
                        **observation_fields(extra_j, k),
                    })
                accept(k, best_current_j, action="observed_current")
                j_cursor = best_current_j + 1
                recover_pool = []
                k += 1
                continue
            best_s = max((sim(k, j) for j in candidates), default=0.0)
            best_obs_j = max(
                candidates,
                key=lambda jj: observe_pause_local(k, jj)[1],
                default=candidates[0],
            )
            best_obs_i, best_obs_s = observe_pause_local(k, best_obs_j)
            weak_current_js = [
                jj for jj in candidates
                if observe_pause_local(k, jj)[0] == k
                and sim(k, jj) >= max(0.0, min_sim - _RECOVER_ECHO_SIM_SLACK)
                and abs(ons[jj] - expected_ms(k)) <= min(3000.0, 0.25 * prior_window_ms)
            ]
            if weak_current_js:
                best_weak_j = choose_current(k, weak_current_js)
                for extra_j in recover_pool + [jj for jj in pending if jj != best_weak_j]:
                    events.append({
                        "boundary": k,
                        "pause": extra_j,
                        "action": "extra",
                        "reason": "low_or_internal_before_weak_observed_current",
                        "sim": round(sim(k, extra_j), 3),
                        "pause_ms": round(ons[extra_j], 1),
                        **observation_fields(extra_j, k),
                    })
                accept(k, best_weak_j, action="weak_observed_current")
                j_cursor = best_weak_j + 1
                recover_pool = []
                k += 1
                continue
            if (
                k == n - 1
                and best_obs_i == k
                and best_obs_s >= _FINAL_WEAK_MATCH_SIM
                and bonus(k, best_obs_j) >= _FINAL_WEAK_TEMPO_FLOOR
            ):
                for extra_j in recover_pool + [jj for jj in pending if jj != best_obs_j]:
                    events.append({
                        "boundary": k,
                        "pause": extra_j,
                        "action": "extra",
                        "reason": "low_or_internal_before_final_weak",
                        "sim": round(sim(k, extra_j), 3),
                        "pause_ms": round(ons[extra_j], 1),
                        **observation_fields(extra_j, k),
                    })
                accept(k, best_obs_j, action="final_weak_current")
                j_cursor = best_obs_j + 1
                recover_pool = []
                k += 1
                continue
            if best_obs_i < k and stale_candidates:
                for extra_j in pending:
                    events.append({
                        "boundary": k,
                        "pause": extra_j,
                        "action": "extra",
                        "reason": "no_confirmed_match_before_stale",
                        "sim": round(sim(k, extra_j), 3),
                        "pause_ms": round(ons[extra_j], 1),
                        **observation_fields(extra_j, k),
                    })
                    recover_pool.append(extra_j)
                events.append({
                    "boundary": k,
                    "pause": max(stale_candidates),
                    "action": "forward_search_after_stale",
                    "reason": "observed_past",
                    "observed_boundary": best_obs_i,
                    "observed_sim": round(best_obs_s, 3),
                    "expected_ms": round(expected_ms(k), 1),
                    "pause_ms": round(ons[max(stale_candidates)], 1),
                })
                j_cursor = max(candidates) + 1
                continue
            if (
                best_obs_i > k
                and best_obs_i == k + 1
                and best_obs_i == n - 1
                and best_obs_s >= _FINAL_WEAK_MATCH_SIM
                and bonus(best_obs_i, best_obs_j) > 0.0
            ):
                events.append({
                    "boundary": k,
                    "pause": best_obs_j,
                    "action": "weak_future_gap",
                    "heard_boundary": best_obs_i,
                    "heard_sim": round(best_obs_s, 3),
                    "current_sim": round(sim(k, best_obs_j), 3),
                    "pause_ms": round(ons[best_obs_j], 1),
                    **observation_fields(best_obs_j, k),
                })
                recovered_j = recover_one(k, recover_pool + pending)
                j_cursor = recovered_j + 1 if recovered_j is not None else j_cursor
                recover_pool = []
                k += 1
                continue
            if best_obs_i > k and best_obs_s >= min_sim:
                deferred_future[best_obs_i] = best_obs_j
                events.append({
                    "boundary": k,
                    "pause": best_obs_j,
                    "action": "future_gap",
                    "heard_boundary": best_obs_i,
                    "heard_sim": round(best_obs_s, 3),
                    "current_sim": round(sim(k, best_obs_j), 3),
                    "pause_ms": round(ons[best_obs_j], 1),
                    **observation_fields(best_obs_j, k),
                })
                recover_js = recover_pool + pending
                recovered_j = None
                recovered_until = k - 1
                for missing_i in range(k, best_obs_i):
                    next_j = recover_one(missing_i, recover_js)
                    if next_j is None:
                        break
                    recovered_j = next_j
                    recovered_until = missing_i
                    recover_js = [jj for jj in recover_js if jj > next_j]
                j_cursor = recovered_j + 1 if recovered_j is not None else j_cursor
                recover_pool = []
                k = recovered_until + 1 if recovered_until >= k else k + 1
                continue
            for extra_j in pending:
                events.append({
                    "boundary": k,
                    "pause": extra_j,
                    "action": "extra",
                    "reason": "no_confirmed_match",
                    "sim": round(sim(k, extra_j), 3),
                    "pause_ms": round(ons[extra_j], 1),
                    **observation_fields(extra_j, k),
                })
                recover_pool.append(extra_j)
            events.append({
                "boundary": k,
                "pause": best_obs_j,
                "action": "unresolved_current",
                "reason": "no_confirmed_match",
                "best_sim": round(best_s, 3),
                "observed_boundary": best_obs_i,
                "observed_sim": round(best_obs_s, 3),
                "direction": (
                    "future" if best_obs_i > k
                    else "past" if best_obs_i < k
                    else "unknown"
                ),
                "expected_ms": round(expected_ms(k), 1),
                "pause_ms": round(ons[best_obs_j], 1),
            })
            events.append({
                "boundary": k,
                "action": "wasl",
                "reason": "unresolved_current",
                "best_sim": round(best_s, 3),
                "expected_ms": round(expected_ms(k), 1),
            })
            j_cursor = max(candidates) + 1
            k += 1
            continue

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
    pause_diag = {d.pause: d for d in pause_decisions}

    def pause_tier(d: candidates.PauseDecision) -> str:
        if not d.accepted:
            return "rejected"
        return "weak" if d.reason == "weak-dsp" else "trusted"

    trusted_decisions = [d for d in pause_decisions if pause_tier(d) == "trusted"]
    low_conf_decisions = [d for d in pause_decisions if pause_tier(d) != "trusted"]
    pauses = [d.pause for d in trusted_decisions]
    pause_onsets = sorted(p.center_ms for p in pauses)
    all_pause_ord = {
        d.pause.center_ms: i + 1
        for i, d in enumerate(sorted(pause_decisions, key=lambda item: item.pause.center_ms))
    }
    active_low_conf_centers: set[float] = set()
    low_conf_activation_debug: list[dict] = []
    ayah_boundary_count = max(0, len(ayah_list) - 1)
    waqf_mark_count = _count_waqf_marks(words)
    ayah_waqf_count = _ayah_waqf_counts(words)
    expected_pause_targets = ayah_boundary_count + waqf_mark_count
    accepted_pause_count = len([d for d in pause_decisions if d.accepted])
    rejected_pause_count = len(pause_decisions) - accepted_pause_count
    trusted_pause_count = len(trusted_decisions)
    weak_pause_count = len([d for d in pause_decisions if pause_tier(d) == "weak"])
    emit(
        f"Сура {surah}: старт fast-разметки, длительность {sec(total_ms)} сек., "
        f"найдено pause-кандидатов: {len(pause_decisions)}, "
        f"trusted: {trusted_pause_count}, weak: {weak_pause_count}, "
        f"rejected: {rejected_pause_count}"
    )
    emit(
        f"Бюджет пауз: границ аятов {ayah_boundary_count}, "
        f"waqf-знаков {waqf_mark_count}, ожидаемых целей ~{expected_pause_targets}; "
        f"первый проход по trusted {trusted_pause_count}, "
        f"резерв weak/rejected {len(low_conf_decisions)}, "
        f"отношение {accepted_pause_count / max(1, expected_pause_targets):.2f}"
    )
    for d in sorted(pause_decisions, key=lambda item: item.pause.center_ms):
        p = d.pause
        dsp = (
            f", tier={pause_tier(d)}, reason={d.reason}, rms={d.rms_ratio:.3f}, "
            f"flat={d.flatness:.3f}, zcr={d.zcr:.3f}, vad={d.vad_ok}"
        )
        emit(
            f"Пауза {all_pause_ord.get(p.center_ms, '?')}: "
            f"{sec(p.start_ms)}-{sec(p.end_ms)} сек., "
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
    raw_content_start = content_start
    intro_pause = next(
        (
            d.pause for d in trusted_decisions
            if d.reason == "intro-vad-tight"
            and d.pause.start_ms <= content_start <= d.pause.end_ms
        ),
        None,
    )
    if intro_pause is not None:
        content_start = intro_pause.center_ms
        emit(
            f"Первый аят: anchor {sec(raw_content_start)} сек. попал внутрь intro-паузы "
            f"{sec(intro_pause.start_ms)}-{sec(intro_pause.end_ms)} сек.; "
            f"ставлю старт {sec(content_start)} сек."
        )
    if basmala and content_start < _BASMALA_MIN_CONTENT_START_MS:
        emit(
            f"Первый аят: anchor {sec(content_start)} сек. раньше нижней границы "
            f"басмалы {sec(_BASMALA_MIN_CONTENT_START_MS)} сек.; сдвигаю старт"
        )
        content_start = _BASMALA_MIN_CONTENT_START_MS
    model_reads.append({
        "phase": "anchor",
        "ayah": a1,
        "window_ms": [round(res["base"], 1), round(res["end"], 1)],
        "words": " ".join(w.text for w in words[0:head_end]),
        "target": words[fa1].text,
        "onset_ms": round(content_start, 1),
        "raw_onset_ms": round(raw_content_start, 1),
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
    tempo_ms_per_unit = content_dur / total_w

    def expected_ayah_ms(a: int) -> float:
        return wunit.get(a, 1.0) * tempo_ms_per_unit

    def prev_speech_ms(j: int) -> float:
        pause = pause_by_center.get(pause_onsets[j])
        if pause is None:
            return 0.0
        if j > 0:
            prev_pause = pause_by_center.get(pause_onsets[j - 1])
            start = prev_pause.end_ms if prev_pause is not None else content_start
        else:
            start = content_start
        return max(0.0, pause.start_ms - start)

    def next_speech_ms(j: int) -> float:
        pause = pause_by_center.get(pause_onsets[j])
        if pause is None:
            return 0.0
        if j + 1 < len(pause_onsets):
            next_pause = pause_by_center.get(pause_onsets[j + 1])
            end = next_pause.start_ms if next_pause is not None else total_ms
        else:
            end = total_ms
        return max(0.0, end - pause.end_ms)

    def wave_tempo_metrics(i: int, j: int) -> dict[str, float | int | bool]:
        """Soft structural prior for boundary i at pause j.

        The previous speech span should often fit the previous ayah, and the next
        span should often fit the ayah that starts at this pause. Waqf signs make
        short spans less suspicious because they may be internal stops.
        """
        prev_a = ayah_list[i]
        cur_a = inner[i]
        prev_ms = prev_speech_ms(j)
        next_ms = next_speech_ms(j)
        prev_expected = expected_ayah_ms(prev_a)
        next_expected = expected_ayah_ms(cur_a)
        prev_quality = _span_ratio_quality(prev_ms, prev_expected)
        next_quality = _span_ratio_quality(next_ms, next_expected)
        waqf_before = ayah_waqf_count.get(prev_a, 0)
        waqf_after = ayah_waqf_count.get(cur_a, 0)
        # One side can be fragmented by an internal waqf pause. Use the better
        # side as the main evidence and reduce penalty when waqf is expected.
        quality = max(prev_quality, next_quality)
        if quality < 0.5 and (waqf_before or waqf_after):
            quality = max(quality, 0.45)
        score = (quality - 0.5) * 2.0 * _TEMPO_SCORE_WEIGHT
        return {
            "tempo_score": score,
            "prev_speech_ms": prev_ms,
            "next_speech_ms": next_ms,
            "prev_expected_ms": prev_expected,
            "next_expected_ms": next_expected,
            "prev_span_ratio": prev_ms / prev_expected if prev_expected > 0 else 0.0,
            "next_span_ratio": next_ms / next_expected if next_expected > 0 else 0.0,
            "prev_quality": prev_quality,
            "next_quality": next_quality,
            "waqf_before": waqf_before,
            "waqf_after": waqf_after,
            "tempo_ms_per_unit": tempo_ms_per_unit,
        }

    tempo_score_cache: dict[tuple[int, int], float] = {}

    def tempo_score(i: int, j: int) -> float:
        key = (i, j)
        score = tempo_score_cache.get(key)
        if score is None:
            score = float(wave_tempo_metrics(i, j)["tempo_score"])
            tempo_score_cache[key] = score
        return score

    # Opening tokens per inner ayah, and a greedy decode per pause — the two
    # sides of the acoustic match. Unique openings still compare only the short
    # prefix; repeated beginnings get more text/audio until the distinguishing
    # part can be heard.
    openings = [
        "".join(word_trs[i] for i in ayah_words[a] if i < len(word_trs))
        for a in inner
    ]
    previous_bodies = [
        "".join(word_trs[i] for i in ayah_words[ayah_list[i]] if i < len(word_trs))
        for i in range(len(inner))
    ]
    opening_shared_prefix = _opening_shared_prefix_chars(openings)
    opening_internal_repeats = _opening_internal_repeats(openings, previous_bodies)
    opening_internal_prefix = {
        i: repeat[0] for i, repeat in opening_internal_repeats.items()
    }
    opening_text_risk = {
        i: max(opening_shared_prefix.get(i, 0), opening_internal_prefix.get(i, 0))
        for i in range(len(openings))
    }
    ambiguous_opening_idx = {
        i for i, n in opening_text_risk.items() if n >= _AMBIG_PREFIX_CHARS
    }
    decode_win_ms = _PAUSE_DECODE_WIN_MS
    decode_ambig_win_ms = max(decode_win_ms, _PAUSE_DECODE_AMBIG_WIN_MS)
    pause_decodes: dict[tuple[int, int], str] = {}
    pause_decode_reasons: dict[float, str] = {}
    pause_decode_starts: dict[float, float] = {}
    pause_decode_windows: dict[float, float] = {}
    micro_pause_debug: list[dict] = []

    def sim_chars_for(i: int) -> int:
        shared = opening_text_risk.get(i, 0)
        if shared < _AMBIG_PREFIX_CHARS:
            return _SIM_CHARS
        return min(_SIM_TEXT_MAX_CHARS, max(_SIM_CHARS, shared + _SIM_TEXT_MARGIN_CHARS))

    def decode_window_for(i: int) -> float:
        shared = opening_text_risk.get(i, 0)
        if shared < _AMBIG_PREFIX_CHARS:
            return decode_win_ms
        text_win = decode_win_ms + shared * _PAUSE_DECODE_TEXT_MS_PER_CHAR
        return min(_PAUSE_DECODE_TEXT_MAX_WIN_MS, max(decode_ambig_win_ms, text_win))

    def decode_pause(j: int, *, win_ms: float | None = None) -> str:
        if win_ms is None:
            win_ms = decode_win_ms
        win_key = int(round(win_ms))
        key = (j, win_key)
        heard = pause_decodes.get(key)
        if heard is not None:
            return heard
        p = pause_onsets[j]
        pause = pause_by_center.get(p)
        decode_start = (
            max(0.0, pause.end_ms - _PAUSE_DECODE_PRE_ROLL_MS)
            if pause is not None else max(0.0, p - 50.0)
        )
        decode_end = min(total_ms, decode_start + win_ms)
        edge_delta = decode_start - pause.end_ms if pause is not None else 0.0
        pause_decode_starts[p] = decode_start
        prev_win = pause_decode_windows.get(p, 0.0)
        if win_ms >= prev_win:
            pause_decode_windows[p] = win_ms
            pause_decode_reasons[p] = "text-aware-opening" if win_ms > decode_win_ms else "default"
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

    sim_cache: dict[tuple[int, int, int], float] = {}

    def needs_extended_decode(i: int, j: int) -> bool:
        return i in ambiguous_opening_idx and prior_delta(i, j) <= _AMBIG_DECODE_PRIOR_MS

    def _sim_ij(i: int, j: int) -> float:
        win_ms = decode_window_for(i) if needs_extended_decode(i, j) else decode_win_ms
        win_key = int(round(win_ms))
        key = (i, j, win_key)
        s = sim_cache.get(key)
        if s is None:
            s = _sim(
                decode_pause(j, win_ms=win_ms),
                openings[i],
                chars=sim_chars_for(i),
            )
            sim_cache[key] = s
        return s

    def prior_delta(i: int, j: int) -> float:
        return abs(pause_onsets[j] - exp[i])

    def far_weak_match(i: int, j: int, s: float) -> bool:
        return (
            prior_delta(i, j) > _MATCH_PRIOR_WINDOW_MS + _MATCH_PRIOR_GRACE_MS
            and s < _FAR_MATCH_SIM
        )

    def weak_pause(j: int) -> bool:
        pause = pause_by_center.get(pause_onsets[j])
        diag = pause_diag.get(pause) if pause is not None else None
        return bool(diag is not None and diag.reason == "weak-dsp")

    def prefix_mismatch(i: int, j: int, s: float) -> bool:
        win_ms = decode_window_for(i) if needs_extended_decode(i, j) else decode_win_ms
        heard = decode_pause(j, win_ms=win_ms)
        common = _onset_prefix_len(heard, openings[i])
        risky_text = opening_text_risk.get(i, 0) >= _AMBIG_PREFIX_CHARS
        if s >= _PREFIX_BYPASS_SIM and not risky_text:
            return False
        min_prefix = _PREFIX_WEAK_MIN if weak_pause(j) else _PREFIX_STRONG_MIN
        if risky_text:
            min_prefix = max(min_prefix, _PREFIX_WEAK_MIN)
        return common < min_prefix

    def internal_repeat_mismatch(i: int, j: int, s: float) -> bool:
        repeat = opening_internal_repeats.get(i)
        if repeat is None:
            return False
        win_ms = decode_window_for(i) if needs_extended_decode(i, j) else decode_win_ms
        heard = decode_pause(j, win_ms=win_ms)
        _repeat_chars, repeat_suffix = repeat
        repeat_s = _sim(heard, repeat_suffix, chars=sim_chars_for(i))
        target_prefix = _onset_prefix_len(heard, openings[i])
        repeat_prefix = _onset_prefix_len(heard, repeat_suffix)
        return (
            repeat_s >= s + _INTERNAL_REPEAT_MATCH_MARGIN
            and repeat_prefix >= target_prefix
        )

    def reject_match(i: int, j: int, s: float) -> str | None:
        if far_weak_match(i, j, s):
            return "far_from_expected"
        if prefix_mismatch(i, j, s):
            return "prefix_mismatch"
        if s < _RECOVER_MATCH_SIM:
            win_ms = decode_window_for(i) if needs_extended_decode(i, j) else decode_win_ms
            heard = decode_pause(j, win_ms=win_ms)
            if _onset_prefix_len(heard, openings[i]) < _PREFIX_BORDERLINE_MIN:
                return "borderline_prefix_mismatch"
        if internal_repeat_mismatch(i, j, s):
            return "internal_repeat_mismatch"
        if (
            weak_pause(j)
            and s < _PAUSE_TRUST_SIM
            and s < _TEMPO_WEAK_BYPASS_SIM
            and tempo_score(i, j) <= _TEMPO_BAD_SCORE
        ):
            return "tempo_mismatch"
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

    def run_sequence_match() -> tuple[list[float | None], list[dict]]:
        return _sequential_pause_match(
            exp,
            pause_onsets,
            _sim_ij,
            min_sim=_MIN_MATCH_SIM,
            prior_window_ms=_MATCH_PRIOR_WINDOW_MS,
            recover_min_sim=_RECOVER_MATCH_SIM,
            reject=reject_match,
            score_bonus=tempo_score,
        )

    def reset_pause_caches() -> None:
        pause_decodes.clear()
        pause_decode_reasons.clear()
        pause_decode_starts.clear()
        pause_decode_windows.clear()
        sim_cache.clear()
        tempo_score_cache.clear()

    def refresh_active_pauses() -> None:
        pauses.sort(key=lambda p: p.center_ms)
        pause_onsets[:] = sorted(
            p.center_ms for p in pauses if p.center_ms > content_start - 200.0
        )
        pause_by_center.clear()
        pause_by_center.update({p.center_ms: p for p in pauses})
        pause_ord.clear()
        pause_ord.update({p: i + 1 for i, p in enumerate(pause_onsets)})
        reset_pause_caches()

    def add_low_conf_pauses_from_gaps(
        matched0: list[float | None],
        events0: list[dict],
    ) -> bool:
        """Activate weak/rejected pause candidates only inside proven gaps."""
        added = False
        active_centers = {p.center_ms for p in pauses}
        for ev in events0:
            action = ev.get("action")
            start_i = ev.get("boundary")
            if (
                action == "wasl"
                and ev.get("reason") == "no_pause_in_window"
                and isinstance(start_i, int)
                and 0 <= start_i < len(inner)
            ):
                lo_i = start_i - 1
                lo_t = matched0[lo_i] if lo_i >= 0 and matched0[lo_i] is not None else content_start
                left_pause = pause_by_center.get(lo_t)
                interval_start = left_pause.end_ms if left_pause is not None else lo_t
                expected_t = float(ev.get("expected_ms", exp[start_i]))
                interval_end = min(total_ms, expected_t + min(3500.0, 0.25 * _MATCH_PRIOR_WINDOW_MS))
                for d in low_conf_decisions:
                    p = d.pause
                    center = p.center_ms
                    if center in active_centers or center in active_low_conf_centers:
                        continue
                    if pause_tier(d) != "weak":
                        continue
                    if center <= content_start - 200.0:
                        continue
                    if not (interval_start <= center <= interval_end):
                        continue
                    pauses.append(p)
                    active_centers.add(center)
                    active_low_conf_centers.add(center)
                    low_conf_activation_debug.append({
                        "idx": all_pause_ord.get(center),
                        "start_ms": round(p.start_ms, 1),
                        "end_ms": round(p.end_ms, 1),
                        "center_ms": round(center, 1),
                        "tier": pause_tier(d),
                        "reason": d.reason,
                        "gap_ayah": inner[start_i],
                        "heard_ayah": None,
                        "interval_ms": [round(interval_start, 1), round(interval_end, 1)],
                    })
                    emit(
                        f"Возврат к слабой паузе {all_pause_ord.get(center, '?')}: "
                        f"{sec(p.start_ms)}-{sec(p.end_ms)} сек., tier={pause_tier(d)}, "
                        f"reason={d.reason}, no-pause ayah {inner[start_i]}"
                    )
                    added = True
                continue
            if (
                action == "final_weak_current"
                and isinstance(start_i, int)
                and 0 <= start_i < len(inner)
                and abs(float(ev.get("expected_delta_ms", 0.0))) > 3000.0
                and isinstance(ev.get("pause"), int)
                and 0 <= int(ev["pause"]) < len(pause_onsets)
            ):
                lo_i = start_i - 1
                lo_t = matched0[lo_i] if lo_i >= 0 and matched0[lo_i] is not None else content_start
                hi_t = pause_onsets[int(ev["pause"])]
                left_pause = pause_by_center.get(lo_t)
                right_pause = pause_by_center.get(hi_t)
                interval_start = left_pause.end_ms if left_pause is not None else lo_t
                interval_end = right_pause.start_ms if right_pause is not None else hi_t
                expected_t = float(ev.get("expected_ms", exp[start_i]))
                interval_start = max(interval_start, expected_t - min(3500.0, 0.25 * _MATCH_PRIOR_WINDOW_MS))
                interval_end = min(interval_end, expected_t + min(3500.0, 0.25 * _MATCH_PRIOR_WINDOW_MS))
                if interval_end <= interval_start:
                    continue
                for d in low_conf_decisions:
                    p = d.pause
                    center = p.center_ms
                    if center in active_centers or center in active_low_conf_centers:
                        continue
                    if pause_tier(d) != "weak":
                        continue
                    if center <= content_start - 200.0:
                        continue
                    if not (interval_start <= center <= interval_end):
                        continue
                    pauses.append(p)
                    active_centers.add(center)
                    active_low_conf_centers.add(center)
                    low_conf_activation_debug.append({
                        "idx": all_pause_ord.get(center),
                        "start_ms": round(p.start_ms, 1),
                        "end_ms": round(p.end_ms, 1),
                        "center_ms": round(center, 1),
                        "tier": pause_tier(d),
                        "reason": d.reason,
                        "gap_ayah": inner[start_i],
                        "heard_ayah": None,
                        "interval_ms": [round(interval_start, 1), round(interval_end, 1)],
                    })
                    emit(
                        f"Возврат к слабой паузе {all_pause_ord.get(center, '?')}: "
                        f"{sec(p.start_ms)}-{sec(p.end_ms)} сек., tier={pause_tier(d)}, "
                        f"reason={d.reason}, before late final ayah {inner[start_i]}"
                    )
                    added = True
                continue
            if action not in ("future_gap", "unresolved_current"):
                continue
            future_i = ev.get("heard_boundary") if action == "future_gap" else ev.get("observed_boundary")
            future_j = ev.get("pause")
            if not (
                isinstance(start_i, int)
                and isinstance(future_i, int)
                and isinstance(future_j, int)
                and 0 <= start_i < future_i < len(inner)
                and 0 <= future_j < len(pause_onsets)
            ):
                continue
            lo_i = start_i - 1
            lo_t = matched0[lo_i] if lo_i >= 0 and matched0[lo_i] is not None else content_start
            hi_t = pause_onsets[future_j]
            if hi_t <= lo_t:
                continue
            left_pause = pause_by_center.get(lo_t)
            right_pause = pause_by_center.get(hi_t)
            interval_start = left_pause.end_ms if left_pause is not None else lo_t
            interval_end = right_pause.start_ms if right_pause is not None else hi_t
            for d in low_conf_decisions:
                p = d.pause
                center = p.center_ms
                if center in active_centers or center in active_low_conf_centers:
                    continue
                if center <= content_start - 200.0:
                    continue
                if not (interval_start <= center <= interval_end):
                    continue
                pauses.append(p)
                active_centers.add(center)
                active_low_conf_centers.add(center)
                low_conf_activation_debug.append({
                    "idx": all_pause_ord.get(center),
                    "start_ms": round(p.start_ms, 1),
                    "end_ms": round(p.end_ms, 1),
                    "center_ms": round(center, 1),
                    "tier": pause_tier(d),
                    "reason": d.reason,
                    "gap_ayah": inner[start_i],
                    "heard_ayah": inner[future_i],
                    "interval_ms": [round(interval_start, 1), round(interval_end, 1)],
                })
                emit(
                    f"Возврат к слабой паузе {all_pause_ord.get(center, '?')}: "
                    f"{sec(p.start_ms)}-{sec(p.end_ms)} сек., tier={pause_tier(d)}, "
                    f"reason={d.reason}, gap {inner[start_i]}->{inner[future_i]}"
                )
                added = True
        tail_i = len(inner) - 1
        if tail_i >= 0 and matched0[tail_i] is not None:
            tail_expected = exp[tail_i]
            tail_matched = matched0[tail_i]
            if tail_matched - tail_expected > 3000.0:
                right_pause = pause_by_center.get(tail_matched)
                interval_start = max(
                    content_start,
                    tail_expected - min(3500.0, 0.25 * _MATCH_PRIOR_WINDOW_MS),
                )
                interval_end = min(
                    right_pause.start_ms if right_pause is not None else tail_matched,
                    tail_expected + min(3500.0, 0.25 * _MATCH_PRIOR_WINDOW_MS),
                )
                for d in low_conf_decisions:
                    p = d.pause
                    center = p.center_ms
                    if center in active_centers or center in active_low_conf_centers:
                        continue
                    if pause_tier(d) != "weak":
                        continue
                    if not (interval_start <= center <= interval_end):
                        continue
                    pauses.append(p)
                    active_centers.add(center)
                    active_low_conf_centers.add(center)
                    low_conf_activation_debug.append({
                        "idx": all_pause_ord.get(center),
                        "start_ms": round(p.start_ms, 1),
                        "end_ms": round(p.end_ms, 1),
                        "center_ms": round(center, 1),
                        "tier": pause_tier(d),
                        "reason": d.reason,
                        "gap_ayah": inner[tail_i],
                        "heard_ayah": None,
                        "interval_ms": [round(interval_start, 1), round(interval_end, 1)],
                    })
                    emit(
                        f"Возврат к слабой паузе {all_pause_ord.get(center, '?')}: "
                        f"{sec(p.start_ms)}-{sec(p.end_ms)} сек., tier={pause_tier(d)}, "
                        f"reason={d.reason}, late tail ayah {inner[tail_i]}"
                    )
                    added = True
        if added:
            refresh_active_pauses()
        return added

    def add_micro_pauses_from_gaps(
        matched0: list[float | None],
        events0: list[dict],
    ) -> bool:
        added = False
        existing = list(pause_onsets)
        for ev in events0:
            action = ev.get("action")
            start_i = ev.get("boundary")
            if (
                action == "wasl"
                and ev.get("reason") == "no_pause_in_window"
                and isinstance(start_i, int)
                and start_i == len(inner) - 1
                and start_i > 0
            ):
                prev_i = start_i - 1
                while prev_i >= 0 and matched0[prev_i] is None:
                    prev_i -= 1
                lo_t = matched0[prev_i] if prev_i >= 0 and matched0[prev_i] is not None else content_start
                left_pause = pause_by_center.get(lo_t)
                interval_start = left_pause.end_ms if left_pause is not None else lo_t
                interval_end = total_ms
                if interval_end - interval_start < 4000.0:
                    continue
                expected_t = interval_start + 750.0
                valley = _find_micro_pause_valley(
                    pcm,
                    sr,
                    interval_start_ms=interval_start,
                    interval_end_ms=interval_end,
                    expected_ms=expected_t,
                )
                if valley is None:
                    micro_pause_debug.append({
                        "ayah": inner[start_i],
                        "expected_ms": round(expected_t, 1),
                        "interval_ms": [round(interval_start, 1), round(interval_end, 1)],
                        "status": "tail_not_found",
                    })
                    continue
                start_ms, end_ms, ratio, seed_start_ms, seed_end_ms = valley
                pause_start_ms = max(start_ms, seed_start_ms)
                center = 0.5 * (pause_start_ms + end_ms)
                if any(abs(center - p) < _MICRO_PAUSE_MIN_SEP_MS for p in existing):
                    micro_pause_debug.append({
                        "ayah": inner[start_i],
                        "expected_ms": round(expected_t, 1),
                        "center_ms": round(center, 1),
                        "seed_start_ms": round(seed_start_ms, 1),
                        "seed_end_ms": round(seed_end_ms, 1),
                        "expanded_start_ms": round(start_ms, 1),
                        "expanded_end_ms": round(end_ms, 1),
                        "effective_start_ms": round(pause_start_ms, 1),
                        "status": "tail_near_existing",
                    })
                    continue
                pause = candidates.Pause(pause_start_ms, end_ms, end_ms - pause_start_ms)
                pauses.append(pause)
                decision = candidates.PauseDecision(
                    pause=pause,
                    dsp_ok=True,
                    vad_ok=None,
                    accepted=True,
                    reason="micro-tempo",
                    flatness=0.0,
                    zcr=0.0,
                    rms_ratio=ratio,
                )
                pause_decisions.append(decision)
                pause_diag[pause] = decision
                existing.append(center)
                micro_pause_debug.append({
                    "ayah": inner[start_i],
                    "expected_ms": round(expected_t, 1),
                    "center_ms": round(center, 1),
                    "start_ms": round(pause_start_ms, 1),
                    "end_ms": round(end_ms, 1),
                    "seed_start_ms": round(seed_start_ms, 1),
                    "seed_end_ms": round(seed_end_ms, 1),
                    "expanded_start_ms": round(start_ms, 1),
                    "expanded_end_ms": round(end_ms, 1),
                    "effective_start_ms": round(pause_start_ms, 1),
                    "rms_ratio": round(ratio, 3),
                    "interval_ms": [round(interval_start, 1), round(interval_end, 1)],
                    "status": "tail_added",
                })
                emit(
                    f"Tail micro-пауза для аята {inner[start_i]}: "
                    f"{sec(pause_start_ms)}-{sec(end_ms)} сек., ratio={ratio:.3f}"
                )
                added = True
                continue
            if action not in ("future_gap", "unresolved_current"):
                continue
            future_i = ev.get("heard_boundary") if action == "future_gap" else ev.get("observed_boundary")
            future_j = ev.get("pause")
            if not (
                isinstance(start_i, int)
                and isinstance(future_i, int)
                and isinstance(future_j, int)
                and 0 <= start_i < future_i < len(inner)
                and 0 <= future_j < len(pause_onsets)
            ):
                continue
            lo_i = start_i - 1
            lo_a = inner[lo_i] if lo_i >= 0 else ayah_list[0]
            lo_t = matched0[lo_i] if lo_i >= 0 and matched0[lo_i] is not None else content_start
            hi_a = inner[future_i]
            hi_t = pause_onsets[future_j]
            if hi_t <= lo_t or cum_u[hi_a] <= cum_u[lo_a]:
                continue
            left_pause = pause_by_center.get(lo_t)
            right_pause = pause_by_center.get(hi_t)
            interval_start = left_pause.end_ms if left_pause is not None else lo_t
            interval_end = right_pause.start_ms if right_pause is not None else hi_t
            if interval_end - interval_start < 300.0:
                continue
            for miss_i in range(start_i, future_i):
                if matched0[miss_i] is not None:
                    continue
                miss_a = inner[miss_i]
                frac = (cum_u[miss_a] - cum_u[lo_a]) / (cum_u[hi_a] - cum_u[lo_a])
                expected_t = lo_t + (hi_t - lo_t) * min(1.0, max(0.0, frac))
                valley = _find_micro_pause_valley(
                    pcm,
                    sr,
                    interval_start_ms=interval_start,
                    interval_end_ms=interval_end,
                    expected_ms=expected_t,
                )
                if valley is None:
                    micro_pause_debug.append({
                        "ayah": miss_a,
                        "expected_ms": round(expected_t, 1),
                        "interval_ms": [round(interval_start, 1), round(interval_end, 1)],
                        "status": "not_found",
                    })
                    continue
                start_ms, end_ms, ratio, seed_start_ms, seed_end_ms = valley
                # The expanded shelf can include the quiet tail of the previous
                # phrase. For the emitted boundary, keep the right edge of the
                # shelf but start at the seed valley to avoid an early-biased
                # center.
                pause_start_ms = max(start_ms, seed_start_ms)
                center = 0.5 * (pause_start_ms + end_ms)
                if any(abs(center - p) < _MICRO_PAUSE_MIN_SEP_MS for p in existing):
                    micro_pause_debug.append({
                        "ayah": miss_a,
                        "expected_ms": round(expected_t, 1),
                        "center_ms": round(center, 1),
                        "seed_start_ms": round(seed_start_ms, 1),
                        "seed_end_ms": round(seed_end_ms, 1),
                        "expanded_start_ms": round(start_ms, 1),
                        "expanded_end_ms": round(end_ms, 1),
                        "effective_start_ms": round(pause_start_ms, 1),
                        "status": "near_existing",
                    })
                    continue
                pause = candidates.Pause(pause_start_ms, end_ms, end_ms - pause_start_ms)
                pauses.append(pause)
                decision = candidates.PauseDecision(
                    pause=pause,
                    dsp_ok=True,
                    vad_ok=None,
                    accepted=True,
                    reason="micro-tempo",
                    flatness=0.0,
                    zcr=0.0,
                    rms_ratio=ratio,
                )
                pause_decisions.append(decision)
                pause_diag[pause] = decision
                existing.append(center)
                micro_pause_debug.append({
                    "ayah": miss_a,
                    "expected_ms": round(expected_t, 1),
                    "center_ms": round(center, 1),
                    "start_ms": round(pause_start_ms, 1),
                    "end_ms": round(end_ms, 1),
                    "seed_start_ms": round(seed_start_ms, 1),
                    "seed_end_ms": round(seed_end_ms, 1),
                    "expanded_start_ms": round(start_ms, 1),
                    "expanded_end_ms": round(end_ms, 1),
                    "effective_start_ms": round(pause_start_ms, 1),
                    "rms_ratio": round(ratio, 3),
                    "interval_ms": [round(interval_start, 1), round(interval_end, 1)],
                    "status": "added",
                })
                emit(
                    f"Micro-пауза для аята {miss_a}: ожидание {sec(expected_t)} сек., "
                    f"найдена RMS-просадка {sec(seed_start_ms)}-{sec(seed_end_ms)} сек., "
                    f"расширена до {sec(start_ms)}-{sec(end_ms)} сек., "
                    f"эффективно {sec(pause_start_ms)}-{sec(end_ms)} сек., "
                    f"ratio={ratio:.3f}"
                )
                added = True
        if added:
            refresh_active_pauses()
        return added

    matched, sequence_events = run_sequence_match()
    if add_low_conf_pauses_from_gaps(matched, sequence_events):
        matched, sequence_events = run_sequence_match()
    if add_micro_pauses_from_gaps(matched, sequence_events):
        accepted_pause_count = len([d for d in pause_decisions if d.accepted])
        rejected_pause_count = len(pause_decisions) - accepted_pause_count
        matched, sequence_events = run_sequence_match()
    if add_low_conf_pauses_from_gaps(matched, sequence_events):
        matched, sequence_events = run_sequence_match()
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
        elif action == "unresolved_current":
            j = ev.get("pause")
            bi = ev.get("boundary")
            oi = ev.get("observed_boundary")
            if isinstance(j, int) and isinstance(bi, int) and isinstance(oi, int):
                observed = inner[oi] if 0 <= oi < len(inner) else "?"
                emit(
                    f"Пауза {pause_ord.get(pause_onsets[j], '?')} не подтвердила "
                    f"ожидаемый аят {inner[bi]}: лучше слышен {observed}, "
                    f"direction={ev.get('direction')}, sim={ev.get('observed_sim')} - "
                    "останавливаю проход и ищу локально"
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
                    and ev.get("action") in (
                        "confirmed",
                        "observed_current",
                        "weak_observed_current",
                        "recovered",
                        "final_weak_current",
                        "future_anchor",
                    )
                ),
                "",
            )
            if action == "recovered":
                min_keep_sim = max(0.0, _MIN_MATCH_SIM - _RECOVER_ECHO_SIM_SLACK)
            elif action == "final_weak_current":
                min_keep_sim = _FINAL_WEAK_MATCH_SIM
            elif action in {"observed_current", "weak_observed_current"}:
                min_keep_sim = max(0.0, _MIN_MATCH_SIM - _RECOVER_ECHO_SIM_SLACK)
            else:
                min_keep_sim = drop_sim
            reason = None
            if sim < min_keep_sim:
                reason = "low_sim"
            elif (
                action not in {"recovered", "observed_current", "weak_observed_current"}
                and far_weak_match(k, j, sim)
            ):
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
        if ev.get("action") in (
            "confirmed",
            "observed_current",
            "recovered",
            "future_anchor",
        )
        and ev.get("boundary") is not None
    }
    used_matched = {m for m in matched if m is not None}
    weak_micro_floor = max(0.0, _MIN_MATCH_SIM - _RECOVER_ECHO_SIM_SLACK)
    weak_pause_floor = max(_RECOVER_MATCH_SIM, weak_micro_floor)
    for k, mk in enumerate(matched):
        if mk is not None or sequence_reason.get(k) != "missing_before_future":
            continue
        prev_t = next((matched[ii] for ii in range(k - 1, -1, -1) if matched[ii] is not None), None)
        next_t = next((matched[ii] for ii in range(k + 1, len(matched)) if matched[ii] is not None), None)
        if prev_t is None or next_t is None:
            continue
        candidates: list[tuple[float, int]] = []
        for j, center in enumerate(pause_onsets):
            if center in used_matched or center <= prev_t or center >= next_t:
                continue
            pause = pause_by_center.get(center)
            diag = pause_diag.get(pause) if pause is not None else None
            if diag is None:
                continue
            s = _sim_ij(k, j)
            if diag.reason == "micro-tempo":
                keep = s >= weak_micro_floor
            else:
                keep = s >= weak_pause_floor
            if keep:
                candidates.append((s, j))
        if candidates:
            _s, j = max(candidates, key=lambda item: item[0])
            matched[k] = pause_onsets[j]
            used_matched.add(pause_onsets[j])
            sequence_action[k] = "weak_pause_recovery"
    for k, mk in enumerate(matched):
        if mk is not None:
            continue
        if sequence_reason.get(k) == "no_pause_in_window":
            continue
        prev_t = next((matched[ii] for ii in range(k - 1, -1, -1) if matched[ii] is not None), None)
        next_t = next((matched[ii] for ii in range(k + 1, len(matched)) if matched[ii] is not None), None)
        rough_t = _interp_wasl(k, exp, matched)
        near_strong_candidates: list[int] = []
        fallback_candidates: list[tuple[float, int]] = []
        for j, center in enumerate(pause_onsets):
            if center in used_matched:
                continue
            if prev_t is not None and center <= prev_t:
                continue
            if next_t is not None and center >= next_t:
                continue
            if (prev_t is None or next_t is None) and abs(center - rough_t) > _MATCH_PRIOR_WINDOW_MS:
                continue
            pause = pause_by_center.get(center)
            diag = pause_diag.get(pause) if pause is not None else None
            if diag is None:
                continue
            if diag.reason != "weak-dsp" and abs(center - rough_t) <= 2200.0:
                near_strong_candidates.append(j)
                continue
            s = _sim_ij(k, j)
            is_strong = diag.reason != "weak-dsp"
            if not is_strong and s < weak_micro_floor:
                continue
            if s < _FINAL_WEAK_MATCH_SIM and abs(center - rough_t) > 2200.0:
                continue
            dist_pen = min(1.0, abs(center - rough_t) / max(1.0, _MATCH_PRIOR_WINDOW_MS))
            score = s + max(0.0, tempo_score(k, j)) - 0.04 * dist_pen
            if is_strong:
                score += 0.03
            fallback_candidates.append((score, j))
        if near_strong_candidates:
            j = min(near_strong_candidates, key=lambda jj: abs(pause_onsets[jj] - rough_t))
            matched[k] = pause_onsets[j]
            used_matched.add(pause_onsets[j])
            sequence_action[k] = "pause_fallback"
            continue
        if fallback_candidates:
            _score, j = max(fallback_candidates, key=lambda item: item[0])
            matched[k] = pause_onsets[j]
            used_matched.add(pause_onsets[j])
            sequence_action[k] = "pause_fallback"
    pause_index_by_center = {center: j for j, center in enumerate(pause_onsets)}

    def opening_prefix_for(k: int, j: int) -> int:
        win_ms = decode_window_for(k) if needs_extended_decode(k, j) else decode_win_ms
        return _onset_prefix_len(decode_pause(j, win_ms=win_ms), openings[k])

    def opening_repair_candidate(k: int, j: int, *, allow_fuzzy: bool) -> bool:
        prefix = opening_prefix_for(k, j)
        required_prefix = 6 if opening_text_risk.get(k, 0) >= _AMBIG_PREFIX_CHARS else 3
        if prefix >= required_prefix:
            return True
        return allow_fuzzy and _sim_ij(k, j) >= 0.27

    repaired_run_until = -1

    def repair_opening_run(start_k: int, *, length: int = 4) -> None:
        nonlocal repaired_run_until
        end_k = min(len(matched) - 1, start_k + length)
        if end_k <= start_k or matched[start_k] is None:
            return
        repaired_run_until = max(repaired_run_until, end_k)
        run_boundaries = list(range(start_k + 1, end_k + 1))
        run_owners = set(run_boundaries)
        owner_by_center = {
            center: owner
            for owner, center in enumerate(matched)
            if center is not None
        }
        start_t = float(matched[start_k])
        old_end = max(
            (float(matched[ii]) for ii in run_boundaries if matched[ii] is not None),
            default=start_t,
        )
        block_end = max(old_end, start_t + 180_000.0)

        def candidate_score(kk: int, j: int) -> float | None:
            prefix = opening_prefix_for(kk, j)
            sim_s = _sim_ij(kk, j)
            if opening_text_risk.get(kk, 0) >= _AMBIG_PREFIX_CHARS:
                if prefix < 6:
                    return None
                return 1.0 + min(0.5, sim_s) + 0.02 * min(prefix, 12)
            if prefix >= 3:
                return 0.8 + min(0.5, sim_s) + 0.02 * min(prefix, 10)
            if sim_s >= 0.27:
                return sim_s
            return None

        layers: list[list[tuple[int, float]]] = []
        for kk in run_boundaries:
            layer: list[tuple[int, float]] = []
            for j, center in enumerate(pause_onsets):
                owner = owner_by_center.get(center)
                if center <= start_t + 2500.0 or center >= block_end:
                    continue
                if center in used_matched and owner not in run_owners:
                    continue
                sc = candidate_score(kk, j)
                if sc is not None:
                    layer.append((j, sc))
            if not layer:
                return
            layers.append(layer)

        states: dict[int, tuple[float, list[int]]] = {}
        for j, sc in layers[0]:
            gap_pen = min(0.4, max(0.0, pause_onsets[j] - start_t) / 180_000.0)
            states[j] = (sc - gap_pen, [j])
        for layer in layers[1:]:
            next_states: dict[int, tuple[float, list[int]]] = {}
            for j, sc in layer:
                center = pause_onsets[j]
                best: tuple[float, list[int]] | None = None
                for prev_j, (prev_score, path) in states.items():
                    prev_center = pause_onsets[prev_j]
                    if center <= prev_center + 2500.0:
                        continue
                    gap = center - prev_center
                    # Keep the block local, but allow long ayahs like 37:66.
                    gap_pen = min(0.5, abs(gap - 9000.0) / 180_000.0)
                    total = prev_score + sc - gap_pen
                    if best is None or total > best[0]:
                        best = (total, path + [j])
                if best is not None:
                    next_states[j] = best
            states = next_states
            if not states:
                return
        _score, path = max(states.values(), key=lambda item: item[0])
        for kk, j in zip(run_boundaries, path):
            old = matched[kk]
            stolen_owner = owner_by_center.get(pause_onsets[j])
            if stolen_owner is not None and stolen_owner != kk:
                matched[stolen_owner] = None
                sequence_action[stolen_owner] = "opening_run_repair_stolen"
            if old is not None:
                used_matched.discard(old)
            matched[kk] = pause_onsets[j]
            used_matched.add(pause_onsets[j])
            sequence_action[kk] = "opening_run_repair"

    for k, mk in enumerate(matched):
        if mk is None:
            continue
        prev_t = next((matched[ii] for ii in range(k - 1, -1, -1) if matched[ii] is not None), None)
        if prev_t is None:
            continue
        current_gap = mk - prev_t
        if current_gap > 30_000.0:
            candidates = [
                j for j, center in enumerate(pause_onsets)
                if center not in used_matched
                and prev_t + 2500.0 < center < mk - 2500.0
                and _sim_ij(k, j) >= 0.28
                and (
                    opening_text_risk.get(k, 0) < _AMBIG_PREFIX_CHARS
                    or opening_prefix_for(k, j) >= 6
                )
            ]
            repeated_opening_candidates = [
                j for j, center in enumerate(pause_onsets)
                if center not in used_matched
                and prev_t + 2500.0 < center < mk + 45_000.0
                and _sim_ij(k, j) >= 0.28
                and (
                    opening_text_risk.get(k, 0) < _AMBIG_PREFIX_CHARS
                    or opening_prefix_for(k, j) >= 6
                )
            ]
            if candidates and len(repeated_opening_candidates) >= 3:
                j = min(candidates, key=lambda jj: pause_onsets[jj])
                used_matched.discard(mk)
                matched[k] = pause_onsets[j]
                used_matched.add(pause_onsets[j])
                sequence_action[k] = "first_opening_repair"
                repair_opening_run(k)
                continue
        if k > repaired_run_until:
            continue
        if sequence_action.get(k) == "opening_run_repair":
            continue
        current_j = pause_index_by_center.get(mk)
        if current_j is not None and opening_repair_candidate(k, current_j, allow_fuzzy=False):
            continue
        prev_t = next((matched[ii] for ii in range(k - 1, -1, -1) if matched[ii] is not None), None)
        if prev_t is None:
            continue
        upper = prev_t + 92_000.0
        owner_by_center = {
            center: owner
            for owner, center in enumerate(matched)
            if center is not None
        }
        candidates = [
            j for j, center in enumerate(pause_onsets)
            if (
                center not in used_matched
                or k < owner_by_center.get(center, -1) <= repaired_run_until
            )
            and prev_t + 3000.0 < center < upper
            and opening_repair_candidate(k, j, allow_fuzzy=True)
        ]
        if candidates:
            j = min(
                candidates,
                key=lambda jj: (
                    0 if opening_prefix_for(k, jj) >= 6 else 1,
                    pause_onsets[jj],
                ),
            )
            stolen_owner = owner_by_center.get(pause_onsets[j])
            if stolen_owner is not None and stolen_owner != k:
                matched[stolen_owner] = None
                sequence_action[stolen_owner] = "opening_order_repair_stolen"
            used_matched.discard(mk)
            matched[k] = pause_onsets[j]
            used_matched.add(pause_onsets[j])
            sequence_action[k] = "opening_order_repair"
    matched_center_for_ayah: dict[int, float] = {}

    def rounded_wave_metrics(i: int, j: int) -> dict[str, float | int | bool | None]:
        if j < 0:
            return {
                "tempo_score": None,
                "prev_speech_ms": None,
                "next_speech_ms": None,
                "prev_span_ratio": None,
                "next_span_ratio": None,
                "waqf_before": None,
                "waqf_after": None,
            }
        m = wave_tempo_metrics(i, j)
        return {
            "tempo_score": round(float(m["tempo_score"]), 3),
            "prev_speech_ms": round(float(m["prev_speech_ms"]), 1),
            "next_speech_ms": round(float(m["next_speech_ms"]), 1),
            "prev_expected_ms": round(float(m["prev_expected_ms"]), 1),
            "next_expected_ms": round(float(m["next_expected_ms"]), 1),
            "prev_span_ratio": round(float(m["prev_span_ratio"]), 3),
            "next_span_ratio": round(float(m["next_span_ratio"]), 3),
            "prev_quality": round(float(m["prev_quality"]), 3),
            "next_quality": round(float(m["next_quality"]), 3),
            "waqf_before": int(m["waqf_before"]),
            "waqf_after": int(m["waqf_after"]),
            "tempo_ms_per_unit": round(float(m["tempo_ms_per_unit"]), 3),
        }

    def pause_timing_point(
        center_ms: float,
        *,
        sim: float | None = None,
        reason: str | None = None,
    ) -> float:
        pause = pause_by_center.get(center_ms)
        if pause is not None and reason != "micro-tempo" and pause.end_ms - pause.start_ms >= 900.0:
            width_ms = pause.end_ms - pause.start_ms
            if (
                width_ms >= _LONG_PAUSE_CENTER_MS
                or (
                    width_ms >= _TRUSTED_PAUSE_CENTER_MS
                    and sim is not None
                    and sim >= _RECOVER_MATCH_SIM
                )
                or (
                    width_ms <= _COMPACT_PAUSE_CENTER_MS
                    and sim is not None
                    and sim >= _MIN_MATCH_SIM
                )
            ):
                return center_ms
            return pause.start_ms
        if pause is not None and sim is not None and sim < _PAUSE_TRUST_SIM and reason != "micro-tempo":
            if pause.end_ms - pause.start_ms >= 700.0:
                return pause.start_ms
            return min(
                pause.end_ms,
                center_ms + _SHORT_PAUSE_RIGHT_BIAS * max(0.0, pause.end_ms - center_ms),
            )
        return center_ms

    if basmala and pause_onsets:
        first_center = pause_onsets[0]
        first_pause = pause_by_center.get(first_center)
        if (
            first_pause is not None
            and first_pause.start_ms <= starts_ms[a1] + 1200.0
            and first_pause.end_ms >= starts_ms[a1] - 100.0
        ):
            first_opening = " ".join(
                word_trs[i] for i in ayah_words[a1] if i < len(word_trs)
            )
            pause_heard = decode_pause(0)
            pause_sim = _sim(pause_heard, first_opening)
            anchor_heard = greedy_decode(
                max(0.0, starts_ms[a1] - 50.0),
                min(total_ms, starts_ms[a1] + _VERIFY_WINDOW_MS),
            )
            anchor_sim = _sim(anchor_heard, first_opening)
            first_diag = pause_diag.get(first_pause)
            if pause_sim >= max(0.50, anchor_sim + 0.10):
                if first_diag is not None and first_diag.reason == "weak-dsp":
                    content_start = min(
                        first_pause.end_ms,
                        first_pause.center_ms
                        + 0.5 * max(0.0, first_pause.end_ms - first_pause.center_ms),
                    )
                else:
                    content_start = pause_timing_point(first_center)
                content_start = max(starts_ms[a1], content_start)
                starts_ms[a1] = content_start
                confs[a1] = max(confs[a1], pause_sim)
                for row in tr_ayahs:
                    if row.get("ayah") == a1:
                        row["onset_ms"] = content_start
                        row["conf"] = confs[a1]
                        row["intro_pause_anchor"] = True
                        row["intro_pause_sim"] = round(pause_sim, 3)
                        row["intro_anchor_sim"] = round(anchor_sim, 3)
                        break
                emit(
                    f"Первый аят: первая пауза {sec(first_pause.start_ms)}-"
                    f"{sec(first_pause.end_ms)} сек. звучит как начало аята "
                    f"(sim={pause_sim:.3f} > anchor={anchor_sim:.3f}); "
                    f"ставлю старт {sec(content_start)} сек."
                )

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
    snapped_wasl_hints: dict[int, float] = {}

    def micro_backtrack_candidate(i: int, j: int) -> int | None:
        pause = pause_by_center.get(pause_onsets[j])
        diag = pause_diag.get(pause) if pause is not None else None
        if diag is None or diag.reason != "micro-tempo":
            return None
        if _sim_ij(i, j) >= _CONFIRM_MATCH_SIM:
            return None
        for jj in range(j - 1, -1, -1):
            if pause_onsets[j] - pause_onsets[jj] > _MICRO_BACKTRACK_MS:
                break
            prev = pause_by_center.get(pause_onsets[jj])
            prev_diag = pause_diag.get(prev) if prev is not None else None
            if prev is None or prev_diag is None or prev_diag.reason == "micro-tempo":
                continue
            if any(other_i != i and matched[other_i] == pause_onsets[jj] for other_i in range(len(matched))):
                continue
            metrics = wave_tempo_metrics(i, jj)
            if (
                _sim_ij(i, jj) >= _FINAL_WEAK_MATCH_SIM
                and float(metrics["next_speech_ms"]) <= _MICRO_BACKTRACK_MS
            ):
                return jj
        return None

    def weak_forward_candidate(i: int, j: int) -> int | None:
        pause = pause_by_center.get(pause_onsets[j])
        diag = pause_diag.get(pause) if pause is not None else None
        if diag is None or diag.reason != "weak-dsp":
            return None
        base_s = _sim_ij(i, j)
        for jj in range(j + 1, len(pause_onsets)):
            if pause_onsets[jj] - pause_onsets[j] > 3000.0:
                break
            if any(other_i != i and matched[other_i] == pause_onsets[jj] for other_i in range(len(matched))):
                continue
            next_pause = pause_by_center.get(pause_onsets[jj])
            next_diag = pause_diag.get(next_pause) if next_pause is not None else None
            if next_diag is None or next_diag.reason not in {"weak-dsp", "dsp"}:
                continue
            if _sim_ij(i, jj) >= base_s + 0.02:
                return jj
        return None

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
            pause = pause_by_center.get(matched[k])
            pause_reason = pause_diag[pause].reason if pause in pause_diag else None
            backtrack_j = micro_backtrack_candidate(k, j)
            if backtrack_j is not None:
                j = backtrack_j
                matched[k] = pause_onsets[j]
            forward_j = weak_forward_candidate(k, j)
            if forward_j is not None:
                j = forward_j
                matched[k] = pause_onsets[j]
            sim = _sim_ij(k, j)
            wave_metrics = rounded_wave_metrics(k, j)
            rough = pause_timing_point(matched[k], sim=sim, reason=pause_reason)
            timing_adjusted = False
            if (
                sequence_action.get(k) == "weak_pause_recovery"
                and pause is not None
                and pause_reason != "micro-tempo"
                and pause.end_ms - pause.start_ms >= 700.0
                and sim >= weak_pause_floor
            ):
                rough = matched[k]
            if (
                pause_reason == "micro-tempo"
                and sim < _CONFIRM_MATCH_SIM
                and (wave_metrics.get("prev_speech_ms") or 0.0) <= 1000.0
                and (wave_metrics.get("next_speech_ms") or 0.0) >= 2500.0
            ):
                rough = min(
                    total_ms,
                    rough + 0.36 * float(wave_metrics.get("next_speech_ms") or 0.0),
                )
                timing_adjusted = True
            if (
                pause is not None
                and pause_reason != "micro-tempo"
                and pause.end_ms - pause.start_ms >= 1800.0
                and (wave_metrics.get("prev_speech_ms") or 0.0) <= 1500.0
                and (wave_metrics.get("next_speech_ms") or 0.0) >= 3000.0
            ):
                rough = pause.end_ms
            matched_center_for_ayah[a] = matched[k]
            place[a] = (rough, True)
            pause_match_by_ayah[a] = {
                "pause": pause_ord.get(matched[k]),
                "sim": sim,
                "center_ms": matched[k],
                "rough_ms": rough,
                "action": sequence_action.get(k),
                "reason": pause_reason,
                "timing_adjusted": timing_adjusted,
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
                "pause_reason": pause_reason,
                **wave_metrics,
            })
            emit(
                f"Пауза {pause_ord.get(matched[k], '?')} -> детект аята {a}, "
                f"черновой старт {sec(rough)} сек., sim={sim:.3f}, "
                f"tempo={wave_metrics.get('tempo_score')}, "
                f"ожидание {sec(expected)} сек., delta_prior={rough - expected:+.0f}мс, "
                f"best_near={pause_ord.get(pause_onsets[best_near_j], '?') if best_near_j >= 0 else '?'}"
                f"/{best_near_sim:.3f}"
            )
        else:
            drop_reason = dropped_match_reason.get(k) or sequence_reason.get(k)
            rough = _interp_wasl(k, exp, matched)
            if sequence_reason.get(k) == "missing_before_future":
                prev_t = next(
                    (matched[ii] for ii in range(k - 1, -1, -1) if matched[ii] is not None),
                    None,
                )
                next_t = next(
                    (matched[ii] for ii in range(k + 1, len(matched)) if matched[ii] is not None),
                    None,
                )
                snap_candidates = [
                    center for center in pause_onsets
                    if center not in used_matched
                    and (prev_t is None or center > prev_t)
                    and (next_t is None or center < next_t)
                    and strong_pause(center)
                    and abs(center - rough) <= 2200.0
                ]
                if snap_candidates:
                    rough = min(snap_candidates, key=lambda center: abs(center - rough))
                    snapped_wasl_hints[a] = rough
            wave_j = best_near_j if best_near_j >= 0 else best_j
            wave_metrics = rounded_wave_metrics(k, wave_j)
            place[a] = (rough, False)
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
                **wave_metrics,
            })
            emit(
                f"Без паузы -> детект середины/wasl аята {a}, "
                f"черновой старт {sec(place[a][0])} сек., "
                f"best_pause={pause_ord.get(pause_onsets[best_j], '?') if best_j >= 0 else '?'}, "
                f"best_sim={best_sim:.3f}, "
                f"tempo={wave_metrics.get('tempo_score')}, "
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
            pause = pause_by_center.get(float(match.get("center_ms") or c0))
            pause_reason = str(match.get("reason") or "")
            timing_adjusted = bool(match.get("timing_adjusted"))
            needs_pause_refine = pause_reason == "micro-tempo"
            if timing_adjusted:
                onset, conf, win_used = c, max(conf_thresh, pause_sim), 0.0
                reliable = True
                emit(
                    f"Пауза {match.get('pause', '?')} дала скорректированный старт "
                    f"аята {a}: {sec(onset)} сек., refine не нужен"
                )
            elif pause_sim >= _PAUSE_TRUST_SIM and not needs_pause_refine:
                onset, conf, win_used = c, max(conf_thresh, pause_sim), 0.0
                reliable = True
                emit(
                    f"Пауза {match.get('pause', '?')} подтвердила аят {a}: "
                    f"sim={pause_sim:.3f}, старт {sec(onset)} сек., refine не нужен"
                )
            else:
                # Weak pause matches use refine as proof that the first word
                # starts next to this pause. The emitted boundary remains the
                # quiet shelf center; cutting at the word onset loses the pause.
                c = min(max(c0, prev_start + 30.0), total_ms)
                refined_onset, refined_conf, win_used = refine(c, fa, prev_start, 0, ayah=a)
                confirm_delta_ms = _PAUSE_REFINE_CONFIRM_MS
                if pause is not None and pause.end_ms - pause.start_ms >= 700.0:
                    confirm_delta_ms = max(confirm_delta_ms, pause.end_ms - pause.start_ms)
                if _pause_refine_confirms_pause(
                    c,
                    refined_onset,
                    refined_conf,
                    max_delta_ms=confirm_delta_ms,
                ):
                    if needs_pause_refine and pause is not None:
                        onset = _pause_center_with_refined_onset(
                            pause.start_ms,
                            c,
                            refined_onset,
                        )
                    elif (
                        pause_reason == "weak-dsp"
                        and refined_onset is not None
                        and refined_conf >= conf_reliable
                        and 300.0 <= c - refined_onset <= confirm_delta_ms
                    ):
                        onset = refined_onset
                    else:
                        onset = c
                    conf = max(conf_thresh, pause_sim, refined_conf)
                    reliable = True
                    emit(
                        f"Пауза {match.get('pause', '?')} подтверждена refine для аята {a}: "
                        f"слово найдено {sec(refined_onset)} сек., "
                        f"оставляю старт в центре паузы {sec(onset)} сек."
                    )
                else:
                    onset = refined_onset
                    conf = refined_conf
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
        repair_match = pause_match_by_ayah.get(a, {})
        repair_pause_reason = str(repair_match.get("reason") or "")
        micro_repair_candidate = (
            place[a][1]
            and repair_pause_reason == "micro-tempo"
            and o2 is not None
            and c2 >= conf_thresh
            and abs(o2 - starts_ms.get(a, o2)) <= 3500.0
        )
        if o2 is not None and (c2 >= confs.get(a, 0.0) or micro_repair_candidate):
            # A refine that still never locks on (conf below thresh) returns a
            # noisy onset — on a long melodic ayah the model may not find the
            # boundary at all. Geometry between two reliable anchors beats model
            # noise, so fall back to the interpolated guess in that case.
            chosen = o2 if c2 >= conf_thresh else pred
            old_onset = starts_ms.get(a, pred)
            pause_hint = wasl_pause_hints.get(a) or snapped_wasl_hints.get(a)
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
            pause_sim = float(repair_match.get("sim") or 0.0)
            strong_pause_repair = (
                place[a][1]
                and pause_sim < _PAUSE_TRUST_SIM
                and c2 >= conf_reliable
                and abs(chosen - old_onset) > 500.0
            )
            micro_pause_repair = (
                micro_repair_candidate
                and abs(chosen - old_onset) <= 3500.0
            )
            repair_on_pause_grid = _dist_pause(chosen) <= 350.0 and old_onset - chosen > 500.0
            off_grid_back_repair = 500.0 < old_onset - chosen <= 2500.0
            if (
                place[a][1]
                and confs.get(a, 0.0) >= conf_thresh
                and _dist_pause(chosen) > 350.0
                and not (micro_pause_repair or (strong_pause_repair and off_grid_back_repair))
            ):
                continue
            if (
                place[a][1]
                and confs.get(a, 0.0) >= conf_thresh
                and abs(old_onset - place[a][0]) <= 80.0
                and abs(chosen - old_onset) > 500.0
                and not (
                    micro_pause_repair
                    or (strong_pause_repair and (repair_on_pause_grid or off_grid_back_repair))
                )
            ):
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
        pause_decode_len_by_ord: dict[int, int] = {}
        for (i, win_key), heard in pause_decodes.items():
            p = pause_onsets[i]
            ord_ = pause_ord[p]
            if win_key >= pause_decode_len_by_ord.get(ord_, 0):
                pause_decode_len_by_ord[ord_] = win_key
                pause_decode_by_ord[ord_] = heard
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
            observed_boundary = out.pop("observed_boundary", None)
            if isinstance(observed_boundary, int) and 0 <= observed_boundary < len(inner):
                out["observed_ayah"] = inner[observed_boundary]
            pause_idx = out.get("pause")
            if isinstance(pause_idx, int) and 0 <= pause_idx < len(pause_onsets):
                out["pause_idx"] = pause_ord.get(pause_onsets[pause_idx])
                out["pause_center_ms"] = round(pause_onsets[pause_idx], 1)
                del out["pause"]
            return out

        def phase1_pause_trace(p) -> dict:
            d = pause_diag.get(p)
            j = pause_ord.get(p.center_ms)
            boundary_i = -1
            wave_metrics = rounded_wave_metrics(0, -1)
            if j is not None and p.center_ms in pause_onsets and exp:
                pj = pause_onsets.index(p.center_ms)
                boundary_i = min(range(len(exp)), key=lambda i: abs(exp[i] - p.center_ms))
                wave_metrics = rounded_wave_metrics(boundary_i, pj)
            return {
                "idx": all_pause_ord.get(p.center_ms, pause_ord.get(p.center_ms)),
                "active_idx": pause_ord.get(p.center_ms),
                "start_ms": round(p.start_ms, 1),
                "end_ms": round(p.end_ms, 1),
                "center_ms": round(p.center_ms, 1),
                "depth_ms": round(p.depth_ms, 1),
                "active": p.center_ms in pause_onsets,
                "activated_low_conf": p.center_ms in active_low_conf_centers,
                "reason": d.reason if d is not None else None,
                "quality": pause_tier(d) if d is not None else "micro",
                "rms_ratio": round(d.rms_ratio, 4) if d is not None else None,
                "flatness": round(d.flatness, 4) if d is not None else None,
                "zcr": round(d.zcr, 4) if d is not None else None,
                "vad_ok": d.vad_ok if d is not None else None,
                "decode_reason": pause_decode_reasons.get(p.center_ms),
                "decode_window_ms": round(pause_decode_windows.get(p.center_ms, 0.0), 1),
                "decode_start_vs_pause_end_ms": round(
                    (
                        pause_ord.get(p.center_ms)
                        and (pause_decode_starts.get(p.center_ms, p.center_ms) - p.end_ms)
                    ) or 0.0,
                    1,
                ),
                "decode": short(pause_decode_by_ord.get(pause_ord.get(p.center_ms), ""), 120),
                "tempo_nearest_ayah": inner[boundary_i] if 0 <= boundary_i < len(inner) else None,
                **wave_metrics,
            }

        trace_pauses = [d.pause for d in pause_decisions]
        for p in pauses:
            if p not in pause_diag and p not in trace_pauses:
                trace_pauses.append(p)
        trace["phase1_pauses"] = [phase1_pause_trace(p) for p in trace_pauses]
        trace["pause_matches"] = pause_match_debug
        trace["micro_pauses"] = micro_pause_debug
        trace["low_conf_activations"] = low_conf_activation_debug
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
            "trusted_pauses": trusted_pause_count,
            "weak_pauses": weak_pause_count,
            "active_pauses": len(pause_onsets),
            "activated_low_conf_pauses": len(active_low_conf_centers),
            "used_pauses": len(used_pause_ord),
            "decoded_pauses": len({i for i, _extended in pause_decodes}),
            "decode_reads": len(pause_decodes),
            "micro_pauses": sum(1 for item in micro_pause_debug if item.get("status") == "added"),
            "ambiguous_openings": [
                {
                    "ayah": inner[i],
                    "shared_prefix_chars": opening_shared_prefix.get(i, 0),
                    "internal_repeat_chars": opening_internal_prefix.get(i, 0),
                    "decode_window_ms": round(decode_window_for(i), 1),
                    "sim_chars": sim_chars_for(i),
                }
                for i in sorted(ambiguous_opening_idx)
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
                win_ms = decode_window_for(k) if needs_extended_decode(k, j) else None
                heard = decode_pause(j, win_ms=win_ms)
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
    reliable_coverage = sum(1 for t in tr_ayahs if t.get("reliable")) / max(1, len(tr_ayahs))
    strong_pause_result = (
        pause_coverage >= _DRIFT_MIN_PAUSE_COVERAGE
        and (_FALLBACK_CONF <= 0.0 or mean_conf >= _FALLBACK_CONF)
    )
    reliable_align_result = reliable_coverage >= 0.95 and mean_conf >= 0.50
    drifted = bool(
        _VERIFY_SIM
        and verify_sim < _VERIFY_SIM
        and not strong_pause_result
        and not reliable_align_result
    )
    if (
        not drifted
        and _FALLBACK_CONF
        and mean_conf < _FALLBACK_CONF
        and pause_coverage < _DRIFT_MIN_PAUSE_COVERAGE
        and not reliable_align_result
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
