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
_COMPACT_PAUSE_CENTER_MS = float(os.environ.get("SC_FAST_COMPACT_PAUSE_CENTER_MS", "1300"))
_MICRO_BACKTRACK_MS = float(os.environ.get("SC_FAST_MICRO_BACKTRACK_MS", "3500"))
_MICRO_EXPECTED_RETRY_MS = float(os.environ.get("SC_FAST_MICRO_EXPECTED_RETRY_MS", "1000"))
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
_MAX_SECTION_REPAIR_ALIGN_MS = float(
    os.environ.get("SC_FAST_MAX_SECTION_REPAIR_ALIGN_MS", "120000")
)
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


def _micro_valley_center(valley: tuple[float, float, float, float, float]) -> float:
    start_ms, end_ms, _ratio, seed_start_ms, _seed_end_ms = valley
    return 0.5 * (max(start_ms, seed_start_ms) + end_ms)


def _prefer_expected_micro_valley(
    current: tuple[float, float, float, float, float],
    expected_candidate: tuple[float, float, float, float, float] | None,
    *,
    expected_ms: float,
    late_ms: float = 1500.0,
    min_back_ms: float = 500.0,
    ratio_slack: float = 1.35,
) -> tuple[float, float, float, float, float]:
    if expected_candidate is None:
        return current
    current_center = _micro_valley_center(current)
    expected_center = _micro_valley_center(expected_candidate)
    if current_center - expected_ms < late_ms:
        return current
    if current_center - expected_center < min_back_ms:
        return current
    if expected_candidate[2] > current[2] * ratio_slack:
        return current
    return expected_candidate


def _micro_shelf_start_candidate(
    *,
    pause_reason: str | None,
    sim: float,
    prev_quality: float,
    next_quality: float,
    center_ms: float,
    shelf_start_ms: float | None,
    confirm_match_sim: float,
    min_back_ms: float = 120.0,
    max_back_ms: float = 700.0,
    min_prev_quality: float = 0.90,
    max_next_quality: float = 0.45,
) -> float | None:
    if pause_reason != "micro-tempo" or sim >= confirm_match_sim:
        return None
    if shelf_start_ms is None:
        return None
    back_ms = center_ms - shelf_start_ms
    if not (min_back_ms <= back_ms <= max_back_ms):
        return None
    if prev_quality < min_prev_quality or next_quality > max_next_quality:
        return None
    return shelf_start_ms


def _rejected_micro_valley_candidate(
    *,
    pause_reason: str | None,
    action: str | None,
    valley_ratio: float,
    shift_ms: float,
    selected_prev_quality: float,
    selected_next_quality: float,
    candidate_prev_quality: float,
    candidate_next_quality: float,
    max_ratio: float = 0.16,
    min_shift_ms: float = 700.0,
    max_min_quality_loss: float = 0.05,
    max_avg_quality_loss: float = 0.08,
) -> bool:
    if pause_reason != "rejected" or action != "recovered":
        return False
    if valley_ratio > max_ratio or shift_ms < min_shift_ms:
        return False
    selected_min = min(selected_prev_quality, selected_next_quality)
    candidate_min = min(candidate_prev_quality, candidate_next_quality)
    if candidate_min < selected_min - max_min_quality_loss:
        return False
    selected_avg = 0.5 * (selected_prev_quality + selected_next_quality)
    candidate_avg = 0.5 * (candidate_prev_quality + candidate_next_quality)
    return candidate_avg >= selected_avg - max_avg_quality_loss


def _weak_dsp_expected_micro_valley_candidate(
    *,
    pause_reason: str | None,
    action: str | None,
    pause_sim: float,
    valley_ratio: float,
    shift_ms: float,
    expected_gain_ms: float,
    selected_prev_quality: float,
    selected_next_quality: float,
    candidate_prev_quality: float,
    candidate_next_quality: float,
    max_sim: float = 0.50,
    max_ratio: float = 0.13,
    min_shift_ms: float = 700.0,
    min_expected_gain_ms: float = 700.0,
    max_min_quality_loss: float = 0.10,
    max_avg_quality_loss: float = 0.10,
) -> bool:
    if pause_reason != "weak-dsp" or action != "recovered":
        return False
    if pause_sim > max_sim:
        return False
    if valley_ratio > max_ratio:
        return False
    if shift_ms < min_shift_ms or expected_gain_ms < min_expected_gain_ms:
        return False
    selected_min = min(selected_prev_quality, selected_next_quality)
    candidate_min = min(candidate_prev_quality, candidate_next_quality)
    if candidate_min < selected_min - max_min_quality_loss:
        return False
    selected_avg = 0.5 * (selected_prev_quality + selected_next_quality)
    candidate_avg = 0.5 * (candidate_prev_quality + candidate_next_quality)
    return candidate_avg >= selected_avg - max_avg_quality_loss


def _weak_dsp_post_selected_micro_valley_candidate(
    *,
    pause_reason: str | None,
    action: str | None,
    pause_sim: float,
    valley_ratio: float,
    forward_shift_ms: float,
    selected_prev_quality: float,
    selected_next_quality: float,
    candidate_prev_quality: float,
    candidate_next_quality: float,
    max_sim: float = 0.35,
    max_ratio: float = 0.13,
    min_forward_shift_ms: float = 1000.0,
    max_selected_prev_quality: float = 0.70,
    max_selected_next_quality: float = 0.70,
    min_quality_gain: float = 0.10,
    min_candidate_quality: float = 0.65,
) -> bool:
    if pause_reason != "weak-dsp" or action != "recovered":
        return False
    if pause_sim > max_sim or valley_ratio > max_ratio:
        return False
    if forward_shift_ms < min_forward_shift_ms:
        return False
    if (
        selected_prev_quality > max_selected_prev_quality
        or selected_next_quality > max_selected_next_quality
    ):
        return False
    if candidate_prev_quality < selected_prev_quality + min_quality_gain:
        return False
    if candidate_next_quality < selected_next_quality + min_quality_gain:
        return False
    return min(candidate_prev_quality, candidate_next_quality) >= min_candidate_quality


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


def _pause_backed_refine_start(
    pause_center_ms: float,
) -> float:
    return pause_center_ms


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


def _assign_opening_block(
    boundaries: list[int],
    candidates: list[int],
    *,
    candidate_time: Callable[[int], float],
    score: Callable[[int, int], float | None],
    after_ms: float,
    expected_time: Callable[[int], float] | None = None,
    after_expected_ms: float | None = None,
    min_gap_ms: float = 2500.0,
    target_gap_ms: float = 9000.0,
    gap_window_ms: float = 180_000.0,
    duration_ceil: float = 12.0,
    abs_gap_ceil_ms: float = 45_000.0,
) -> dict[int, int]:
    """Order-preserving local assignment of repeated-opening ayahs to pauses.

    This is stricter than a per-boundary best match: every returned assignment
    belongs to one monotonic block, so a later repeated opening cannot steal an
    earlier ayah if that leaves the neighboring ayahs unexplained.
    """
    if not boundaries or not candidates:
        return {}

    def plausible_gap(observed_gap_ms: float, expected_gap_ms: float | None) -> bool:
        if observed_gap_ms <= min_gap_ms:
            return False
        if expected_gap_ms is None or expected_gap_ms <= 0:
            return True
        return observed_gap_ms <= max(abs_gap_ceil_ms, duration_ceil * expected_gap_ms)

    layers: list[list[tuple[int, float]]] = []
    for boundary in boundaries:
        layer: list[tuple[int, float]] = []
        for candidate in candidates:
            if candidate_time(candidate) <= after_ms + min_gap_ms:
                continue
            candidate_score = score(boundary, candidate)
            if candidate_score is not None:
                layer.append((candidate, candidate_score))
        if not layer:
            return {}
        layers.append(layer)

    def better(
        candidate: tuple[float, list[int]],
        current: tuple[float, list[int]] | None,
    ) -> bool:
        if current is None:
            return True
        candidate_score, candidate_path = candidate
        current_score, current_path = current
        if candidate_score > current_score + 1e-9:
            return True
        if abs(candidate_score - current_score) <= 1e-9:
            return [candidate_time(j) for j in candidate_path] < [
                candidate_time(j) for j in current_path
            ]
        return False

    states: dict[int, tuple[float, list[int]]] = {}
    first_expected_gap = None
    if expected_time is not None and after_expected_ms is not None:
        first_expected_gap = expected_time(boundaries[0]) - after_expected_ms
    for candidate, candidate_score in layers[0]:
        first_gap = candidate_time(candidate) - after_ms
        if not plausible_gap(first_gap, first_expected_gap):
            continue
        gap_penalty = min(0.4, max(0.0, first_gap) / gap_window_ms)
        states[candidate] = (candidate_score - gap_penalty, [candidate])

    for layer_index, layer in enumerate(layers[1:], start=1):
        boundary = boundaries[layer_index]
        previous_boundary = boundaries[layer_index - 1]
        expected_gap = (
            expected_time(boundary) - expected_time(previous_boundary)
            if expected_time is not None
            else None
        )
        next_states: dict[int, tuple[float, list[int]]] = {}
        for candidate, candidate_score in layer:
            current_time = candidate_time(candidate)
            best: tuple[float, list[int]] | None = None
            for previous_candidate, (previous_score, path) in states.items():
                previous_time = candidate_time(previous_candidate)
                gap = current_time - previous_time
                if not plausible_gap(gap, expected_gap):
                    continue
                gap_penalty = min(0.5, abs(gap - target_gap_ms) / gap_window_ms)
                proposed = (previous_score + candidate_score - gap_penalty, path + [candidate])
                if better(proposed, best):
                    best = proposed
            if best is not None and better(best, next_states.get(candidate)):
                next_states[candidate] = best
        states = next_states
        if not states:
            return {}

    _score, path = max(
        states.values(),
        key=lambda item: (
            item[0],
            [-candidate_time(candidate) for candidate in item[1]],
        ),
    )
    return dict(zip(boundaries, path, strict=True))


def _skips_dense_pause_block(
    boundary: int,
    selected: int,
    candidates: list[int],
    *,
    candidate_time: Callable[[int], float],
    expected_time: Callable[[int], float],
    previous_boundary: int,
    previous_time: float,
    next_boundary: int,
    min_gap_ms: float = 2500.0,
    min_quality: float = 0.35,
    selected_quality_ceiling: float = 0.45,
) -> bool:
    """Does ``selected`` jump over a plausible local sequence of ayah starts?"""
    if boundary + 1 >= next_boundary:
        return False
    expected_first = expected_time(boundary) - expected_time(previous_boundary)
    selected_quality = _span_ratio_quality(
        candidate_time(selected) - previous_time,
        expected_first,
    )
    if selected_quality >= selected_quality_ceiling:
        return False

    selected_time = candidate_time(selected)
    ordered = [
        candidate
        for candidate in sorted(candidates, key=candidate_time)
        if previous_time + min_gap_ms < candidate_time(candidate) < selected_time - min_gap_ms
    ]
    for first in ordered:
        first_time = candidate_time(first)
        first_quality = _span_ratio_quality(first_time - previous_time, expected_first)
        if first_quality < min_quality:
            continue
        expected_second = expected_time(boundary + 1) - expected_time(boundary)
        for second in ordered:
            second_time = candidate_time(second)
            if second_time <= first_time + min_gap_ms:
                continue
            if _span_ratio_quality(second_time - first_time, expected_second) >= min_quality:
                return True
    return False


def _section_repair_member(
    *,
    from_pause: bool,
    action: str | None,
    sim: float | None,
    min_sim: float,
) -> bool:
    if not from_pause:
        return True
    return action in {"pause_fallback", "weak_pause_recovery"} and (sim or 0.0) < min_sim


def _prior_pause_before_internal_stop(
    previous_time: float,
    raw_time: float,
    expected_gap_ms: float,
    strong_candidate_times: list[float],
    *,
    raw_is_internal_stop: bool,
    min_gap_ms: float = 2500.0,
    min_quality: float = 0.55,
    quality_margin: float = 0.20,
) -> float | None:
    """Pick a strong prior pause only when raw onset is a likely internal stop."""
    if not raw_is_internal_stop:
        return None
    raw_quality = _span_ratio_quality(raw_time - previous_time, expected_gap_ms)
    best: tuple[float, float] | None = None
    for candidate_time in strong_candidate_times:
        if not (previous_time + min_gap_ms < candidate_time < raw_time - min_gap_ms):
            continue
        quality = _span_ratio_quality(candidate_time - previous_time, expected_gap_ms)
        if quality < min_quality or quality < raw_quality + quality_margin:
            continue
        if best is None or quality > best[0] or (
            abs(quality - best[0]) <= 1e-9 and candidate_time < best[1]
        ):
            best = (quality, candidate_time)
    return best[1] if best is not None else None


def _accept_later_pause_grid_repair(
    *,
    from_pause: bool,
    original_reason: str,
    original_sim: float,
    old_onset: float,
    repaired_onset: float,
    repaired_conf: float,
    nearest_pause_dist_ms: float,
    nearest_pause_reason: str | None,
    previous_time: float,
    next_time: float | None,
    expected_prev_gap_ms: float,
    expected_next_gap_ms: float | None,
    conf_reliable: float,
    pause_trust_sim: float,
    min_forward_ms: float = 500.0,
    max_forward_ms: float = 9000.0,
    min_span_quality: float = 0.55,
) -> bool:
    """Allow repair to move a weak micro pause onto a later real pause grid."""
    if not from_pause or original_reason != "micro-tempo":
        return False
    if original_sim >= pause_trust_sim or repaired_conf < conf_reliable:
        return False
    forward_ms = repaired_onset - old_onset
    if not (min_forward_ms < forward_ms <= max_forward_ms):
        return False
    if nearest_pause_dist_ms > 350.0 or nearest_pause_reason not in {"dsp", "weak-dsp"}:
        return False
    if _span_ratio_quality(repaired_onset - previous_time, expected_prev_gap_ms) < min_span_quality:
        return False
    if next_time is not None and expected_next_gap_ms is not None:
        if _span_ratio_quality(next_time - repaired_onset, expected_next_gap_ms) < min_span_quality:
            return False
    return True


def _prior_pause_between_anchors(
    previous_time: float,
    selected_time: float,
    next_time: float,
    expected_prev_gap_ms: float,
    expected_next_gap_ms: float,
    strong_candidate_times: list[float],
    *,
    selected_is_internal_stop: bool,
    min_gap_ms: float = 2500.0,
    min_quality: float = 0.55,
    quality_margin: float = 0.25,
) -> float | None:
    """Choose a prior pause only if it better explains both neighboring spans."""
    if not selected_is_internal_stop:
        return None

    selected_prev = _span_ratio_quality(selected_time - previous_time, expected_prev_gap_ms)
    selected_next = _span_ratio_quality(next_time - selected_time, expected_next_gap_ms)
    selected_pair = min(selected_prev, selected_next)
    best: tuple[float, float, float] | None = None
    for candidate_time in strong_candidate_times:
        if not (previous_time + min_gap_ms < candidate_time < selected_time - min_gap_ms):
            continue
        candidate_prev = _span_ratio_quality(candidate_time - previous_time, expected_prev_gap_ms)
        candidate_next = _span_ratio_quality(next_time - candidate_time, expected_next_gap_ms)
        candidate_pair = min(candidate_prev, candidate_next)
        candidate_avg = 0.5 * (candidate_prev + candidate_next)
        selected_avg = 0.5 * (selected_prev + selected_next)
        if candidate_pair < min_quality:
            continue
        if candidate_pair < selected_pair + quality_margin:
            continue
        if candidate_avg < selected_avg + quality_margin:
            continue
        if best is None or candidate_pair > best[0] or (
            abs(candidate_pair - best[0]) <= 1e-9
            and (candidate_avg > best[1] or (
                abs(candidate_avg - best[1]) <= 1e-9 and candidate_time < best[2]
            ))
        ):
            best = (candidate_pair, candidate_avg, candidate_time)
    return best[2] if best is not None else None


def _internal_stop_repair_candidate(
    *,
    from_pause: bool,
    action: str | None,
    reason: str | None,
    sim: float,
    pause_duration_ms: float,
    prev_speech_ms: float,
    next_speech_ms: float,
    has_micro_stop: bool,
    pause_trust_sim: float,
) -> bool:
    """A selected pause is likely an internal stop, needing local section decode."""
    return (
        from_pause
        and action == "observed_current"
        and reason != "micro-tempo"
        and sim < pause_trust_sim
        and pause_duration_ms >= 700.0
        and prev_speech_ms >= 1800.0
        and next_speech_ms <= 500.0
        and has_micro_stop
    )


def _reject_low_conf_off_grid_repair(
    *,
    from_pause: bool,
    repair_conf: float,
    conf_thresh: float,
    old_pause_dist_ms: float,
    chosen_pause_dist_ms: float,
    shift_ms: float,
) -> bool:
    return (
        from_pause
        and repair_conf < conf_thresh
        and old_pause_dist_ms <= 350.0
        and chosen_pause_dist_ms > 350.0
        and shift_ms > 500.0
    )


def _accept_pause_backed_repair(
    *,
    from_pause: bool,
    original_pause_reason: str | None,
    old_onset_ms: float,
    chosen_onset_ms: float,
    chosen_pause_dist_ms: float,
    chosen_pause_reason: str | None,
    later_pause_grid_repair: bool = False,
) -> bool:
    if not from_pause:
        return True
    if original_pause_reason == "micro-tempo":
        return later_pause_grid_repair or chosen_onset_ms <= old_onset_ms + 500.0
    if chosen_pause_reason == "micro-tempo" and chosen_onset_ms > old_onset_ms + 500.0:
        return False
    return not (
        abs(chosen_onset_ms - old_onset_ms) > 500.0
        and chosen_pause_dist_ms > 350.0
    )


def _final_weak_current_activation_window(
    *,
    expected_ms: float,
    matched_ms: float,
    matched_pause_start_ms: float,
    matched_pause_end_ms: float,
    total_ms: float,
    prior_window_ms: float,
    min_delta_ms: float = 3000.0,
) -> tuple[float, float] | None:
    delta = matched_ms - expected_ms
    if abs(delta) <= min_delta_ms:
        return None
    slack_ms = min(3500.0, 0.25 * prior_window_ms)
    if delta > 0.0:
        start_ms = expected_ms - slack_ms
        end_ms = min(matched_pause_start_ms, expected_ms + slack_ms)
    else:
        start_ms = max(matched_pause_end_ms, expected_ms - slack_ms)
        end_ms = min(total_ms, expected_ms + slack_ms)
    if end_ms <= start_ms:
        return None
    return start_ms, end_ms


def _first_ayah_content_start(
    raw_content_start_ms: float,
    *,
    basmala: bool,
    intro_pause_start_ms: float | None,
    intro_pause_end_ms: float | None,
    min_content_start_ms: float = _BASMALA_MIN_CONTENT_START_MS,
    long_intro_pause_ms: float = 1000.0,
) -> float:
    if intro_pause_start_ms is not None and intro_pause_end_ms is not None:
        if intro_pause_end_ms - intro_pause_start_ms >= long_intro_pause_ms:
            return intro_pause_end_ms
        return 0.5 * (intro_pause_start_ms + intro_pause_end_ms)
    if basmala and raw_content_start_ms < min_content_start_ms:
        return min_content_start_ms
    return raw_content_start_ms


def _pause_match_min_keep_sim(
    *,
    action: str,
    pause_reason: str | None,
    drop_sim: float,
    min_match_sim: float,
    recover_echo_slack: float,
    final_weak_match_sim: float,
) -> float:
    if action in {"recovered", "pause_fallback"}:
        floor = max(0.0, min_match_sim - recover_echo_slack)
        if pause_reason == "micro-tempo":
            return max(floor, min_match_sim)
        return floor
    if action == "final_weak_current":
        return final_weak_match_sim
    if action in {"observed_current", "weak_observed_current"}:
        return max(0.0, min_match_sim - recover_echo_slack)
    return drop_sim


def _accept_fallback_pause_candidate(
    *,
    pause_reason: str | None,
    sim: float,
    min_match_sim: float,
    expected_delta_ms: float,
    prev_quality: float,
    next_quality: float,
    local_delta_ms: float = 5000.0,
    weak_prev_quality: float = 0.40,
    strong_next_quality: float = 0.70,
    strong_prev_quality: float = 0.70,
    weak_next_quality: float = 0.45,
    short_next_speech_ms: float = 1500.0,
    next_speech_ms: float | None = None,
) -> bool:
    if pause_reason == "micro-tempo" and sim < min_match_sim:
        internal_before_current = (
            abs(expected_delta_ms) <= local_delta_ms
            and prev_quality < weak_prev_quality
            and next_quality >= strong_next_quality
        )
        internal_after_current = (
            abs(expected_delta_ms) <= local_delta_ms
            and prev_quality >= strong_prev_quality
            and next_quality < weak_next_quality
            and next_speech_ms is not None
            and next_speech_ms <= short_next_speech_ms
        )
        return not (internal_before_current or internal_after_current)
    return True


def _section_long_pause_start_candidate(
    onset_ms: float,
    pause_start_ms: float,
    pause_end_ms: float,
    *,
    pause_reason: str | None,
    min_pause_ms: float = 1300.0,
    min_inside_ms: float = 600.0,
    max_before_end_ms: float = 350.0,
) -> float | None:
    if pause_reason != "dsp":
        return None
    if pause_end_ms - pause_start_ms < min_pause_ms:
        return None
    if pause_start_ms + min_inside_ms <= onset_ms <= pause_end_ms - max_before_end_ms:
        return None
    if pause_start_ms + min_inside_ms <= onset_ms <= pause_end_ms:
        return pause_start_ms
    return None


def _long_pause_start_timing_candidate(
    *,
    pause_duration_ms: float,
    prev_speech_ms: float,
    next_speech_ms: float,
    sim: float,
    pause_trust_sim: float,
    min_pause_ms: float = 1800.0,
    max_prev_speech_ms: float = 100.0,
    max_next_speech_ms: float = 600.0,
) -> bool:
    return (
        sim >= pause_trust_sim
        and pause_duration_ms >= min_pause_ms
        and prev_speech_ms <= max_prev_speech_ms
        and next_speech_ms <= max_next_speech_ms
    )


def _long_pause_start_after_speech_candidate(
    *,
    pause_duration_ms: float,
    prev_speech_ms: float,
    next_speech_ms: float,
    sim: float,
    min_sim: float,
    min_pause_ms: float = 1500.0,
    min_prev_speech_ms: float = 1800.0,
    max_next_speech_ms: float = 100.0,
) -> bool:
    return (
        sim >= min_sim
        and pause_duration_ms >= min_pause_ms
        and prev_speech_ms >= min_prev_speech_ms
        and next_speech_ms <= max_next_speech_ms
    )


def _long_pause_start_balanced_speech_candidate(
    *,
    pause_duration_ms: float,
    prev_speech_ms: float,
    next_speech_ms: float,
    prev_quality: float,
    next_quality: float,
    sim: float,
    min_sim: float,
    max_sim: float,
    action: str | None,
    min_pause_ms: float = 1800.0,
    min_speech_ms: float = 4000.0,
    min_quality: float = 0.55,
) -> bool:
    return (
        action == "pause_fallback"
        and min_sim <= sim < max_sim
        and pause_duration_ms >= min_pause_ms
        and prev_speech_ms >= min_speech_ms
        and next_speech_ms >= min_speech_ms
        and prev_quality >= min_quality
        and next_quality >= min_quality
    )


def _compact_dsp_start_candidate(
    *,
    pause_reason: str | None,
    pause_duration_ms: float,
    prev_quality: float,
    next_quality: float,
    sim: float,
    min_sim: float,
    action: str | None,
    min_pause_ms: float = 800.0,
    max_pause_ms: float = 1200.0,
    min_prev_quality: float = 0.75,
    min_next_quality: float = 0.60,
) -> bool:
    return (
        pause_reason == "dsp"
        and min_pause_ms <= pause_duration_ms <= max_pause_ms
        and prev_quality >= min_prev_quality
        and next_quality >= min_next_quality
        and sim >= min_sim
        and action in {"recovered", "confirmed"}
    )


def _compact_dsp_end_candidate(
    *,
    pause_reason: str | None,
    pause_duration_ms: float,
    prev_speech_ms: float,
    next_speech_ms: float,
    prev_quality: float,
    next_quality: float,
    sim: float,
    action: str | None,
    min_pause_ms: float = 800.0,
    max_pause_ms: float = 900.0,
    min_speech_ms: float = 3000.0,
    max_speech_ms: float = 4200.0,
    min_quality: float = 0.85,
    max_sim: float = 0.45,
) -> bool:
    return (
        pause_reason == "dsp"
        and action == "recovered"
        and min_pause_ms <= pause_duration_ms <= max_pause_ms
        and min_speech_ms <= prev_speech_ms <= max_speech_ms
        and min_speech_ms <= next_speech_ms <= max_speech_ms
        and prev_quality >= min_quality
        and next_quality >= min_quality
        and sim <= max_sim
    )


def _dsp_edge_timing_candidate(
    *,
    pause_reason: str | None,
    pause_duration_ms: float,
    prev_speech_ms: float,
    next_speech_ms: float,
    prev_quality: float,
    next_quality: float,
    waqf_before: int = 0,
    waqf_after: int = 0,
    sim: float,
    min_sim: float,
    action: str | None,
    min_pause_ms: float = 1100.0,
    max_pause_ms: float = 2200.0,
    min_speech_ms: float = 3000.0,
    min_quality_gap: float = 0.15,
    min_end_prev_quality: float = 0.40,
    medium_end_min_pause_ms: float = 1200.0,
    medium_end_max_pause_ms: float = 1800.0,
    medium_end_max_sim: float = 0.50,
) -> str | None:
    if (
        pause_reason != "dsp"
        or not (min_pause_ms <= pause_duration_ms <= max_pause_ms)
        or sim < min_sim
    ):
        return None
    if (
        prev_speech_ms >= min_speech_ms
        and next_speech_ms >= min_speech_ms
        and prev_quality - next_quality >= min_quality_gap
        and action in {"confirmed", "observed_current", "weak_pause_recovery"}
    ):
        return "start"
    if (
        prev_speech_ms >= 1500.0
        and next_speech_ms >= min_speech_ms
        and prev_quality >= min_end_prev_quality
        and next_quality - prev_quality >= 0.20
        and action == "recovered"
    ):
        return "end"
    if (
        action == "recovered"
        and medium_end_min_pause_ms <= pause_duration_ms <= medium_end_max_pause_ms
        and sim <= medium_end_max_sim
        and prev_quality <= 0.30
        and 0.35 <= next_quality <= 0.60
        and waqf_before > 0
        and waqf_after == 0
    ):
        return "end"
    return None


def _preword_micro_timing_candidate(
    *,
    pause_reason: str | None,
    pause_start_ms: float,
    refined_onset_ms: float | None,
    refined_conf: float,
    conf_thresh: float,
    sim: float,
    min_sim: float,
    min_before_pause_ms: float = 500.0,
) -> bool:
    return (
        pause_reason == "dsp"
        and refined_onset_ms is not None
        and refined_conf >= conf_thresh
        and sim >= min_sim
        and refined_onset_ms <= pause_start_ms - min_before_pause_ms
    )


def _weak_dsp_prior_recovery_candidate(
    *,
    pause_reason: str | None,
    action: str | None,
    prev_speech_ms: float,
    next_speech_ms: float,
    next_quality: float,
    max_prev_speech_ms: float = 500.0,
    min_next_speech_ms: float = 3000.0,
    min_next_quality: float = 0.70,
) -> bool:
    return (
        pause_reason == "weak-dsp"
        and action == "recovered"
        and prev_speech_ms <= max_prev_speech_ms
        and next_speech_ms >= min_next_speech_ms
        and next_quality >= min_next_quality
    )


def _weak_dsp_prior_trusted_recovery_candidate(
    *,
    pause_reason: str | None,
    action: str | None,
    prev_speech_ms: float,
    next_speech_ms: float,
    prev_quality: float,
    next_quality: float,
    waqf_after: int,
    max_prev_speech_ms: float = 1500.0,
    min_next_speech_ms: float = 4000.0,
    max_prev_quality: float = 0.05,
    min_next_quality: float = 0.80,
) -> bool:
    """A weak pause right after a trusted pause is likely an internal repeat."""
    return (
        pause_reason == "weak-dsp"
        and action == "recovered"
        and prev_speech_ms <= max_prev_speech_ms
        and next_speech_ms >= min_next_speech_ms
        and prev_quality <= max_prev_quality
        and next_quality >= min_next_quality
        and waqf_after > 0
    )


def _weak_dsp_wasl_recovery_candidate(
    *,
    pause_reason: str | None,
    action: str | None,
    prev_speech_ms: float,
    next_speech_ms: float,
    prev_quality: float,
    next_quality: float,
    sim: float,
    min_sim: float,
    max_sim_slack: float = 0.02,
) -> bool:
    return (
        pause_reason == "weak-dsp"
        and action == "recovered"
        and sim <= min_sim + max_sim_slack
        and 1500.0 <= prev_speech_ms <= 3500.0
        and 1500.0 <= next_speech_ms <= 3500.0
        and prev_quality <= 0.45
        and next_quality >= 0.85
    )


def _long_dsp_fractional_start_candidate(
    *,
    pause_reason: str | None,
    pause_duration_ms: float,
    prev_quality: float,
    next_quality: float,
    waqf_before: int,
    waqf_after: int,
    sim: float,
    min_sim: float,
    action: str | None,
    min_pause_ms: float = 2500.0,
    min_late_pause_ms: float = 1800.0,
    max_late_sim: float = 0.49,
    min_late_next_quality: float = 0.50,
) -> float | None:
    if pause_reason != "dsp" or sim < min_sim:
        return None
    if pause_duration_ms >= min_pause_ms:
        if action == "opening_run_repair" and prev_quality <= 0.50 and next_quality <= 0.55:
            return 0.30
        if action == "pause_fallback" and prev_quality >= 0.70 and next_quality >= 0.70:
            return 0.12
        if (
            action in {"observed_current", "recovered"}
            and pause_duration_ms >= 4500.0
            and 0.30 <= prev_quality <= 0.65
            and 0.30 <= next_quality <= 0.65
        ):
            return 0.12
        if (
            action in {"recovered", "observed_current"}
            and waqf_before > 0
            and waqf_after > 0
            and next_quality <= 0.10
            and prev_quality >= 0.60
        ):
            return 0.85
    if (
        action in {"observed_current", "recovered"}
        and pause_duration_ms >= min_late_pause_ms
        and sim <= max_late_sim
        and next_quality >= min_late_next_quality
    ):
        return 0.85
    return None


def _short_low_sim_dsp_start_candidate(
    *,
    pause_reason: str | None,
    pause_duration_ms: float,
    prev_quality: float,
    next_speech_ms: float,
    sim: float,
    action: str | None,
    max_pause_ms: float = 700.0,
    min_prev_quality: float = 0.70,
    max_next_speech_ms: float = 800.0,
    max_sim: float = 0.12,
) -> bool:
    return (
        pause_reason == "dsp"
        and pause_duration_ms <= max_pause_ms
        and action == "weak_pause_recovery"
        and prev_quality >= min_prev_quality
        and next_speech_ms <= max_next_speech_ms
        and sim <= max_sim
    )


def _late_repeated_dsp_prior_candidate(
    *,
    pause_reason: str | None,
    pause_duration_ms: float,
    prev_speech_ms: float,
    next_speech_ms: float,
    sim: float,
    action: str | None,
    min_sim: float,
    min_pause_ms: float = 1500.0,
    min_prev_speech_ms: float = 5000.0,
    max_next_speech_ms: float = 1800.0,
) -> bool:
    return (
        pause_reason == "dsp"
        and pause_duration_ms >= min_pause_ms
        and prev_speech_ms >= min_prev_speech_ms
        and next_speech_ms <= max_next_speech_ms
        and sim >= min_sim
        and action in {"observed_current", "recovered"}
    )


def _dense_prior_opening_candidate(
    previous_time: float,
    selected_time: float,
    expected_gap_ms: float,
    candidates: list[tuple[float, float, int]],
    *,
    pause_reason: str | None,
    selected_prev_quality: float,
    selected_sim: float,
    action: str | None,
    min_candidate_sim: float = 0.30,
    min_selected_sim: float = 0.40,
    max_selected_prev_quality: float = 0.12,
    min_candidate_prev_quality: float = 0.20,
    quality_margin: float = 0.15,
    max_back_ms: float = 9000.0,
    min_gap_ms: float = 2500.0,
    max_sim_loss: float = 0.25,
    min_prefix: int = 1,
) -> float | None:
    """Recover an earlier opening pause when the selected pause is internal."""
    if (
        pause_reason != "dsp"
        or action not in {"confirmed", "observed_current", "recovered"}
        or selected_sim < min_selected_sim
        or selected_prev_quality > max_selected_prev_quality
    ):
        return None
    best: tuple[float, float, float] | None = None
    for candidate_time, candidate_sim, candidate_prefix in candidates:
        if not (previous_time + min_gap_ms < candidate_time < selected_time - min_gap_ms):
            continue
        if selected_time - candidate_time > max_back_ms:
            continue
        if candidate_prefix < min_prefix:
            continue
        if candidate_sim < min_candidate_sim or candidate_sim < selected_sim - max_sim_loss:
            continue
        candidate_quality = _span_ratio_quality(candidate_time - previous_time, expected_gap_ms)
        if candidate_quality < min_candidate_prev_quality:
            continue
        if candidate_quality < selected_prev_quality + quality_margin:
            continue
        if best is None or candidate_quality > best[0] or (
            abs(candidate_quality - best[0]) <= 1e-9
            and (candidate_sim > best[1] or (
                abs(candidate_sim - best[1]) <= 1e-9 and candidate_time > best[2]
            ))
        ):
            best = (candidate_quality, candidate_sim, candidate_time)
    return best[2] if best is not None else None


def _dense_later_opening_candidate(
    previous_time: float,
    selected_time: float,
    next_time: float,
    expected_prev_gap_ms: float,
    expected_next_gap_ms: float,
    candidates: list[tuple[float, float, int]],
    *,
    pause_reason: str | None,
    selected_sim: float,
    action: str | None,
    min_candidate_sim: float = 0.45,
    sim_margin: float = 0.15,
    min_prefix: int = 3,
    max_forward_ms: float = 9000.0,
    min_gap_ms: float = 2500.0,
    max_selected_pair_quality: float = 0.35,
    min_candidate_pair_quality: float = 0.70,
    quality_margin: float = 0.35,
) -> float | None:
    """Move a boundary forward when the next pause explains the local pair."""
    if pause_reason != "dsp" or action not in {"confirmed", "observed_current", "recovered"}:
        return None
    selected_prev = _span_ratio_quality(selected_time - previous_time, expected_prev_gap_ms)
    selected_next = _span_ratio_quality(next_time - selected_time, expected_next_gap_ms)
    selected_pair = min(selected_prev, selected_next)
    if selected_pair > max_selected_pair_quality:
        return None
    best: tuple[float, float, float] | None = None
    for candidate_time, candidate_sim, candidate_prefix in candidates:
        if not (selected_time + min_gap_ms < candidate_time < next_time - min_gap_ms):
            continue
        if candidate_time - selected_time > max_forward_ms:
            continue
        if candidate_prefix < min_prefix:
            continue
        if candidate_sim < min_candidate_sim or candidate_sim < selected_sim + sim_margin:
            continue
        candidate_prev = _span_ratio_quality(candidate_time - previous_time, expected_prev_gap_ms)
        candidate_next = _span_ratio_quality(next_time - candidate_time, expected_next_gap_ms)
        candidate_pair = min(candidate_prev, candidate_next)
        candidate_avg = 0.5 * (candidate_prev + candidate_next)
        selected_avg = 0.5 * (selected_prev + selected_next)
        if candidate_pair < min_candidate_pair_quality:
            continue
        if candidate_pair < selected_pair + quality_margin:
            continue
        if candidate_avg < selected_avg + quality_margin:
            continue
        if best is None or candidate_pair > best[0] or (
            abs(candidate_pair - best[0]) <= 1e-9
            and (candidate_sim > best[1] or (
                abs(candidate_sim - best[1]) <= 1e-9 and candidate_time < best[2]
            ))
        ):
            best = (candidate_pair, candidate_sim, candidate_time)
    return best[2] if best is not None else None


def _weak_dsp_later_prefix_candidate(
    previous_time: float,
    selected_time: float,
    next_time: float,
    expected_prev_gap_ms: float,
    expected_next_gap_ms: float,
    candidates: list[tuple[float, float, int]],
    *,
    pause_reason: str | None,
    selected_sim: float,
    selected_prefix: int,
    action: str | None,
    min_candidate_sim: float = 0.50,
    sim_margin: float = 0.15,
    max_forward_ms: float = 5000.0,
    min_gap_ms: float = 2500.0,
    max_selected_pair_quality: float = 0.35,
    min_candidate_prev_quality: float = 0.25,
    min_candidate_next_quality: float = 0.80,
    min_avg_gain: float = 0.15,
) -> float | None:
    """Move an early weak pause to a later prefix-backed real pause."""
    if (
        pause_reason != "weak-dsp"
        or action not in {"confirmed", "observed_current", "recovered"}
    ):
        return None
    _ = selected_prefix
    selected_prev = _span_ratio_quality(selected_time - previous_time, expected_prev_gap_ms)
    selected_next = _span_ratio_quality(next_time - selected_time, expected_next_gap_ms)
    selected_pair = min(selected_prev, selected_next)
    selected_avg = 0.5 * (selected_prev + selected_next)
    if selected_pair > max_selected_pair_quality:
        return None
    best: tuple[float, float, float] | None = None
    for candidate_time, candidate_sim, candidate_prefix in candidates:
        if not (selected_time + min_gap_ms < candidate_time < next_time - min_gap_ms):
            continue
        if candidate_time - selected_time > max_forward_ms:
            continue
        _ = candidate_prefix
        if candidate_sim < min_candidate_sim or candidate_sim < selected_sim + sim_margin:
            continue
        candidate_prev = _span_ratio_quality(candidate_time - previous_time, expected_prev_gap_ms)
        candidate_next = _span_ratio_quality(next_time - candidate_time, expected_next_gap_ms)
        candidate_avg = 0.5 * (candidate_prev + candidate_next)
        if candidate_prev < min_candidate_prev_quality:
            continue
        if candidate_next < min_candidate_next_quality:
            continue
        if candidate_avg < selected_avg + min_avg_gain:
            continue
        if best is None or candidate_next > best[0] or (
            abs(candidate_next - best[0]) <= 1e-9
            and (candidate_sim > best[1] or (
                abs(candidate_sim - best[1]) <= 1e-9 and candidate_time < best[2]
            ))
        ):
            best = (candidate_next, candidate_sim, candidate_time)
    return best[2] if best is not None else None


def _missing_wasl_pause_candidate(
    *,
    rough_ms: float,
    expected_ms: float,
    previous_time: float | None,
    next_time: float | None,
    candidates: list[float],
    max_expected_dist_ms: float = 5000.0,
    min_expected_gain_ms: float = 1800.0,
    max_rough_dist_ms: float = 4500.0,
    min_gap_ms: float = 2500.0,
) -> float | None:
    """Recover a visible pause for a missing-before-future wasl boundary."""
    best: tuple[float, float] | None = None
    rough_expected_dist = abs(rough_ms - expected_ms)
    for candidate_time in candidates:
        if previous_time is not None and candidate_time <= previous_time + min_gap_ms:
            continue
        if next_time is not None and candidate_time >= next_time - min_gap_ms:
            continue
        if abs(candidate_time - rough_ms) > max_rough_dist_ms:
            continue
        candidate_expected_dist = abs(candidate_time - expected_ms)
        if candidate_expected_dist > max_expected_dist_ms:
            continue
        if candidate_expected_dist > rough_expected_dist - min_expected_gain_ms:
            continue
        if best is None or candidate_expected_dist < best[0] or (
            abs(candidate_expected_dist - best[0]) <= 1e-9
            and abs(candidate_time - rough_ms) < abs(best[1] - rough_ms)
        ):
            best = (candidate_expected_dist, candidate_time)
    return best[1] if best is not None else None


def _dense_later_final_opening_candidate(
    previous_time: float,
    selected_time: float,
    expected_gap_ms: float,
    candidates: list[tuple[float, float, int]],
    *,
    pause_reason: str | None,
    selected_sim: float,
    action: str | None,
    min_candidate_sim: float = 0.45,
    sim_margin: float = 0.15,
    min_prefix: int = 3,
    max_forward_ms: float = 9000.0,
    min_gap_ms: float = 2500.0,
    max_selected_quality: float = 0.35,
    min_candidate_quality: float = 0.65,
    quality_margin: float = 0.35,
) -> float | None:
    """Move the final ayah forward when a later pause is the clearer opening."""
    if pause_reason != "dsp" or action not in {"confirmed", "observed_current", "recovered"}:
        return None
    selected_quality = _span_ratio_quality(selected_time - previous_time, expected_gap_ms)
    if selected_quality > max_selected_quality:
        return None
    best: tuple[float, float, float] | None = None
    for candidate_time, candidate_sim, candidate_prefix in candidates:
        if not (selected_time + min_gap_ms < candidate_time):
            continue
        if candidate_time - selected_time > max_forward_ms:
            continue
        if candidate_prefix < min_prefix:
            continue
        if candidate_sim < min_candidate_sim or candidate_sim < selected_sim + sim_margin:
            continue
        candidate_quality = _span_ratio_quality(candidate_time - previous_time, expected_gap_ms)
        if candidate_quality < min_candidate_quality:
            continue
        if candidate_quality < selected_quality + quality_margin:
            continue
        if best is None or candidate_quality > best[0] or (
            abs(candidate_quality - best[0]) <= 1e-9
            and (candidate_sim > best[1] or (
                abs(candidate_sim - best[1]) <= 1e-9 and candidate_time < best[2]
            ))
        ):
            best = (candidate_quality, candidate_sim, candidate_time)
    return best[2] if best is not None else None


def _left_edge_repeated_opening_candidate(
    candidates: list[tuple[float, float, bool]],
    *,
    pause_start_ms: float,
    pause_center_ms: float,
    min_conf: float,
    min_shift_ms: float = 500.0,
) -> tuple[float, float] | None:
    """Use a high-confidence left-edge read only as proof inside one pause shelf."""
    accepted = [
        (onset_ms, conf)
        for onset_ms, conf, is_left_edge in candidates
        if is_left_edge
        and conf >= min_conf
        and pause_start_ms <= onset_ms <= pause_center_ms - min_shift_ms
    ]
    if not accepted:
        return None
    return max(accepted, key=lambda item: item[0])


def _prior_close_dsp_before_micro_stop(
    micro_time_ms: float,
    strong_candidate_times: list[float],
    *,
    max_before_ms: float = 900.0,
) -> float | None:
    candidates = [
        candidate
        for candidate in strong_candidate_times
        if 0.0 < micro_time_ms - candidate <= max_before_ms
    ]
    return max(candidates, default=None)


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
            d.pause for d in pause_decisions
            if d.reason == "intro-vad-tight"
            and d.pause.start_ms <= content_start <= d.pause.end_ms
        ),
        None,
    )
    content_start = _first_ayah_content_start(
        content_start,
        basmala=basmala,
        intro_pause_start_ms=intro_pause.start_ms if intro_pause is not None else None,
        intro_pause_end_ms=intro_pause.end_ms if intro_pause is not None else None,
    )
    if intro_pause is not None:
        emit(
            f"Первый аят: anchor {sec(raw_content_start)} сек. попал внутрь intro-паузы "
            f"{sec(intro_pause.start_ms)}-{sec(intro_pause.end_ms)} сек.; "
            f"ставлю старт {sec(content_start)} сек."
        )
    elif basmala and content_start < _BASMALA_MIN_CONTENT_START_MS:
        emit(
            f"Первый аят: anchor {sec(content_start)} сек. раньше нижней границы "
            f"басмалы {sec(_BASMALA_MIN_CONTENT_START_MS)} сек.; сдвигаю старт"
        )
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
                right_pause = pause_by_center.get(hi_t)
                expected_t = float(ev.get("expected_ms", exp[start_i]))
                window = _final_weak_current_activation_window(
                    expected_ms=expected_t,
                    matched_ms=hi_t,
                    matched_pause_start_ms=right_pause.start_ms if right_pause is not None else hi_t,
                    matched_pause_end_ms=right_pause.end_ms if right_pause is not None else hi_t,
                    total_ms=total_ms,
                    prior_window_ms=_MATCH_PRIOR_WINDOW_MS,
                )
                if window is None:
                    continue
                interval_start, interval_end = window
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
                if valley is not None:
                    valley = _prefer_expected_micro_valley(
                        valley,
                        _find_micro_pause_valley(
                            pcm,
                            sr,
                            interval_start_ms=interval_start,
                            interval_end_ms=interval_end,
                            expected_ms=expected_t,
                            search_ms=_MICRO_EXPECTED_RETRY_MS,
                        ),
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
                if valley is not None:
                    valley = _prefer_expected_micro_valley(
                        valley,
                        _find_micro_pause_valley(
                            pcm,
                            sr,
                            interval_start_ms=interval_start,
                            interval_end_ms=interval_end,
                            expected_ms=expected_t,
                            search_ms=_MICRO_EXPECTED_RETRY_MS,
                        ),
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
            if action in {"recovered", "pause_fallback"}:
                pause = pause_by_center.get(mk)
                diag = pause_diag.get(pause) if pause is not None else None
                pause_reason = diag.reason if diag is not None else None
            else:
                pause_reason = None
            min_keep_sim = _pause_match_min_keep_sim(
                action=action,
                pause_reason=pause_reason,
                drop_sim=drop_sim,
                min_match_sim=_MIN_MATCH_SIM,
                recover_echo_slack=_RECOVER_ECHO_SIM_SLACK,
                final_weak_match_sim=_FINAL_WEAK_MATCH_SIM,
            )
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

    def text_gap_plausible(observed_gap_ms: float, expected_gap_ms: float) -> bool:
        if expected_gap_ms <= 0.0:
            return True
        return observed_gap_ms <= max(45_000.0, 12.0 * expected_gap_ms)

    def preserves_neighbor_text_span(
        k: int,
        center: float,
        prev_i: int | None,
        prev_t: float | None,
        next_i: int | None,
        next_t: float | None,
    ) -> bool:
        if prev_i is not None and prev_t is not None:
            if not text_gap_plausible(center - prev_t, exp[k] - exp[prev_i]):
                return False
        if next_i is not None and next_t is not None:
            if not text_gap_plausible(next_t - center, exp[next_i] - exp[k]):
                return False
        return True

    def dense_pause_block_before(
        k: int,
        j: int,
        prev_i: int | None,
        prev_t: float | None,
        next_i: int | None,
    ) -> bool:
        if prev_i is None or prev_t is None or next_i is None:
            return False
        if not (k + 1 < next_i):
            return False
        selected_center = pause_onsets[j]
        block_candidates: list[int] = []
        for jj, center in enumerate(pause_onsets):
            if jj == j:
                block_candidates.append(jj)
                continue
            if center in used_matched or center <= prev_t or center >= selected_center:
                continue
            pause = pause_by_center.get(center)
            diag = pause_diag.get(pause) if pause is not None else None
            if diag is None or diag.reason not in {"dsp", "weak-dsp"}:
                continue
            block_candidates.append(jj)
        return _skips_dense_pause_block(
            k,
            j,
            block_candidates,
            candidate_time=lambda jj: pause_onsets[jj],
            expected_time=lambda kk: exp[kk],
            previous_boundary=prev_i,
            previous_time=prev_t,
            next_boundary=next_i,
        )

    for k, mk in enumerate(matched):
        if mk is not None or sequence_reason.get(k) != "missing_before_future":
            continue
        prev_i = next((ii for ii in range(k - 1, -1, -1) if matched[ii] is not None), None)
        next_i = next((ii for ii in range(k + 1, len(matched)) if matched[ii] is not None), None)
        prev_t = matched[prev_i] if prev_i is not None else None
        next_t = matched[next_i] if next_i is not None else None
        if prev_t is None or next_t is None:
            continue
        candidates: list[tuple[float, int]] = []
        for j, center in enumerate(pause_onsets):
            if center in used_matched or center <= prev_t or center >= next_t:
                continue
            if not preserves_neighbor_text_span(k, center, prev_i, prev_t, next_i, next_t):
                continue
            if dense_pause_block_before(k, j, prev_i, prev_t, next_i):
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
        prev_i = next((ii for ii in range(k - 1, -1, -1) if matched[ii] is not None), None)
        next_i = next((ii for ii in range(k + 1, len(matched)) if matched[ii] is not None), None)
        prev_t = matched[prev_i] if prev_i is not None else None
        next_t = matched[next_i] if next_i is not None else None
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
            if not preserves_neighbor_text_span(k, center, prev_i, prev_t, next_i, next_t):
                continue
            if dense_pause_block_before(k, j, prev_i, prev_t, next_i):
                continue
            pause = pause_by_center.get(center)
            diag = pause_diag.get(pause) if pause is not None else None
            if diag is None:
                continue
            s = _sim_ij(k, j)
            wave_metrics = wave_tempo_metrics(k, j)
            if not _accept_fallback_pause_candidate(
                pause_reason=diag.reason,
                sim=s,
                min_match_sim=_MIN_MATCH_SIM,
                expected_delta_ms=center - exp[k],
                prev_quality=float(wave_metrics["prev_quality"]),
                next_quality=float(wave_metrics["next_quality"]),
                next_speech_ms=float(wave_metrics["next_speech_ms"]),
            ):
                continue
            if diag.reason != "weak-dsp" and abs(center - rough_t) <= 2200.0:
                near_strong_candidates.append(j)
                continue
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

    def opening_run_candidate_score(kk: int, j: int) -> float | None:
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

    repaired_run_until = -1

    def repair_opening_run(
        start_k: int,
        *,
        include_start: bool = False,
        after_k: int | None = None,
        after_ms_override: float | None = None,
        max_length: int = 9,
        min_length: int = 4,
    ) -> bool:
        nonlocal repaired_run_until
        max_end_k = min(len(matched) - 1, start_k + max_length)
        if max_end_k <= start_k or matched[start_k] is None:
            return False
        owner_by_center = {
            center: owner
            for owner, center in enumerate(matched)
            if center is not None
        }
        start_t = float(matched[start_k])
        first_k = start_k if include_start else start_k + 1
        if first_k > max_end_k:
            return False
        anchor_t = float(after_ms_override) if after_ms_override is not None else start_t
        anchor_expected = exp[after_k] if after_k is not None else exp[start_k]
        min_end_k = min(max_end_k, first_k + min_length - 1)
        old_end = max(
            (
                float(matched[ii])
                for ii in range(first_k, max_end_k + 1)
                if matched[ii] is not None
            ),
            default=start_t,
        )
        block_end = max(old_end, anchor_t + 180_000.0)

        assignment: dict[int, int] = {}
        run_boundaries: list[int] = []
        for end_k in range(max_end_k, min_end_k - 1, -1):
            current_boundaries = list(range(first_k, end_k + 1))
            run_owners = set(current_boundaries)
            candidate_pool = [
                j
                for j, center in enumerate(pause_onsets)
                if center < block_end
                and (
                    center not in used_matched
                    or owner_by_center.get(center) in run_owners
                )
            ]
            current_assignment = _assign_opening_block(
                current_boundaries,
                candidate_pool,
                candidate_time=lambda j: pause_onsets[j],
                score=opening_run_candidate_score,
                after_ms=anchor_t,
                expected_time=lambda kk: exp[kk],
                after_expected_ms=anchor_expected,
            )
            if set(current_assignment) == set(current_boundaries):
                assignment = current_assignment
                run_boundaries = current_boundaries
                break
        if not assignment:
            return False

        repaired_run_until = max(repaired_run_until, run_boundaries[-1])
        for kk in run_boundaries:
            j = assignment[kk]
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
        return True

    for k, mk in enumerate(matched):
        if mk is None:
            continue
        prev_pair = next(
            ((ii, matched[ii]) for ii in range(k - 1, -1, -1) if matched[ii] is not None),
            None,
        )
        if prev_pair is None:
            continue
        prev_i, prev_t = prev_pair
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
                has_strong_current_candidate = any(
                    (opening_run_candidate_score(k, j) or 0.0) >= 0.8 for j in candidates
                )
                if (
                    not has_strong_current_candidate
                    and repair_opening_run(
                        k,
                        include_start=True,
                        after_k=prev_i,
                        after_ms_override=float(prev_t),
                    )
                ):
                    continue
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

    micro_shelf_start_by_center = {
        float(item["center_ms"]): float(item["expanded_start_ms"])
        for item in micro_pause_debug
        if item.get("status") in {"added", "tail_added"}
        and item.get("center_ms") is not None
        and item.get("expanded_start_ms") is not None
    }

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
            if pause is not None:
                previous_a = ayah_list[k]
                previous_place = place.get(previous_a)
                previous_time = (
                    previous_place[0]
                    if previous_place is not None
                    else starts_ms.get(previous_a, content_start)
                )
                expected_gap = (
                    (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                    if total_w > 0.0
                    else 0.0
                )
                prior_pause = _dense_prior_opening_candidate(
                    previous_time,
                    matched[k],
                    expected_gap,
                    [
                        (
                            center,
                            _sim_ij(k, candidate_j),
                            _onset_prefix_len(decode_pause(candidate_j), openings[k]),
                        )
                        for candidate_j, center in enumerate(pause_onsets)
                        if center not in used_matched
                        and (candidate := pause_by_center.get(center)) is not None
                        and (diag := pause_diag.get(candidate)) is not None
                        and diag.reason == "dsp"
                    ],
                    pause_reason=pause_reason,
                    selected_prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                    selected_sim=sim,
                    action=sequence_action.get(k),
                )
                if prior_pause is not None:
                    old_matched = matched[k]
                    matched[k] = prior_pause
                    used_matched.discard(old_matched)
                    used_matched.add(prior_pause)
                    j = pause_onsets.index(prior_pause)
                    pause = pause_by_center.get(prior_pause)
                    pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                    sim = _sim_ij(k, j)
                    wave_metrics = rounded_wave_metrics(k, j)
                    rough = pause_timing_point(prior_pause, sim=sim, reason=pause_reason)
                    timing_adjusted = True
            if pause is not None:
                next_i = next(
                    (ii for ii in range(k + 1, len(matched)) if matched[ii] is not None),
                    None,
                )
                previous_a = ayah_list[k]
                previous_place = place.get(previous_a)
                previous_time = (
                    previous_place[0]
                    if previous_place is not None
                    else starts_ms.get(previous_a, content_start)
                )
                if next_i is not None:
                    next_a = inner[next_i]
                    next_time = float(matched[next_i])
                    later_pause = _weak_dsp_later_prefix_candidate(
                        previous_time,
                        matched[k],
                        next_time,
                        (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0,
                        (cum_u[next_a] - cum_u[a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0,
                        [
                            (
                                center,
                                _sim_ij(k, candidate_j),
                                _onset_prefix_len(decode_pause(candidate_j), openings[k]),
                            )
                            for candidate_j, center in enumerate(pause_onsets)
                            if not any(
                                other_i != k and matched[other_i] == center
                                for other_i in range(len(matched))
                            )
                            and (candidate := pause_by_center.get(center)) is not None
                            and (diag := pause_diag.get(candidate)) is not None
                            and diag.reason == "dsp"
                        ],
                        pause_reason=pause_reason,
                        selected_sim=sim,
                        selected_prefix=_onset_prefix_len(decode_pause(j), openings[k]),
                        action=sequence_action.get(k),
                    )
                    if later_pause is not None:
                        old_matched = matched[k]
                        matched[k] = later_pause
                        used_matched.discard(old_matched)
                        used_matched.add(later_pause)
                        j = pause_onsets.index(later_pause)
                        pause = pause_by_center.get(later_pause)
                        pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                        sim = _sim_ij(k, j)
                        wave_metrics = rounded_wave_metrics(k, j)
                        rough = pause_timing_point(later_pause, sim=sim, reason=pause_reason)
                        timing_adjusted = True
                    later_pause = _dense_later_opening_candidate(
                        previous_time,
                        matched[k],
                        next_time,
                        (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0,
                        (cum_u[next_a] - cum_u[a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0,
                        [
                            (
                                center,
                                _sim_ij(k, candidate_j),
                                _onset_prefix_len(decode_pause(candidate_j), openings[k]),
                            )
                            for candidate_j, center in enumerate(pause_onsets)
                            if center not in used_matched
                            and (candidate := pause_by_center.get(center)) is not None
                            and (diag := pause_diag.get(candidate)) is not None
                            and diag.reason == "dsp"
                        ],
                        pause_reason=pause_reason,
                        selected_sim=sim,
                        action=sequence_action.get(k),
                    )
                    if later_pause is not None:
                        old_matched = matched[k]
                        matched[k] = later_pause
                        used_matched.discard(old_matched)
                        used_matched.add(later_pause)
                        j = pause_onsets.index(later_pause)
                        pause = pause_by_center.get(later_pause)
                        pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                        sim = _sim_ij(k, j)
                        wave_metrics = rounded_wave_metrics(k, j)
                        rough = pause_timing_point(later_pause, sim=sim, reason=pause_reason)
                        timing_adjusted = True
                else:
                    later_pause = _dense_later_final_opening_candidate(
                        previous_time,
                        matched[k],
                        (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0,
                        [
                            (
                                center,
                                _sim_ij(k, candidate_j),
                                _onset_prefix_len(decode_pause(candidate_j), openings[k]),
                            )
                            for candidate_j, center in enumerate(pause_onsets)
                            if center not in used_matched
                            and (candidate := pause_by_center.get(center)) is not None
                            and (diag := pause_diag.get(candidate)) is not None
                            and diag.reason == "dsp"
                        ],
                        pause_reason=pause_reason,
                        selected_sim=sim,
                        action=sequence_action.get(k),
                    )
                    if later_pause is not None:
                        old_matched = matched[k]
                        matched[k] = later_pause
                        used_matched.discard(old_matched)
                        used_matched.add(later_pause)
                        j = pause_onsets.index(later_pause)
                        pause = pause_by_center.get(later_pause)
                        pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                        sim = _sim_ij(k, j)
                        wave_metrics = rounded_wave_metrics(k, j)
                        rough = pause_timing_point(later_pause, sim=sim, reason=pause_reason)
                        timing_adjusted = True
            if (
                pause is not None
                and _late_repeated_dsp_prior_candidate(
                    pause_reason=pause_reason,
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    sim=sim,
                    action=sequence_action.get(k),
                    min_sim=_PAUSE_TRUST_SIM,
                )
            ):
                previous_a = ayah_list[k]
                previous_place = place.get(previous_a)
                previous_time = (
                    previous_place[0]
                    if previous_place is not None
                    else starts_ms.get(previous_a, content_start)
                )
                expected_gap = (
                    (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                    if total_w > 0.0
                    else 0.0
                )
                prior_pause = _prior_pause_before_internal_stop(
                    previous_time,
                    pause.end_ms,
                    expected_gap,
                    [
                        center
                        for center in pause_onsets
                        if center not in used_matched
                        and (candidate := pause_by_center.get(center)) is not None
                        and (diag := pause_diag.get(candidate)) is not None
                        and diag.reason == "dsp"
                    ],
                    raw_is_internal_stop=True,
                )
                if prior_pause is not None:
                    old_matched = matched[k]
                    matched[k] = prior_pause
                    used_matched.discard(old_matched)
                    used_matched.add(prior_pause)
                    j = pause_onsets.index(prior_pause)
                    pause = pause_by_center.get(prior_pause)
                    pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                    sim = _sim_ij(k, j)
                    wave_metrics = rounded_wave_metrics(k, j)
                    rough = pause_timing_point(prior_pause, sim=sim, reason=pause_reason)
                    timing_adjusted = True
            if _weak_dsp_wasl_recovery_candidate(
                pause_reason=pause_reason,
                action=sequence_action.get(k),
                prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                next_quality=float(wave_metrics.get("next_quality") or 0.0),
                sim=sim,
                min_sim=_MIN_MATCH_SIM,
            ):
                used_matched.discard(matched[k])
                matched[k] = None
                rough = _interp_wasl(k, exp, matched)
                place[a] = (rough, False)
                pause_match_debug.append({
                    "ayah": a,
                    "expected_ms": round(expected, 1),
                    "rough_ms": round(rough, 1),
                    "from_pause": False,
                    "pause": pause_ord.get(pause_onsets[j]),
                    "pause_center_ms": round(pause_onsets[j], 1),
                    "pause_end_ms": round(pause.end_ms, 1) if pause is not None else None,
                    "pause_sim": round(sim, 3),
                    "drop_reason": "weak_dsp_wasl_recovery",
                    "match_action": sequence_action.get(k),
                    "pause_reason": pause_reason,
                    **wave_metrics,
                })
                continue
            if pause_reason == "micro-tempo" and sim < _CONFIRM_MATCH_SIM:
                prior_dsp = _prior_close_dsp_before_micro_stop(
                    matched[k],
                    [
                        center
                        for center in pause_onsets
                        if center not in used_matched
                        and (candidate := pause_by_center.get(center)) is not None
                        and (diag := pause_diag.get(candidate)) is not None
                        and diag.reason == "dsp"
                    ],
                )
                if prior_dsp is not None:
                    old_matched = matched[k]
                    matched[k] = prior_dsp
                    used_matched.discard(old_matched)
                    used_matched.add(prior_dsp)
                    j = pause_onsets.index(prior_dsp)
                    pause = pause_by_center.get(prior_dsp)
                    pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                    sim = _sim_ij(k, j)
                    wave_metrics = rounded_wave_metrics(k, j)
                    rough = pause_timing_point(prior_dsp, sim=sim, reason=pause_reason)
                    timing_adjusted = True
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
            micro_shelf_start = _micro_shelf_start_candidate(
                pause_reason=pause_reason,
                sim=sim,
                prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                next_quality=float(wave_metrics.get("next_quality") or 0.0),
                center_ms=matched[k],
                shelf_start_ms=micro_shelf_start_by_center.get(round(matched[k], 1)),
                confirm_match_sim=_CONFIRM_MATCH_SIM,
            )
            if micro_shelf_start is not None:
                rough = micro_shelf_start
                timing_adjusted = True
            if (
                pause_reason in {"rejected", "weak-dsp"}
                and sequence_action.get(k) == "recovered"
            ):
                previous_a = ayah_list[k]
                previous_place = place.get(previous_a)
                previous_time = (
                    previous_place[0]
                    if previous_place is not None
                    else starts_ms.get(previous_a, content_start)
                )
                next_i = next(
                    (ii for ii in range(k + 1, len(matched)) if matched[ii] is not None),
                    None,
                )
                if next_i is not None:
                    next_a = inner[next_i]
                    next_time = float(matched[next_i])
                    expected_prev_gap = (
                        (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0
                    )
                    expected_next_gap = (
                        (cum_u[next_a] - cum_u[a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0
                    )
                    interval_start = max(previous_time + 300.0, rough - 3500.0)
                    interval_end = min(next_time - 300.0, rough + 3500.0)
                    valley = (
                        _find_micro_pause_valley(
                            pcm,
                            sr,
                            interval_start_ms=interval_start,
                            interval_end_ms=interval_end,
                            expected_ms=previous_time + expected_prev_gap,
                        )
                        if interval_end - interval_start >= 700.0
                        else None
                    )
                    if valley is not None:
                        valley_center = _micro_valley_center(valley)
                        selected_prev_quality = _span_ratio_quality(
                            rough - previous_time,
                            expected_prev_gap,
                        )
                        selected_next_quality = _span_ratio_quality(
                            next_time - rough,
                            expected_next_gap,
                        )
                        candidate_prev_quality = _span_ratio_quality(
                            valley_center - previous_time,
                            expected_prev_gap,
                        )
                        candidate_next_quality = _span_ratio_quality(
                            next_time - valley_center,
                            expected_next_gap,
                        )
                        local_expected = previous_time + expected_prev_gap
                        if _rejected_micro_valley_candidate(
                            pause_reason=pause_reason,
                            action=sequence_action.get(k),
                            valley_ratio=valley[2],
                            shift_ms=abs(valley_center - rough),
                            selected_prev_quality=selected_prev_quality,
                            selected_next_quality=selected_next_quality,
                            candidate_prev_quality=candidate_prev_quality,
                            candidate_next_quality=candidate_next_quality,
                        ):
                            rough = valley_center
                            timing_adjusted = True
                        elif _weak_dsp_expected_micro_valley_candidate(
                            pause_reason=pause_reason,
                            action=sequence_action.get(k),
                            pause_sim=sim,
                            valley_ratio=valley[2],
                            shift_ms=abs(valley_center - rough),
                            expected_gain_ms=abs(rough - local_expected)
                            - abs(valley_center - local_expected),
                            selected_prev_quality=selected_prev_quality,
                            selected_next_quality=selected_next_quality,
                            candidate_prev_quality=candidate_prev_quality,
                            candidate_next_quality=candidate_next_quality,
                        ):
                            rough = valley_center
                            timing_adjusted = True
                        elif (
                            pause is not None
                            and pause_reason == "weak-dsp"
                            and sequence_action.get(k) == "recovered"
                        ):
                            post_interval_start = max(
                                pause.end_ms + 80.0,
                                rough + 120.0,
                                previous_time + 300.0,
                            )
                            post_interval_end = min(next_time - 300.0, rough + 3500.0)
                            post_valley = (
                                _find_micro_pause_valley(
                                    pcm,
                                    sr,
                                    interval_start_ms=post_interval_start,
                                    interval_end_ms=post_interval_end,
                                    expected_ms=local_expected,
                                    search_ms=2500.0,
                                )
                                if post_interval_end - post_interval_start >= 700.0
                                else None
                            )
                            if post_valley is not None:
                                post_valley_center = _micro_valley_center(post_valley)
                                post_candidate_prev_quality = _span_ratio_quality(
                                    post_valley_center - previous_time,
                                    expected_prev_gap,
                                )
                                post_candidate_next_quality = _span_ratio_quality(
                                    next_time - post_valley_center,
                                    expected_next_gap,
                                )
                                if _weak_dsp_post_selected_micro_valley_candidate(
                                    pause_reason=pause_reason,
                                    action=sequence_action.get(k),
                                    pause_sim=sim,
                                    valley_ratio=post_valley[2],
                                    forward_shift_ms=post_valley_center - rough,
                                    selected_prev_quality=selected_prev_quality,
                                    selected_next_quality=selected_next_quality,
                                    candidate_prev_quality=post_candidate_prev_quality,
                                    candidate_next_quality=post_candidate_next_quality,
                                ):
                                    rough = post_valley_center
                                    timing_adjusted = True
            if (
                pause is not None
                and _weak_dsp_prior_recovery_candidate(
                    pause_reason=pause_reason,
                    action=sequence_action.get(k),
                    prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    next_quality=float(wave_metrics.get("next_quality") or 0.0),
                )
            ):
                prior_dsp = _prior_close_dsp_before_micro_stop(
                    matched[k],
                    [
                        center
                        for center in pause_onsets
                        if center not in used_matched
                        and (candidate := pause_by_center.get(center)) is not None
                        and (diag := pause_diag.get(candidate)) is not None
                        and diag.reason == "dsp"
                    ],
                )
                if prior_dsp is not None:
                    old_matched = matched[k]
                    matched[k] = prior_dsp
                    used_matched.discard(old_matched)
                    used_matched.add(prior_dsp)
                    j = pause_onsets.index(prior_dsp)
                    pause = pause_by_center.get(prior_dsp)
                    pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                    sim = _sim_ij(k, j)
                    wave_metrics = rounded_wave_metrics(k, j)
                    rough = pause_timing_point(prior_dsp, sim=sim, reason=pause_reason)
                    timing_adjusted = True
            if (
                pause is not None
                and _weak_dsp_prior_trusted_recovery_candidate(
                    pause_reason=pause_reason,
                    action=sequence_action.get(k),
                    prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                    next_quality=float(wave_metrics.get("next_quality") or 0.0),
                    waqf_after=int(wave_metrics.get("waqf_after") or 0),
                )
            ):
                prior_dsp = _prior_close_dsp_before_micro_stop(
                    matched[k],
                    [
                        center
                        for center in pause_onsets
                        if center not in used_matched
                        and (candidate := pause_by_center.get(center)) is not None
                        and (diag := pause_diag.get(candidate)) is not None
                        and diag.reason == "dsp"
                    ],
                    max_before_ms=1800.0,
                )
                if prior_dsp is not None:
                    old_matched = matched[k]
                    matched[k] = prior_dsp
                    used_matched.discard(old_matched)
                    used_matched.add(prior_dsp)
                    j = pause_onsets.index(prior_dsp)
                    pause = pause_by_center.get(prior_dsp)
                    pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                    sim = _sim_ij(k, j)
                    wave_metrics = rounded_wave_metrics(k, j)
                    rough = pause_timing_point(prior_dsp, sim=sim, reason=pause_reason)
                    timing_adjusted = True
            if (
                pause is not None
                and pause_reason == "dsp"
            ):
                compact_pause_start = _compact_dsp_start_candidate(
                    pause_reason=pause_reason,
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                    next_quality=float(wave_metrics.get("next_quality") or 0.0),
                    sim=sim,
                    min_sim=_MIN_MATCH_SIM,
                    action=sequence_action.get(k),
                )
                compact_pause_end = _compact_dsp_end_candidate(
                    pause_reason=pause_reason,
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                    next_quality=float(wave_metrics.get("next_quality") or 0.0),
                    sim=sim,
                    action=sequence_action.get(k),
                )
                long_pause_start_timing = _long_pause_start_timing_candidate(
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    sim=sim,
                    pause_trust_sim=_PAUSE_TRUST_SIM,
                ) or _long_pause_start_after_speech_candidate(
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    sim=sim,
                    min_sim=_RECOVER_MATCH_SIM,
                )
                long_pause_balanced_speech = _long_pause_start_balanced_speech_candidate(
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                    next_quality=float(wave_metrics.get("next_quality") or 0.0),
                    sim=sim,
                    min_sim=_RECOVER_MATCH_SIM,
                    max_sim=_PAUSE_TRUST_SIM,
                    action=sequence_action.get(k),
                )
                dsp_edge_timing = _dsp_edge_timing_candidate(
                    pause_reason=pause_reason,
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_speech_ms=float(wave_metrics.get("prev_speech_ms") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                    next_quality=float(wave_metrics.get("next_quality") or 0.0),
                    waqf_before=int(wave_metrics.get("waqf_before") or 0),
                    waqf_after=int(wave_metrics.get("waqf_after") or 0),
                    sim=sim,
                    min_sim=_RECOVER_MATCH_SIM,
                    action=sequence_action.get(k),
                )
                long_dsp_fraction = _long_dsp_fractional_start_candidate(
                    pause_reason=pause_reason,
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                    next_quality=float(wave_metrics.get("next_quality") or 0.0),
                    waqf_before=int(wave_metrics.get("waqf_before") or 0),
                    waqf_after=int(wave_metrics.get("waqf_after") or 0),
                    sim=sim,
                    min_sim=_MIN_MATCH_SIM,
                    action=sequence_action.get(k),
                )
                short_low_sim_start = _short_low_sim_dsp_start_candidate(
                    pause_reason=pause_reason,
                    pause_duration_ms=pause.end_ms - pause.start_ms,
                    prev_quality=float(wave_metrics.get("prev_quality") or 0.0),
                    next_speech_ms=float(wave_metrics.get("next_speech_ms") or 0.0),
                    sim=sim,
                    action=sequence_action.get(k),
                )
                if compact_pause_start or long_pause_start_timing or long_pause_balanced_speech:
                    rough = pause.start_ms
                    timing_adjusted = True
                if compact_pause_end:
                    rough = pause.end_ms
                    timing_adjusted = True
                if long_dsp_fraction is not None:
                    rough = pause.start_ms + long_dsp_fraction * (pause.end_ms - pause.start_ms)
                    timing_adjusted = True
                if short_low_sim_start:
                    rough = pause.start_ms
                    timing_adjusted = True
                if dsp_edge_timing == "start":
                    rough = pause.start_ms
                    timing_adjusted = True
                elif dsp_edge_timing == "end":
                    rough = pause.end_ms
                    timing_adjusted = True
            short_low_sim_lock = (
                pause is not None
                and pause_reason == "dsp"
                and pause.end_ms - pause.start_ms <= 700.0
                and float(wave_metrics.get("prev_quality") or 0.0) >= 0.70
                and float(wave_metrics.get("next_speech_ms") or 0.0) <= 800.0
                and sim <= 0.12
            )
            if short_low_sim_lock:
                rough = pause.start_ms
                timing_adjusted = True
            if (
                pause is not None
                and pause_reason != "micro-tempo"
                and pause.end_ms - pause.start_ms >= 1800.0
                and (wave_metrics.get("prev_speech_ms") or 0.0) <= 1500.0
                and (wave_metrics.get("next_speech_ms") or 0.0) >= 3000.0
            ):
                previous_a = ayah_list[k]
                previous_place = place.get(previous_a)
                previous_time = (
                    previous_place[0]
                    if previous_place is not None
                    else starts_ms.get(previous_a, content_start)
                )
                expected_gap = (
                    (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                    if total_w > 0.0
                    else 0.0
                )
                prior_pause = _prior_pause_before_internal_stop(
                    previous_time,
                    pause.end_ms,
                    expected_gap,
                    [
                        center
                        for center in pause_onsets
                        if (center == matched[k] or center not in used_matched)
                        and (candidate := pause_by_center.get(center)) is not None
                        and (diag := pause_diag.get(candidate)) is not None
                        and diag.reason == "dsp"
                    ],
                    raw_is_internal_stop=True,
                )
                if prior_pause is not None and prior_pause != matched[k]:
                    old_matched = matched[k]
                    matched[k] = prior_pause
                    used_matched.discard(old_matched)
                    used_matched.add(prior_pause)
                    j = pause_onsets.index(prior_pause)
                    pause = pause_by_center.get(prior_pause)
                    pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                    sim = _sim_ij(k, j)
                    wave_metrics = rounded_wave_metrics(k, j)
                    rough = pause_timing_point(prior_pause, sim=sim, reason=pause_reason)
                    timing_adjusted = True
                else:
                    rough = pause.end_ms
            if pause is not None and not timing_adjusted:
                next_i = next(
                    (ii for ii in range(k + 1, len(matched)) if matched[ii] is not None),
                    None,
                )
                selected_has_micro_stop = any(
                    (micro_pause := pause_by_center.get(center)) is not None
                    and (micro_diag := pause_diag.get(micro_pause)) is not None
                    and micro_diag.reason == "micro-tempo"
                    and pause.start_ms <= center <= pause.end_ms + 300.0
                    for center in pause_onsets
                )
                if next_i is not None:
                    previous_a = ayah_list[k]
                    next_a = inner[next_i]
                    previous_place = place.get(previous_a)
                    previous_time = (
                        previous_place[0]
                        if previous_place is not None
                        else starts_ms.get(previous_a, content_start)
                    )
                    next_time = float(matched[next_i])
                    prior_pause = _prior_pause_between_anchors(
                        previous_time,
                        rough,
                        next_time,
                        (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0,
                        (cum_u[next_a] - cum_u[a]) / total_w * content_dur
                        if total_w > 0.0 else 0.0,
                        [
                            center
                            for center in pause_onsets
                            if center not in used_matched
                            and (candidate := pause_by_center.get(center)) is not None
                            and (diag := pause_diag.get(candidate)) is not None
                            and diag.reason == "dsp"
                        ],
                        selected_is_internal_stop=(
                            pause_reason != "micro-tempo"
                            and sim < _PAUSE_TRUST_SIM
                            and pause.end_ms - pause.start_ms >= 700.0
                            and float(wave_metrics.get("prev_speech_ms") or 0.0) >= 1800.0
                            and float(wave_metrics.get("next_speech_ms") or 0.0) <= 500.0
                            and selected_has_micro_stop
                        ),
                    )
                    if prior_pause is not None:
                        old_matched = matched[k]
                        matched[k] = prior_pause
                        used_matched.discard(old_matched)
                        used_matched.add(prior_pause)
                        j = pause_onsets.index(prior_pause)
                        pause = pause_by_center.get(prior_pause)
                        pause_reason = pause_diag[pause].reason if pause in pause_diag else None
                        sim = _sim_ij(k, j)
                        wave_metrics = rounded_wave_metrics(k, j)
                        rough = pause_timing_point(prior_pause, sim=sim, reason=pause_reason)
                        timing_adjusted = True
            final_short_low_sim_lock = (
                pause is not None
                and pause_reason == "dsp"
                and sequence_action.get(k) == "weak_pause_recovery"
                and pause.end_ms - pause.start_ms <= 700.0
                and sim <= 0.12
                and float(wave_metrics.get("prev_speech_ms") or 0.0) >= 5000.0
                and float(wave_metrics.get("next_speech_ms") or 0.0) <= 800.0
            )
            if final_short_low_sim_lock:
                rough = pause.start_ms
                timing_adjusted = True
                short_low_sim_lock = True
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
                "timing_lock": "short_low_sim_start" if short_low_sim_lock else None,
                "prev_speech_ms": wave_metrics.get("prev_speech_ms"),
                "next_speech_ms": wave_metrics.get("next_speech_ms"),
            }
            pause_match_debug.append({
                "ayah": a,
                "expected_ms": round(expected, 1),
                "rough_ms": round(rough, 1),
                "from_pause": True,
                "pause": pause_ord.get(matched[k]),
                "pause_start_ms": round(pause.start_ms, 1) if pause is not None else None,
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
                "timing_adjusted": timing_adjusted,
                "timing_lock": "short_low_sim_start" if short_low_sim_lock else None,
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
            recovered_wasl_pause: float | None = None
            recovered_wasl_pause_reason: str | None = None
            recovered_wasl_sim: float | None = None
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
                pause_candidate = _missing_wasl_pause_candidate(
                    rough_ms=rough,
                    expected_ms=expected,
                    previous_time=prev_t,
                    next_time=next_t,
                    candidates=[
                        center for center in pause_onsets
                        if center not in used_matched and strong_pause(center)
                    ],
                )
                if pause_candidate is not None:
                    recovered_wasl_pause = pause_candidate
                    j = pause_onsets.index(pause_candidate)
                    pause = pause_by_center.get(pause_candidate)
                    recovered_wasl_pause_reason = (
                        pause_diag[pause].reason if pause in pause_diag else None
                    )
                    recovered_wasl_sim = _sim_ij(k, j)
                    matched[k] = pause_candidate
                    used_matched.add(pause_candidate)
                    rough = pause_candidate
            wave_j = (
                pause_onsets.index(recovered_wasl_pause)
                if recovered_wasl_pause is not None
                else best_near_j if best_near_j >= 0 else best_j
            )
            wave_metrics = rounded_wave_metrics(k, wave_j)
            if recovered_wasl_pause is not None:
                place[a] = (rough, True)
                pause_match_by_ayah[a] = {
                    "pause": pause_ord.get(recovered_wasl_pause),
                    "sim": recovered_wasl_sim,
                    "center_ms": recovered_wasl_pause,
                    "rough_ms": rough,
                    "action": "missing_wasl_pause_recovery",
                    "reason": recovered_wasl_pause_reason,
                    "timing_adjusted": True,
                    "timing_lock": None,
                    "prev_speech_ms": wave_metrics.get("prev_speech_ms"),
                    "next_speech_ms": wave_metrics.get("next_speech_ms"),
                }
                pause_match_debug.append({
                    "ayah": a,
                    "expected_ms": round(expected, 1),
                    "rough_ms": round(rough, 1),
                    "from_pause": True,
                    "pause": pause_ord.get(recovered_wasl_pause),
                    "pause_start_ms": round(pause.start_ms, 1) if pause is not None else None,
                    "pause_center_ms": round(recovered_wasl_pause, 1),
                    "pause_end_ms": round(pause.end_ms, 1) if pause is not None else None,
                    "pause_sim": round(recovered_wasl_sim or 0.0, 3),
                    "best_pause": pause_ord.get(pause_onsets[best_j]) if best_j >= 0 else None,
                    "best_sim": round(best_sim, 3),
                    "best_near_pause": (
                        pause_ord.get(pause_onsets[best_near_j]) if best_near_j >= 0 else None
                    ),
                    "best_near_sim": round(best_near_sim, 3),
                    "prior_window_ms": round(_MATCH_PRIOR_WINDOW_MS, 1),
                    "expected_delta_ms": round(rough - expected, 1),
                    "drop_reason": drop_reason or "missing_before_future",
                    "match_action": "missing_wasl_pause_recovery",
                    "pause_reason": recovered_wasl_pause_reason,
                    "timing_adjusted": True,
                    "timing_lock": None,
                    **wave_metrics,
                })
                continue
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
        edge_candidates: list[tuple[float, float, bool]] | None = None,
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
            left_edge = cand - res["base"] < guard
            near_edge = left_edge or (res["end"] - cand < guard)
            if edge_candidates is not None:
                edge_candidates.append((cand, cconf, left_edge))
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
    section_onsets: dict[int, tuple[float, float]] = {}
    internal_stop_onsets: dict[int, tuple[float, float]] = {}

    def ayah_opening(a: int) -> str:
        return "".join(word_trs[i] for i in ayah_words[a][:4] if i < len(word_trs))

    def section_repair_member(ai: int) -> bool:
        a = ayah_list[ai]
        from_pause = place[a][1]
        match = pause_match_by_ayah.get(a, {})
        return _section_repair_member(
            from_pause=from_pause,
            action=str(match.get("action") or ""),
            sim=float(match.get("sim") or 0.0) if from_pause else None,
            min_sim=_MIN_MATCH_SIM,
        )

    def fill_section_repair(ai: int) -> None:
        left_a = ayah_list[ai - 1]
        i0 = ayah_words[left_a][0]
        t0 = starts_ms[left_a]
        j = ai
        run: list[int] = []
        while j < len(ayah_list) and section_repair_member(j):
            run.append(ayah_list[j])
            j += 1
        if len(run) < 2 or j >= len(ayah_list):
            return
        right_a = ayah_list[j]
        i1 = ayah_words[right_a][0] + 1
        t1 = place[right_a][0]
        if t1 - t0 > _MAX_SECTION_REPAIR_ALIGN_MS:
            emit(
                f"Section-repair {run[0]}-{run[-1]}: окно {(t1 - t0) / 1000.0:.2f} сек. "
                "слишком длинное для fast"
            )
            return
        res = align_slice(i0, i1, t0 - 60.0, t1 + 60.0, basmala=False)
        if res is None:
            return
        previous_a = left_a
        previous_onset = t0
        for a in run:
            fa = ayah_words[a][0]
            k = fa - i0
            onset = res["starts"][k]
            conf = res["confs"][k]
            raw_hint = nearest_pause_point(onset, strong_only=False)
            raw_pause = pause_by_center.get(raw_hint) if raw_hint is not None else None
            raw_diag = pause_diag.get(raw_pause) if raw_pause is not None else None
            expected_gap = (
                (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
                if total_w > 0
                else 0.0
            )
            prior_pause = _prior_pause_before_internal_stop(
                previous_onset,
                onset,
                expected_gap,
                [
                    center for center in pause_onsets
                    if (pause := pause_by_center.get(center)) is not None
                    and (diag := pause_diag.get(pause)) is not None
                    and diag.reason == "dsp"
                ],
                raw_is_internal_stop=(
                    raw_hint is not None
                    and abs(onset - raw_hint) <= _WASL_PAUSE_HINT_MS
                    and raw_diag is not None
                    and raw_diag.reason == "micro-tempo"
                ),
            )
            if prior_pause is not None:
                onset = prior_pause
            else:
                for center in pause_onsets:
                    pause = pause_by_center.get(center)
                    diag = pause_diag.get(pause) if pause is not None else None
                    if pause is None or diag is None:
                        continue
                    candidate = _section_long_pause_start_candidate(
                        onset,
                        pause.start_ms,
                        pause.end_ms,
                        pause_reason=diag.reason,
                        min_pause_ms=_LONG_PAUSE_CENTER_MS,
                    )
                    if candidate is not None:
                        onset = candidate
                        break
            hint = nearest_pause_point(onset, strong_only=False)
            if hint is not None and abs(onset - hint) <= _WASL_PAUSE_HINT_MS:
                onset = hint
            section_onsets[a] = (onset, conf)
            previous_a = a
            previous_onset = onset
        model_reads.append({
            "phase": "section-repair",
            "ayah": run[0],
            "window_ms": [round(res["base"], 1), round(res["end"], 1)],
            "words": " ".join(w.text for w in words[i0:i1]),
            "target": " ".join(words[ayah_words[a][0]].text for a in run),
            "onset_ms": round(section_onsets[run[0]][0], 1),
            "conf": round(section_onsets[run[0]][1], 3),
        })
        log_read(model_reads[-1])

    def fill_internal_stop_repair(ai: int) -> None:
        a = ayah_list[ai]
        if ai + 1 >= len(ayah_list):
            return
        from_pause = place[a][1]
        match = pause_match_by_ayah.get(a, {})
        center = float(match.get("center_ms") or place[a][0])
        pause = pause_by_center.get(center)
        diag = pause_diag.get(pause) if pause is not None else None
        if pause is None or diag is None:
            return
        has_micro_stop = any(
            (micro_pause := pause_by_center.get(candidate)) is not None
            and (micro_diag := pause_diag.get(micro_pause)) is not None
            and micro_diag.reason == "micro-tempo"
            and pause.start_ms <= candidate <= pause.end_ms + 300.0
            for candidate in pause_onsets
        )
        if not _internal_stop_repair_candidate(
            from_pause=from_pause,
            action=str(match.get("action") or ""),
            reason=str(match.get("reason") or ""),
            sim=float(match.get("sim") or 0.0),
            pause_duration_ms=pause.end_ms - pause.start_ms,
            prev_speech_ms=float(match.get("prev_speech_ms") or 0.0),
            next_speech_ms=float(match.get("next_speech_ms") or 0.0),
            has_micro_stop=has_micro_stop,
            pause_trust_sim=_PAUSE_TRUST_SIM,
        ):
            return
        prev_a = ayah_list[ai - 1]
        next_a = ayah_list[ai + 1]
        prior_candidates = [
            candidate
            for candidate in pause_onsets
            if starts_ms[prev_a] + 2500.0 < candidate < center - 500.0
            and (candidate_pause := pause_by_center.get(candidate)) is not None
            and (candidate_diag := pause_diag.get(candidate_pause)) is not None
            and candidate_diag.reason == "dsp"
        ]
        if not prior_candidates:
            return
        i0 = ayah_words[prev_a][0]
        i1 = ayah_words[next_a][0] + 1
        t0 = starts_ms[prev_a] - 60.0
        t1 = place[next_a][0] + 60.0
        if t1 - t0 > 30_000.0:
            return
        res = align_slice(i0, i1, t0, t1, basmala=False)
        if res is None:
            return
        fa = ayah_words[a][0]
        onset = res["starts"][fa - i0]
        conf = res["confs"][fa - i0]
        if conf < conf_reliable:
            return
        hint = min(prior_candidates, key=lambda candidate: abs(candidate - onset))
        if abs(hint - onset) > _WASL_PAUSE_HINT_MS:
            return
        internal_stop_onsets[a] = (hint, conf)
        model_reads.append({
            "phase": "internal-stop-repair",
            "ayah": a,
            "window_ms": [round(res["base"], 1), round(res["end"], 1)],
            "words": " ".join(w.text for w in words[i0:i1]),
            "target": words[fa].text,
            "onset_ms": round(hint, 1),
            "conf": round(conf, 3),
        })
        log_read(model_reads[-1])

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
        timing_locked = pause_match_by_ayah.get(a, {}).get("timing_lock") == "short_low_sim_start"

        internal_stop_repaired = False
        if from_pause and not timing_locked:
            if a not in internal_stop_onsets:
                fill_internal_stop_repair(ai)
            if a in internal_stop_onsets:
                onset, conf = internal_stop_onsets[a]
                win_used = 0.0
                reliable = conf >= conf_thresh
                internal_stop_repaired = True

        section_repaired = False
        if not timing_locked and not internal_stop_repaired and section_repair_member(ai):
            if a not in section_onsets:
                fill_section_repair(ai)
            if a in section_onsets:
                onset, conf = section_onsets[a]
                win_used = 0.0
                reliable = conf >= conf_thresh
                section_repaired = True

        if internal_stop_repaired or section_repaired:
            pass
        elif from_pause:
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
                edge_candidates: list[tuple[float, float, bool]] = []
                refined_onset, refined_conf, win_used = refine(
                    c,
                    fa,
                    prev_start,
                    0,
                    ayah=a,
                    edge_candidates=edge_candidates,
                )
                confirm_delta_ms = _PAUSE_REFINE_CONFIRM_MS
                if pause is not None and pause.end_ms - pause.start_ms >= 700.0:
                    confirm_delta_ms = max(confirm_delta_ms, pause.end_ms - pause.start_ms)
                actual_reason = (
                    pause_diag[pause].reason
                    if pause is not None and pause in pause_diag
                    else pause_reason
                )
                left_edge_candidate = (
                    _left_edge_repeated_opening_candidate(
                        edge_candidates,
                        pause_start_ms=pause.start_ms,
                        pause_center_ms=c,
                        min_conf=conf_reliable,
                    )
                    if pause is not None
                    and actual_reason == "dsp"
                    and pause.end_ms - pause.start_ms >= _LONG_PAUSE_CENTER_MS
                    and str(match.get("action") or "") == "opening_run_repair"
                    and pause_sim >= _MIN_MATCH_SIM
                    else None
                )
                if left_edge_candidate is not None:
                    onset, edge_conf = left_edge_candidate
                    conf = max(conf_thresh, pause_sim, edge_conf)
                    reliable = True
                    emit(
                        f"Пауза {match.get('pause', '?')} подтверждена left-edge "
                        f"decode для repeated opening аята {a}: старт {sec(onset)} сек."
                    )
                elif _pause_refine_confirms_pause(
                    c,
                    refined_onset,
                    refined_conf,
                    max_delta_ms=confirm_delta_ms,
                ):
                    onset = _pause_backed_refine_start(c)
                    conf = max(conf_thresh, pause_sim, refined_conf)
                    reliable = True
                    emit(
                        f"Пауза {match.get('pause', '?')} подтверждена refine для аята {a}: "
                        f"слово найдено {sec(refined_onset)} сек., "
                        f"оставляю старт в центре паузы {sec(onset)} сек."
                    )
                else:
                    preword_valley = None
                    if (
                        pause is not None
                        and _preword_micro_timing_candidate(
                            pause_reason=actual_reason,
                            pause_start_ms=pause.start_ms,
                            refined_onset_ms=refined_onset,
                            refined_conf=refined_conf,
                            conf_thresh=conf_thresh,
                            sim=pause_sim,
                            min_sim=_MIN_MATCH_SIM,
                        )
                    ):
                        interval_start = max(prev_start + 300.0, refined_onset - 3500.0)
                        interval_end = min(pause.start_ms, refined_onset - 120.0)
                        if interval_end - interval_start >= 300.0:
                            preword_valley = _find_micro_pause_valley(
                                pcm,
                                sr,
                                interval_start_ms=interval_start,
                                interval_end_ms=interval_end,
                                expected_ms=refined_onset - 700.0,
                            )
                    if preword_valley is not None:
                        _start_ms, end_ms, ratio, seed_start_ms, seed_end_ms = preword_valley
                        onset = end_ms
                        conf = max(conf_thresh, pause_sim, refined_conf)
                        reliable = True
                        emit(
                            f"Пауза {match.get('pause', '?')} поздняя для аята {a}; "
                            f"refine нашёл слово {sec(refined_onset)} сек., "
                            f"ставлю старт на pre-word micro-паузу "
                            f"{sec(seed_start_ms)}-{sec(seed_end_ms)} сек. "
                            f"(rms={ratio:.3f})"
                        )
                    else:
                        onset = c
                        conf = max(pause_sim, refined_conf)
                        reliable = False
                if not reliable:  # never locked on → trust the matched guess
                    onset = c
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

    for ai in range(1, len(ayah_list) - 1):
        a = ayah_list[ai]
        if not place.get(a, (0.0, False))[1]:
            continue
        match = pause_match_by_ayah.get(a, {})
        if match.get("reason") != "weak-dsp" or match.get("action") != "recovered":
            continue
        pause = pause_by_center.get(float(match.get("center_ms") or place[a][0]))
        if pause is None:
            continue
        previous_a = ayah_list[ai - 1]
        next_a = ayah_list[ai + 1]
        if previous_a not in starts_ms or next_a not in starts_ms or a not in starts_ms:
            continue
        current_time = starts_ms[a]
        previous_time = starts_ms[previous_a]
        next_time = starts_ms[next_a]
        expected_prev_gap = (
            (cum_u[a] - cum_u[previous_a]) / total_w * content_dur
            if total_w > 0.0 else 0.0
        )
        expected_next_gap = (
            (cum_u[next_a] - cum_u[a]) / total_w * content_dur
            if total_w > 0.0 else 0.0
        )
        interval_start = max(
            pause.end_ms + 80.0,
            current_time + 120.0,
            previous_time + 300.0,
        )
        interval_end = min(next_time - 300.0, current_time + 3500.0)
        if interval_end - interval_start < 700.0:
            continue
        valley = _find_micro_pause_valley(
            pcm,
            sr,
            interval_start_ms=interval_start,
            interval_end_ms=interval_end,
            expected_ms=previous_time + expected_prev_gap,
            search_ms=2500.0,
        )
        if valley is None:
            continue
        valley_center = _micro_valley_center(valley)
        selected_prev_quality = _span_ratio_quality(
            current_time - previous_time,
            expected_prev_gap,
        )
        selected_next_quality = _span_ratio_quality(
            next_time - current_time,
            expected_next_gap,
        )
        candidate_prev_quality = _span_ratio_quality(
            valley_center - previous_time,
            expected_prev_gap,
        )
        candidate_next_quality = _span_ratio_quality(
            next_time - valley_center,
            expected_next_gap,
        )
        pause_sim = float(match.get("sim") or 0.0)
        if not _weak_dsp_post_selected_micro_valley_candidate(
            pause_reason=str(match.get("reason") or ""),
            action=str(match.get("action") or ""),
            pause_sim=pause_sim,
            valley_ratio=valley[2],
            forward_shift_ms=valley_center - current_time,
            selected_prev_quality=selected_prev_quality,
            selected_next_quality=selected_next_quality,
            candidate_prev_quality=candidate_prev_quality,
            candidate_next_quality=candidate_next_quality,
        ):
            continue
        starts_ms[a] = valley_center
        confs[a] = max(conf_thresh, confs.get(a, 0.0), pause_sim)
        for t in tr_ayahs:
            if t["ayah"] == a:
                t.update(
                    onset_ms=valley_center,
                    conf=round(confs[a], 3),
                    reliable=True,
                    refine_skipped=True,
                    post_selected_micro_repair=True,
                )
                break
        match["rough_ms"] = valley_center
        match["timing_adjusted"] = True
        match["post_selected_micro_repair"] = True

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

    def _nearest_pause_grid(t: float) -> tuple[float, str | None]:
        best: tuple[float, str | None] = (1e9, None)
        for center in pause_onsets:
            pause = pause_by_center.get(center)
            diag = pause_diag.get(pause) if pause is not None else None
            reason = diag.reason if diag is not None else None
            for point in (center, pause_timing_point(center, reason=reason)):
                dist = abs(t - point)
                if dist < best[0]:
                    best = (dist, reason)
        return best

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
        if place[a][1] and repair_match.get("timing_lock") == "short_low_sim_start":
            continue
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
            if _reject_low_conf_off_grid_repair(
                from_pause=place[a][1],
                repair_conf=c2,
                conf_thresh=conf_thresh,
                old_pause_dist_ms=_dist_pause(old_onset),
                chosen_pause_dist_ms=_dist_pause(chosen),
                shift_ms=abs(chosen - old_onset),
            ):
                continue
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
            nearest_repair_pause_dist, nearest_repair_pause_reason = _nearest_pause_grid(chosen)
            if hi is not None and cum[hi] > cum[lo]:
                expected_prev_gap = pred - starts_ms[lo]
                expected_next_gap: float | None = starts_ms[hi] - pred
                next_anchor_time: float | None = starts_ms[hi]
            else:
                expected_prev_gap = pred - starts_ms[lo]
                expected_next_gap = None
                next_anchor_time = None
            later_pause_grid_repair = _accept_later_pause_grid_repair(
                from_pause=place[a][1],
                original_reason=repair_pause_reason,
                original_sim=pause_sim,
                old_onset=old_onset,
                repaired_onset=chosen,
                repaired_conf=c2,
                nearest_pause_dist_ms=nearest_repair_pause_dist,
                nearest_pause_reason=nearest_repair_pause_reason,
                previous_time=starts_ms[lo],
                next_time=next_anchor_time,
                expected_prev_gap_ms=expected_prev_gap,
                expected_next_gap_ms=expected_next_gap,
                conf_reliable=conf_reliable,
                pause_trust_sim=_PAUSE_TRUST_SIM,
            )
            if not _accept_pause_backed_repair(
                from_pause=place[a][1],
                original_pause_reason=repair_pause_reason,
                old_onset_ms=old_onset,
                chosen_onset_ms=chosen,
                chosen_pause_dist_ms=nearest_repair_pause_dist,
                chosen_pause_reason=nearest_repair_pause_reason,
                later_pause_grid_repair=later_pause_grid_repair,
            ):
                continue
            if (
                place[a][1]
                and confs.get(a, 0.0) >= conf_thresh
                and _dist_pause(chosen) > 350.0
                and not (
                    micro_pause_repair
                    or later_pause_grid_repair
                    or (strong_pause_repair and off_grid_back_repair)
                )
            ):
                continue
            if (
                place[a][1]
                and confs.get(a, 0.0) >= conf_thresh
                and abs(old_onset - place[a][0]) <= 80.0
                and abs(chosen - old_onset) > 500.0
                and not (
                    micro_pause_repair
                    or later_pause_grid_repair
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
