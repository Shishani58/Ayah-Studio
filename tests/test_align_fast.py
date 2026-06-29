"""Unit tests for the pure (model-free) helpers of the fast aligner.

Only the top-level functions are exercised — they import nothing heavy, so these
run without the ONNX model or any audio. The DP matcher ``_match_boundaries`` is
the off-by-one-prone heart of Phase 1c; the rest guard the small primitives the
char prior and acoustic cost are built on.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from surah_cutter.core.align_fast import (
    _ambiguous_openings,
    _ayah_waqf_counts,
    _collapse,
    _common_prefix_len,
    _count_waqf_marks,
    _find_micro_pause_valley,
    _interp_wasl,
    _match_boundaries,
    _opening_shared_prefix_chars,
    _pause_center_with_refined_onset,
    _pause_refine_confirms_pause,
    _sequential_pause_match,
    _sim,
    _span_ratio_quality,
    _unit_len,
)


def test_collapse_runs():
    assert _collapse("") == ""
    assert _collapse("aaa") == "a"
    assert _collapse("aabbc") == "abc"
    assert _collapse("abc") == "abc"


def test_common_prefix_len_uses_collapsed_text():
    assert _common_prefix_len("waa", "wa-idha") == 2
    assert _common_prefix_len("samāu", "wa-idha") == 0


def test_unit_len_long_vowels_double():
    assert _unit_len("abc") == 3.0          # no long vowels
    assert _unit_len("ā") == 2.0            # one madd carrier
    assert _unit_len("bā") == 3.0           # 1 + 2
    assert _unit_len("") == 0.0


def test_span_ratio_quality_prefers_matching_duration():
    assert _span_ratio_quality(1000.0, 1000.0) == 1.0
    assert _span_ratio_quality(1000.0, 2500.0) == 0.0
    assert _span_ratio_quality(0.0, 1000.0) == 0.0


def test_find_micro_pause_valley_detects_local_rms_dip():
    sr = 1000
    pcm = np.ones(3000, dtype=np.float32) * 0.2
    pcm[1350:1650] = 0.08
    pcm[1450:1550] = 0.01

    valley = _find_micro_pause_valley(
        pcm,
        sr,
        interval_start_ms=500.0,
        interval_end_ms=2500.0,
        expected_ms=1500.0,
        search_ms=400.0,
        frame_ms=40.0,
        hop_ms=10.0,
        max_width_ms=160.0,
        shelf_max_width_ms=500.0,
        shelf_ratio=2.0,
        rms_ratio=0.5,
    )

    assert valley is not None
    start_ms, end_ms, ratio, seed_start_ms, seed_end_ms = valley
    assert 1330.0 <= start_ms <= 1420.0
    assert 1580.0 <= end_ms <= 1670.0
    assert 1380.0 <= seed_start_ms <= 1510.0
    assert 1490.0 <= seed_end_ms <= 1600.0
    assert end_ms - start_ms > seed_end_ms - seed_start_ms
    assert ratio < 0.2


def test_sim_identical_and_collapsed():
    assert _sim("kafaru", "kafaru") == 1.0
    # Recitation stretches vowels; collapse makes the stretched onset match.
    assert _sim("kafaaaaruuu", "kafaru") == 1.0
    assert _sim("xyz", "kafaru") < 0.3


def test_pause_refine_confirms_pause_when_word_found_near_center():
    assert _pause_refine_confirms_pause(
        52_750.0,
        53_021.4,
        0.23,
        max_delta_ms=900.0,
        min_conf=0.05,
    )
    assert not _pause_refine_confirms_pause(
        52_750.0,
        54_726.3,
        0.23,
        max_delta_ms=900.0,
        min_conf=0.05,
    )
    assert not _pause_refine_confirms_pause(
        52_750.0,
        53_021.4,
        0.0,
        max_delta_ms=900.0,
        min_conf=0.05,
    )


def test_pause_center_with_refined_onset_uses_word_as_right_edge():
    assert _pause_center_with_refined_onset(205_029.0, 205_479.0, 206_229.0) == 205_629.0
    assert _pause_center_with_refined_onset(205_029.0, 205_479.0, 205_420.0) == 205_479.0
    assert _pause_center_with_refined_onset(205_029.0, 205_479.0, None) == 205_479.0


def test_match_perfect_one_to_one():
    exp = [1000.0, 2000.0, 3000.0]
    ons = [1000.0, 2000.0, 3000.0]
    out = _match_boundaries(exp, ons, anchor=0.0, wasl_pen=2600.0, extra_pen=350.0)
    assert out == [1000.0, 2000.0, 3000.0]


def test_match_skips_extra_breath_onset():
    # 1500 is a within-region breath; the three boundaries must still land on
    # the real onsets, not be dragged onto the extra.
    exp = [1000.0, 2000.0, 3000.0]
    ons = [1000.0, 1500.0, 2000.0, 3000.0]
    out = _match_boundaries(exp, ons, anchor=0.0, wasl_pen=2600.0, extra_pen=350.0)
    assert out == [1000.0, 2000.0, 3000.0]


def test_match_wasl_boundary_left_unmatched():
    # Middle boundary has no pause → left wasl (None); the two bracketing
    # boundaries still match their onsets.
    exp = [1000.0, 2000.0, 3000.0]
    ons = [1000.0, 3000.0]
    out = _match_boundaries(exp, ons, anchor=0.0, wasl_pen=2600.0, extra_pen=350.0)
    assert out[0] == 1000.0
    assert out[2] == 3000.0
    assert out[1] is None


def test_match_empty_inputs():
    assert _match_boundaries([], [1.0], anchor=0.0, wasl_pen=1.0, extra_pen=1.0) == []
    assert _match_boundaries([1.0, 2.0], [], anchor=0.0, wasl_pen=1.0, extra_pen=1.0) == [
        None,
        None,
    ]


def test_match_is_order_preserving():
    exp = [1000.0, 2000.0, 3000.0, 4000.0]
    ons = [1000.0, 2000.0, 3000.0, 4000.0]
    out = _match_boundaries(exp, ons, anchor=0.0, wasl_pen=2600.0, extra_pen=350.0)
    placed = [o for o in out if o is not None]
    assert placed == sorted(placed)


def test_interp_wasl_brackets_both_sides():
    exp = [0.0, 1000.0, 2000.0]
    matched = [0.0, None, 2000.0]
    # Boundary 1 sits at the prior midpoint → interpolated to the onset midpoint.
    assert _interp_wasl(1, exp, matched) == 1000.0


def test_interp_wasl_end_extrapolation():
    exp = [0.0, 1000.0, 2000.0]
    # Only the left neighbour is matched → extrapolate by the prior offset.
    matched = [500.0, None, None]
    assert _interp_wasl(1, exp, matched) == 1500.0
    # Only the right neighbour is matched → back off by the prior offset.
    matched = [None, None, 3000.0]
    assert _interp_wasl(1, exp, matched) == 2000.0


def test_interp_wasl_no_anchor_returns_prior():
    exp = [0.0, 1000.0, 2000.0]
    matched = [None, None, None]
    assert _interp_wasl(1, exp, matched) == 1000.0


def test_sequential_pause_match_confirms_in_order():
    exp = [1000.0, 2000.0, 3000.0]
    ons = [990.0, 2010.0, 3020.0]

    def sim(i: int, j: int) -> float:
        return 0.9 if i == j else 0.05

    matched, events = _sequential_pause_match(
        exp, ons, sim, min_sim=0.4, prior_window_ms=1200.0
    )

    assert matched == ons
    assert [e["action"] for e in events if e["action"] == "confirmed"] == [
        "confirmed",
        "confirmed",
        "confirmed",
    ]


def test_ambiguous_openings_marks_near_repeated_openings_only():
    openings = [
        "wawajadakaḍāllanfahadā",
        "wawajadakaʿāilanfa-aghnā",
        "fa-ammāl-yatīmafalātaqhar",
        "wa-ammāl-sāilafalātanhar",
        "wa-ammābiniʿ'matirabbikafaḥaddith",
    ]

    assert _ambiguous_openings(openings, lookahead=1, prefix_chars=6) == {0, 1, 3, 4}


def test_opening_shared_prefix_chars_measures_long_repeated_starts():
    openings = [
        "innalladhīnakafarūminahlikitābi",
        "innalladhīnaāmanūwaʿamilūṣāliḥāti",
        "fa-ammāl-yatīmafalātaqhar",
    ]

    shared = _opening_shared_prefix_chars(openings, lookahead=1, prefix_chars=6)

    assert shared[0] >= 10
    assert shared[1] == shared[0]
    assert shared[2] == 0


def test_sim_can_compare_past_default_prefix_for_repeated_starts():
    dec = "innalladhīnaāmanū"
    kafaru = "innalladhīnakafarū"
    amanu = "innalladhīnaāmanū"

    assert _sim(dec, kafaru, chars=10) == _sim(dec, amanu, chars=10)
    assert _sim(dec, amanu, chars=24) > _sim(dec, kafaru, chars=24)


def test_sequential_pause_match_stops_on_future_ayah_gap():
    exp = [1000.0, 2000.0, 3000.0]
    ons = [1000.0, 3000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.9,
            (1, 1): 0.1,
            (2, 1): 0.92,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp, ons, sim, min_sim=0.4, prior_window_ms=1500.0, lookahead=2
    )

    assert matched == [1000.0, None, 3000.0]
    assert any(e["action"] == "future_gap" and e["heard_boundary"] == 2 for e in events)
    assert not any(e["action"] == "future_anchor" for e in events)
    assert any(e["action"] == "confirmed" and e["boundary"] == 2 for e in events)
    assert any(
        e["action"] == "wasl" and e["boundary"] == 1
        and e["reason"] == "missing_before_future"
        for e in events
    )


def test_sequential_pause_match_recovers_early_pause_before_future_anchor():
    exp = [2000.0, 4000.0, 6000.0]
    ons = [1500.0, 6000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.82,
            (1, 1): 0.05,
            (2, 1): 0.9,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.4,
        prior_window_ms=300.0,
        prior_grace_ms=0.0,
        lookahead=2,
    )

    assert matched == [1500.0, None, 6000.0]
    assert any(e["action"] == "recovered" and e["boundary"] == 0 for e in events)
    assert not any(e["action"] == "future_anchor" for e in events)
    assert any(e["action"] == "confirmed" and e["boundary"] == 2 for e in events)


def test_sequential_pause_match_keeps_near_early_pause_inside_grace():
    exp = [392_571.7, 407_456.7]
    ons = [377_340.0, 385_590.0, 393_150.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.595,
            (0, 1): 0.350,
            (0, 2): 0.188,
            (1, 2): 0.438,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        recover_min_sim=0.34,
        prior_window_ms=15_000.0,
        prior_grace_ms=1_000.0,
        lookahead=1,
    )

    assert matched == [377_340.0, 393_150.0]
    assert any(e["action"] == "confirmed" and e["boundary"] == 0 for e in events)
    assert any(e["action"] == "observed_current" and e["boundary"] == 1 for e in events)


def test_sequential_pause_match_prefers_trusted_weak_pause_before_later_worse_match():
    exp = [497_612.6]
    ons = [482_910.0, 486_930.0, 492_810.0, 503_150.0]

    def sim(i: int, j: int) -> float:
        return [0.500, 0.154, 0.242, 0.467][j]

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        prior_window_ms=15_000.0,
    )

    assert matched == [482_910.0]
    assert any(e["action"] == "confirmed" and e["pause"] == 0 for e in events)


def test_sequential_pause_match_keeps_observed_current_below_confirm_threshold():
    exp = [1000.0, 2000.0]
    ons = [1000.0, 2000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.46,
            (0, 1): 0.10,
            (1, 1): 0.80,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        confirm_min_sim=0.50,
        prior_window_ms=1200.0,
        lookahead=1,
    )

    assert matched == [1000.0, 2000.0]
    assert any(
        e["action"] == "observed_current_candidate" and e["boundary"] == 0
        for e in events
    )
    assert any(e["action"] == "recovered" and e["boundary"] == 0 for e in events)
    assert any(e["action"] == "confirmed" and e["boundary"] == 1 for e in events)


def test_sequential_pause_match_recovery_prefers_stronger_current_read():
    exp = [1000.0, 2000.0, 3000.0]
    ons = [1000.0, 1900.0, 2800.0, 3000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (1, 1): 0.43,
            (1, 2): 0.49,
            (2, 3): 0.90,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        confirm_min_sim=0.50,
        prior_window_ms=1200.0,
        lookahead=2,
    )

    assert matched == [1000.0, 2800.0, 3000.0]
    assert any(e["action"] == "recovered" and e["boundary"] == 1 and e["pause"] == 2 for e in events)


def test_sequential_pause_match_recovery_rejects_observed_wrong_pause():
    exp = [233_388.8, 253_340.0, 270_000.0]
    ons = [216_170.0, 234_640.0, 244_670.0, 253_340.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.438,  # early, but this pause actually sounds like current
            (0, 1): 0.529,  # best current read, despite weak-pause/reject guards
            (0, 2): 0.700,  # later candidate; fuzzy-matches current
            (1, 3): 0.537,  # future gap proves current was skipped
            (2, 2): 0.750,  # but the later candidate sounds more like another ayah
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        recover_min_sim=0.34,
        confirm_min_sim=0.80,
        prior_window_ms=15_000.0,
        prior_grace_ms=1_000.0,
        lookahead=1,
    )

    assert matched[0] == 234_640.0
    assert matched[0] != 244_670.0
    assert any(
        e["action"] == "early_pool" and e["pause"] == 0 and e["observed_boundary"] == 0
        for e in events
    )
    assert any(
        e["action"] == "skip_recovery_observed_mismatch" and e["pause"] == 2
        for e in events
    )
    assert any(
        e["action"] == "recovered" and e["boundary"] == 0 and e["pause"] == 1
        for e in events
    )


def test_sequential_pause_match_future_gap_ignores_far_global_match():
    exp = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    ons = [1000.0, 2000.0, 3000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (1, 1): 0.22,
            (4, 1): 0.92,  # far global false match; must not trigger future_gap
            (1, 2): 0.78,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        confirm_min_sim=0.50,
        prior_window_ms=1500.0,
        lookahead=1,
    )

    assert matched[1] == 3000.0
    assert not any(e["action"] == "future_gap" and e["boundary"] == 1 for e in events)
    assert any(
        e["action"] == "extra"
        and e["boundary"] == 1
        and e["pause"] == 1
        and e.get("observed_global_boundary") == 4
        for e in events
    )


def test_sequential_pause_match_recovers_borderline_pause_before_future_anchor():
    exp = [1000.0, 2000.0, 3000.0]
    ons = [1000.0, 2000.0, 3000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (1, 1): 0.37,  # below normal threshold, valid only as recovery
            (1, 2): 0.10,
            (2, 2): 0.90,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        recover_min_sim=0.34,
        prior_window_ms=1200.0,
        lookahead=2,
    )

    assert matched == ons
    assert any(e["action"] == "recovered" and e["boundary"] == 1 for e in events)
    assert not any(e["action"] == "future_anchor" for e in events)
    assert any(e["action"] == "confirmed" and e["boundary"] == 2 for e in events)


def test_sequential_pause_match_rejects_tail_prefix_mismatch():
    exp = [1000.0, 2000.0]
    ons = [1000.0, 2000.0, 3000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (1, 1): 0.50,  # fuzzy match, but heard text starts from previous tail
            (1, 2): 0.80,
        }
        return scores.get((i, j), 0.02)

    def reject(i: int, j: int, s: float) -> str | None:
        if i == 1 and j == 1:
            return "prefix_mismatch"
        return None

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.4,
        prior_window_ms=2500.0,
        reject=reject,
    )

    assert matched == [1000.0, 3000.0]
    assert any(
        e["action"] == "extra" and e["pause"] == 1 and e["reason"] == "low_or_internal_before_confirmed"
        for e in events
    )


def test_sequential_pause_match_stops_on_unclear_wrong_place():
    exp = [1000.0, 2000.0]
    ons = [1000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.10,
            (1, 0): 0.35,  # sounds more like a future ayah, but below threshold
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        prior_window_ms=1200.0,
        lookahead=1,
    )

    assert matched == [None, None]
    assert any(
        e["action"] == "unresolved_current"
        and e["boundary"] == 0
        and e["observed_boundary"] == 1
        and e["direction"] == "future"
        for e in events
    )
    assert not any(e["action"] == "wasl" and e["boundary"] == 0 for e in events)
    assert not any(e.get("boundary") == 1 and e["action"] == "confirmed" for e in events)


def test_sequential_pause_match_goes_forward_after_stale_pause():
    exp = [1000.0, 2000.0]
    ons = [1000.0, 1900.0, 2300.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (1, 1): 0.46,  # current, but below immediate confirm threshold
            (0, 2): 0.88,  # later pause sounds like an already passed ayah
            (1, 2): 0.10,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.40,
        confirm_min_sim=0.50,
        prior_window_ms=1500.0,
        lookahead=1,
    )

    assert matched == [1000.0, 1900.0]
    assert any(
        e["action"] == "forward_after_stale" and e["boundary"] == 1
        for e in events
    )


def test_count_waqf_marks_from_mushaf_words():
    words = [
        SimpleNamespace(text="عَلَيۡهِۚ"),
        SimpleNamespace(text="ثُمَّ"),
        SimpleNamespace(text="قَلِيلٗاۖ"),
    ]

    assert _count_waqf_marks(words) == 2


def test_ayah_waqf_counts_from_mushaf_words():
    words = [
        SimpleNamespace(ayah=1, text="عَلَيۡهِۚ"),
        SimpleNamespace(ayah=1, text="ثُمَّ"),
        SimpleNamespace(ayah=2, text="قَلِيلٗاۖ"),
    ]

    assert _ayah_waqf_counts(words) == {1: 1, 2: 1}
