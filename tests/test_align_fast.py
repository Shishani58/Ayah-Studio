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
    _accept_fallback_pause_candidate,
    _accept_later_pause_grid_repair,
    _accept_pause_backed_repair,
    _ambiguous_openings,
    _assign_opening_block,
    _ayah_waqf_counts,
    _collapse,
    _common_prefix_len,
    _compact_dsp_start_candidate,
    _compact_dsp_end_candidate,
    _count_waqf_marks,
    _dense_prior_opening_candidate,
    _dense_later_opening_candidate,
    _dense_later_final_opening_candidate,
    _dsp_edge_timing_candidate,
    _find_micro_pause_valley,
    _final_weak_current_activation_window,
    _first_ayah_content_start,
    _internal_stop_repair_candidate,
    _interp_wasl,
    _left_edge_repeated_opening_candidate,
    _late_repeated_dsp_prior_candidate,
    _long_dsp_fractional_start_candidate,
    _long_pause_start_after_speech_candidate,
    _long_pause_start_balanced_speech_candidate,
    _long_pause_start_timing_candidate,
    _match_boundaries,
    _missing_wasl_pause_candidate,
    _onset_prefix_len,
    _opening_internal_repeats,
    _opening_shared_prefix_chars,
    _micro_shelf_start_candidate,
    _prefer_expected_micro_valley,
    _pause_backed_refine_start,
    _pause_center_with_refined_onset,
    _pause_match_min_keep_sim,
    _pause_refine_confirms_pause,
    _preword_micro_timing_candidate,
    _prior_pause_before_internal_stop,
    _prior_pause_between_anchors,
    _prior_close_dsp_before_micro_stop,
    _rejected_micro_valley_candidate,
    _reject_low_conf_off_grid_repair,
    _sequential_pause_match,
    _section_repair_member,
    _section_long_pause_start_candidate,
    _skips_dense_pause_block,
    _sim,
    _short_low_sim_dsp_start_candidate,
    _span_ratio_quality,
    _unit_len,
    _weak_dsp_prior_recovery_candidate,
    _weak_dsp_prior_trusted_recovery_candidate,
    _weak_dsp_later_prefix_candidate,
    _weak_dsp_expected_micro_valley_candidate,
    _weak_dsp_post_selected_micro_valley_candidate,
    _weak_dsp_wasl_recovery_candidate,
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


def test_prefer_expected_micro_valley_only_for_late_comparable_candidate():
    current = (255_817.0, 255_953.0, 0.127, 255_825.0, 255_945.0)
    expected = (254_503.0, 254_685.0, 0.166, 254_503.0, 254_683.0)
    weak_expected = (254_503.0, 254_685.0, 0.25, 254_503.0, 254_683.0)

    assert (
        _prefer_expected_micro_valley(
            current,
            expected,
            expected_ms=253_685.0,
        )
        is expected
    )
    assert (
        _prefer_expected_micro_valley(
            current,
            weak_expected,
            expected_ms=253_685.0,
        )
        is current
    )
    assert (
        _prefer_expected_micro_valley(
            current,
            expected,
            expected_ms=256_000.0,
        )
        is current
    )


def test_micro_shelf_start_candidate_requires_late_structural_micro():
    assert (
        _micro_shelf_start_candidate(
            pause_reason="micro-tempo",
            sim=0.342,
            prev_quality=0.998,
            next_quality=0.373,
            center_ms=314_090.5,
            shelf_start_ms=313_905.5,
            confirm_match_sim=0.50,
        )
        == 313_905.5
    )
    assert (
        _micro_shelf_start_candidate(
            pause_reason="micro-tempo",
            sim=0.55,
            prev_quality=0.998,
            next_quality=0.373,
            center_ms=314_090.5,
            shelf_start_ms=313_905.5,
            confirm_match_sim=0.50,
        )
        is None
    )
    assert (
        _micro_shelf_start_candidate(
            pause_reason="micro-tempo",
            sim=0.342,
            prev_quality=0.758,
            next_quality=0.949,
            center_ms=524_074.1,
            shelf_start_ms=523_444.1,
            confirm_match_sim=0.50,
        )
        is None
    )


def test_sim_identical_and_collapsed():
    assert _sim("kafaru", "kafaru") == 1.0
    # Recitation stretches vowels; collapse makes the stretched onset match.
    assert _sim("kafaaaaruuu", "kafaru") == 1.0
    assert _sim("xyz", "kafaru") < 0.3


def test_sim_penalizes_previous_tail_before_opening():
    clean = _sim("innalladhīnaāmanū", "innalladhīnaāmanū", chars=24)
    with_tail = _sim("qālainnaalladhīnaāmanū", "innalladhīnaāmanū", chars=24)

    assert clean == 1.0
    assert with_tail < clean - 0.20


def test_onset_prefix_len_tolerates_dropped_initial_article_vowel():
    assert _onset_prefix_len("l-ladhīkadhabawatawallā", "alladhīkadhabawatawallā") >= 6


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


def test_pause_backed_refine_start_keeps_regular_pause_point():
    assert _pause_backed_refine_start(82_410.0) == 82_410.0


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


def test_assign_opening_block_keeps_pause_order_over_late_repeat():
    boundaries = [66, 67, 68]
    candidate_ms = {
        10: 480_150.0,
        11: 505_000.0,
        12: 514_730.0,
        13: 524_000.0,
    }
    scores = {
        (66, 10): 1.10,  # real pause-backed start
        (66, 12): 1.50,  # later repeated opening sounds stronger in isolation
        (67, 11): 1.10,
        (67, 13): 0.90,
        (68, 12): 1.10,
    }

    assignment = _assign_opening_block(
        boundaries,
        [10, 11, 12, 13],
        candidate_time=candidate_ms.__getitem__,
        score=lambda i, j: scores.get((i, j)),
        after_ms=470_000.0,
    )

    assert assignment == {66: 10, 67: 11, 68: 12}


def test_assign_opening_block_rejects_text_impossible_late_pause():
    candidate_ms = {
        10: 480_000.0,
        11: 570_000.0,
        12: 807_000.0,
    }
    expected_ms = {
        66: 516_000.0,
        67: 527_000.0,
        68: 536_000.0,
    }

    assignment = _assign_opening_block(
        [67, 68],
        [11, 12],
        candidate_time=candidate_ms.__getitem__,
        score=lambda _i, _j: 1.0,
        after_ms=candidate_ms[10],
        expected_time=expected_ms.__getitem__,
        after_expected_ms=expected_ms[66],
    )

    assert assignment == {}


def test_assign_opening_block_prefers_stronger_later_opening_chain():
    candidate_ms = {
        49: 247_160.0,
        50: 251_400.0,
        52: 254_260.0,
        53: 257_590.0,
        54: 261_170.0,
        55: 265_430.0,
    }
    expected_ms = {
        32: 221_180.0,
        33: 249_256.0,
        34: 253_296.0,
        35: 261_218.0,
        36: 265_478.0,
    }
    scores = {
        (33, 49): 0.284,
        (33, 50): 0.285,
        (33, 52): 0.571,
        (34, 50): 0.285,
        (34, 53): 0.342,
        (35, 54): 0.295,
        (36, 55): 0.800,
    }

    assignment = _assign_opening_block(
        [33, 34, 35, 36],
        [49, 50, 52, 53, 54, 55],
        candidate_time=candidate_ms.__getitem__,
        score=lambda i, j: scores.get((i, j)),
        after_ms=candidate_ms[49] - 25_980.0,
        expected_time=expected_ms.__getitem__,
        after_expected_ms=expected_ms[32],
    )

    assert assignment == {33: 52, 34: 53, 35: 54, 36: 55}


def test_skips_dense_pause_block_detects_late_jump_over_sequence():
    candidate_ms = {
        10: 1_050_000.0,
        11: 1_054_500.0,
        12: 1_105_000.0,
    }
    expected_ms = {
        119: 934_000.0,
        120: 941_000.0,
        121: 947_000.0,
        130: 1_013_000.0,
    }

    assert _skips_dense_pause_block(
        120,
        12,
        [10, 11, 12],
        candidate_time=candidate_ms.__getitem__,
        expected_time=expected_ms.__getitem__,
        previous_boundary=119,
        previous_time=1_045_000.0,
        next_boundary=130,
    )


def test_section_repair_member_is_limited_to_wasl_and_low_fallback():
    assert _section_repair_member(
        from_pause=False,
        action="wasl",
        sim=None,
        min_sim=0.31,
    )
    assert _section_repair_member(
        from_pause=True,
        action="pause_fallback",
        sim=0.12,
        min_sim=0.31,
    )
    assert not _section_repair_member(
        from_pause=True,
        action="pause_fallback",
        sim=0.45,
        min_sim=0.31,
    )
    assert not _section_repair_member(
        from_pause=True,
        action="observed_current",
        sim=0.12,
        min_sim=0.31,
    )


def test_prior_pause_before_internal_stop_requires_internal_stop():
    kwargs = {
        "previous_time": 727_770.0,
        "raw_time": 742_620.0,
        "expected_gap_ms": 7_770.0,
        "strong_candidate_times": [734_280.0, 741_910.0],
    }

    assert _prior_pause_before_internal_stop(**kwargs, raw_is_internal_stop=True) == 734_280.0
    assert _prior_pause_before_internal_stop(**kwargs, raw_is_internal_stop=False) is None


def test_prior_pause_before_internal_stop_handles_long_pause_end():
    assert (
        _prior_pause_before_internal_stop(
            previous_time=853_722.0,
            raw_time=869_680.0,
            expected_gap_ms=7_770.0,
            strong_candidate_times=[861_490.0, 866_050.0],
            raw_is_internal_stop=True,
        )
        == 861_490.0
    )


def test_dense_prior_opening_candidate_recovers_earlier_opening_pause():
    kwargs = {
        "previous_time": 385_020.0,
        "selected_time": 413_910.0,
        "expected_gap_ms": 11_418.3,
        "candidates": [(408_140.0, 0.313, 1), (396_840.0, 0.554, 0)],
        "pause_reason": "dsp",
        "selected_prev_quality": 0.0,
        "selected_sim": 0.498,
        "action": "recovered",
    }

    assert _dense_prior_opening_candidate(**kwargs) == 408_140.0
    assert (
        _dense_prior_opening_candidate(
            **{**kwargs, "selected_prev_quality": 0.55}
        )
        is None
    )
    assert (
        _dense_prior_opening_candidate(
            **{**kwargs, "candidates": [(408_140.0, 0.20, 1)]}
        )
        is None
    )
    assert (
        _dense_prior_opening_candidate(
            **{**kwargs, "candidates": [(408_140.0, 0.313, 0)]}
        )
        is None
    )


def test_dense_later_opening_candidate_requires_pair_quality_and_text_gain():
    kwargs = {
        "previous_time": 332_800.0,
        "selected_time": 336_710.0,
        "next_time": 345_190.0,
        "expected_prev_gap_ms": 8_493.9,
        "expected_next_gap_ms": 3_893.0,
        "candidates": [(341_390.0, 0.582, 4)],
        "pause_reason": "dsp",
        "selected_sim": 0.356,
        "action": "observed_current",
    }

    assert _dense_later_opening_candidate(**kwargs) == 341_390.0
    assert (
        _dense_later_opening_candidate(
            **{**kwargs, "candidates": [(341_390.0, 0.582, 0)]}
        )
        is None
    )
    assert (
        _dense_later_opening_candidate(
            **{**kwargs, "candidates": [(341_390.0, 0.410, 4)]}
        )
        is None
    )


def test_dense_later_final_opening_candidate_uses_previous_span_only():
    kwargs = {
        "previous_time": 340_260.0,
        "selected_time": 345_470.0,
        "expected_gap_ms": 12_350.4,
        "candidates": [(349_990.0, 0.528, 7)],
        "pause_reason": "dsp",
        "selected_sim": 0.352,
        "action": "observed_current",
    }

    assert _dense_later_final_opening_candidate(**kwargs) == 349_990.0
    assert (
        _dense_later_final_opening_candidate(
            **{**kwargs, "candidates": [(349_990.0, 0.528, 0)]}
        )
        is None
    )
    assert (
        _dense_later_final_opening_candidate(
            **{**kwargs, "candidates": [(349_990.0, 0.40, 7)]}
        )
        is None
    )


def test_weak_dsp_later_prefix_candidate_moves_to_real_prefix_pause():
    kwargs = {
        "previous_time": 10_765.0,
        "selected_time": 13_000.0,
        "next_time": 19_020.0,
        "expected_prev_gap_ms": 2_970.3,
        "expected_next_gap_ms": 2_970.3,
        "candidates": [(16_220.0, 0.520, 0)],
        "pause_reason": "weak-dsp",
        "selected_sim": 0.325,
        "selected_prefix": 0,
        "action": "observed_current",
    }

    assert _weak_dsp_later_prefix_candidate(**kwargs) == 16_220.0
    assert (
        _weak_dsp_later_prefix_candidate(
            **{**kwargs, "next_time": 17_900.0}
        )
        is None
    )


def test_missing_wasl_pause_candidate_prefers_expected_pause_grid():
    kwargs = {
        "rough_ms": 761_630.0,
        "expected_ms": 754_500.0,
        "previous_time": 750_350.0,
        "next_time": 770_840.0,
        "candidates": [758_630.0, 771_270.0],
    }

    assert _missing_wasl_pause_candidate(**kwargs) == 758_630.0
    assert (
        _missing_wasl_pause_candidate(
            **{**kwargs, "candidates": [756_000.0]}
        )
        is None
    )
    assert (
        _missing_wasl_pause_candidate(
            **{**kwargs, "rough_ms": 759_000.0}
        )
        is None
    )


def test_accept_later_pause_grid_repair_requires_pause_grid_and_text_span():
    kwargs = {
        "from_pause": True,
        "original_reason": "micro-tempo",
        "original_sim": 0.204,
        "old_onset": 334_526.0,
        "repaired_onset": 339_178.0,
        "repaired_conf": 0.685,
        "nearest_pause_dist_ms": 102.0,
        "nearest_pause_reason": "weak-dsp",
        "previous_time": 331_080.0,
        "next_time": 347_360.0,
        "expected_prev_gap_ms": 8_348.0,
        "expected_next_gap_ms": 7_932.0,
        "conf_reliable": 0.55,
        "pause_trust_sim": 0.55,
    }

    assert _accept_later_pause_grid_repair(**kwargs)
    assert not _accept_later_pause_grid_repair(
        **{**kwargs, "nearest_pause_dist_ms": 700.0}
    )


def test_prior_pause_between_anchors_requires_internal_stop_and_pair_quality():
    kwargs = {
        "previous_time": 362_710.0,
        "selected_time": 372_970.0,
        "next_time": 381_510.0,
        "expected_prev_gap_ms": 6_400.0,
        "expected_next_gap_ms": 12_500.0,
        "strong_candidate_times": [369_040.0],
    }

    assert (
        _prior_pause_between_anchors(**kwargs, selected_is_internal_stop=True)
        == 369_040.0
    )
    assert _prior_pause_between_anchors(**kwargs, selected_is_internal_stop=False) is None


def test_internal_stop_repair_candidate_requires_micro_inside_pause():
    kwargs = {
        "from_pause": True,
        "action": "observed_current",
        "reason": "dsp",
        "sim": 0.466,
        "pause_duration_ms": 930.0,
        "prev_speech_ms": 2800.0,
        "next_speech_ms": 0.0,
        "has_micro_stop": True,
        "pause_trust_sim": 0.50,
    }

    assert _internal_stop_repair_candidate(**kwargs)
    assert not _internal_stop_repair_candidate(**{**kwargs, "has_micro_stop": False})
    assert not _internal_stop_repair_candidate(**{**kwargs, "sim": 0.55})


def test_reject_low_conf_off_grid_repair_only_for_pause_backed_shift():
    kwargs = {
        "from_pause": True,
        "repair_conf": 0.205,
        "conf_thresh": 0.25,
        "old_pause_dist_ms": 0.0,
        "chosen_pause_dist_ms": 1200.0,
        "shift_ms": 5500.0,
    }

    assert _reject_low_conf_off_grid_repair(**kwargs)
    assert not _reject_low_conf_off_grid_repair(**{**kwargs, "repair_conf": 0.4})
    assert not _reject_low_conf_off_grid_repair(**{**kwargs, "from_pause": False})


def test_accept_pause_backed_repair_rejects_forward_micro_steal():
    assert _accept_pause_backed_repair(
        from_pause=True,
        original_pause_reason="dsp",
        old_onset_ms=82_410.0,
        chosen_onset_ms=79_900.0,
        chosen_pause_dist_ms=40.0,
        chosen_pause_reason=None,
    )
    assert _accept_pause_backed_repair(
        from_pause=True,
        original_pause_reason="dsp",
        old_onset_ms=82_410.0,
        chosen_onset_ms=82_430.0,
        chosen_pause_dist_ms=1200.0,
        chosen_pause_reason="dsp",
    )
    assert _accept_pause_backed_repair(
        from_pause=True,
        original_pause_reason="dsp",
        old_onset_ms=82_410.0,
        chosen_onset_ms=85_162.0,
        chosen_pause_dist_ms=120.0,
        chosen_pause_reason="dsp",
    )
    assert _accept_pause_backed_repair(
        from_pause=False,
        original_pause_reason="micro-tempo",
        old_onset_ms=82_410.0,
        chosen_onset_ms=85_162.0,
        chosen_pause_dist_ms=1200.0,
        chosen_pause_reason="micro-tempo",
    )
    assert not _accept_pause_backed_repair(
        from_pause=True,
        original_pause_reason="dsp",
        old_onset_ms=82_410.0,
        chosen_onset_ms=85_162.0,
        chosen_pause_dist_ms=120.0,
        chosen_pause_reason="micro-tempo",
    )
    assert _accept_pause_backed_repair(
        from_pause=True,
        original_pause_reason="micro-tempo",
        old_onset_ms=225_963.0,
        chosen_onset_ms=223_373.0,
        chosen_pause_dist_ms=1600.0,
        chosen_pause_reason="weak-dsp",
    )
    assert not _accept_pause_backed_repair(
        from_pause=True,
        original_pause_reason="micro-tempo",
        old_onset_ms=82_410.0,
        chosen_onset_ms=85_162.0,
        chosen_pause_dist_ms=120.0,
        chosen_pause_reason="weak-dsp",
    )
    assert _accept_pause_backed_repair(
        from_pause=True,
        original_pause_reason="micro-tempo",
        old_onset_ms=334_527.0,
        chosen_onset_ms=339_178.0,
        chosen_pause_dist_ms=102.0,
        chosen_pause_reason="weak-dsp",
        later_pause_grid_repair=True,
    )
    assert not _accept_pause_backed_repair(
        from_pause=True,
        original_pause_reason="weak-dsp",
        old_onset_ms=181_015.0,
        chosen_onset_ms=178_643.0,
        chosen_pause_dist_ms=1300.0,
        chosen_pause_reason="weak-dsp",
    )


def test_final_weak_current_activation_window_handles_early_and_late_match():
    assert _final_weak_current_activation_window(
        expected_ms=777_336.0,
        matched_ms=761_830.0,
        matched_pause_start_ms=761_620.0,
        matched_pause_end_ms=762_040.0,
        total_ms=790_000.0,
        prior_window_ms=15_000.0,
    ) == (773_836.0, 780_836.0)
    assert _final_weak_current_activation_window(
        expected_ms=777_336.0,
        matched_ms=786_000.0,
        matched_pause_start_ms=785_800.0,
        matched_pause_end_ms=786_200.0,
        total_ms=790_000.0,
        prior_window_ms=15_000.0,
    ) == (773_836.0, 780_836.0)
    assert _final_weak_current_activation_window(
        expected_ms=777_336.0,
        matched_ms=778_000.0,
        matched_pause_start_ms=777_800.0,
        matched_pause_end_ms=778_200.0,
        total_ms=790_000.0,
        prior_window_ms=15_000.0,
    ) is None


def test_first_ayah_content_start_prefers_intro_pause_over_basmala_floor():
    assert (
        _first_ayah_content_start(
            3726.7,
            basmala=True,
            intro_pause_start_ms=3292.0,
            intro_pause_end_ms=3804.0,
            min_content_start_ms=6000.0,
        )
        == 3548.0
    )
    assert (
        _first_ayah_content_start(
            5352.0,
            basmala=True,
            intro_pause_start_ms=4568.0,
            intro_pause_end_ms=6136.0,
            min_content_start_ms=6000.0,
        )
        == 6136.0
    )
    assert (
        _first_ayah_content_start(
            3726.7,
            basmala=True,
            intro_pause_start_ms=None,
            intro_pause_end_ms=None,
            min_content_start_ms=6000.0,
        )
        == 6000.0
    )
    assert (
        _first_ayah_content_start(
            3726.7,
            basmala=False,
            intro_pause_start_ms=None,
            intro_pause_end_ms=None,
            min_content_start_ms=6000.0,
        )
        == 3726.7
    )


def test_pause_match_min_keep_sim_requires_normal_floor_for_recovered_micro():
    kwargs = {
        "drop_sim": 0.34,
        "min_match_sim": 0.31,
        "recover_echo_slack": 0.12,
        "final_weak_match_sim": 0.18,
    }

    assert _pause_match_min_keep_sim(
        action="recovered",
        pause_reason="micro-tempo",
        **kwargs,
    ) == 0.31
    assert _pause_match_min_keep_sim(
        action="recovered",
        pause_reason="dsp",
        **kwargs,
    ) == 0.19
    assert _pause_match_min_keep_sim(
        action="pause_fallback",
        pause_reason="micro-tempo",
        **kwargs,
    ) == 0.31


def test_accept_fallback_pause_candidate_rejects_local_internal_micro():
    assert not _accept_fallback_pause_candidate(
        pause_reason="micro-tempo",
        sim=0.24,
        min_match_sim=0.31,
        expected_delta_ms=-1819.2,
        prev_quality=0.316,
        next_quality=0.858,
    )
    assert not _accept_fallback_pause_candidate(
        pause_reason="micro-tempo",
        sim=0.26,
        min_match_sim=0.31,
        expected_delta_ms=518.7,
        prev_quality=0.878,
        next_quality=0.0,
        next_speech_ms=1172.0,
    )
    assert _accept_fallback_pause_candidate(
        pause_reason="micro-tempo",
        sim=0.26,
        min_match_sim=0.31,
        expected_delta_ms=64_796.7,
        prev_quality=0.878,
        next_quality=0.0,
        next_speech_ms=1172.0,
    )
    assert _accept_fallback_pause_candidate(
        pause_reason="micro-tempo",
        sim=0.24,
        min_match_sim=0.31,
        expected_delta_ms=-64796.7,
        prev_quality=0.0,
        next_quality=0.806,
    )
    assert _accept_fallback_pause_candidate(
        pause_reason="micro-tempo",
        sim=0.31,
        min_match_sim=0.31,
        expected_delta_ms=-1819.2,
        prev_quality=0.316,
        next_quality=0.858,
    )
    assert _accept_fallback_pause_candidate(
        pause_reason="dsp",
        sim=0.24,
        min_match_sim=0.31,
        expected_delta_ms=-1819.2,
        prev_quality=0.316,
        next_quality=0.858,
    )


def test_section_long_pause_start_candidate_requires_tail_of_long_dsp():
    assert (
        _section_long_pause_start_candidate(
            708_906.0,
            706_260.0,
            709_080.0,
            pause_reason="dsp",
        )
        == 706_260.0
    )
    assert (
        _section_long_pause_start_candidate(
            707_670.0,
            706_260.0,
            709_080.0,
            pause_reason="dsp",
        )
        is None
    )
    assert (
        _section_long_pause_start_candidate(
            708_906.0,
            708_400.0,
            709_080.0,
            pause_reason="dsp",
        )
        is None
    )


def test_long_pause_start_timing_candidate_requires_trusted_shelf_boundary():
    assert _long_pause_start_timing_candidate(
        pause_duration_ms=2060.0,
        prev_speech_ms=0.0,
        next_speech_ms=180.0,
        sim=0.537,
        pause_trust_sim=0.50,
    )
    assert not _long_pause_start_timing_candidate(
        pause_duration_ms=2060.0,
        prev_speech_ms=900.0,
        next_speech_ms=180.0,
        sim=0.537,
        pause_trust_sim=0.50,
    )
    assert not _long_pause_start_timing_candidate(
        pause_duration_ms=2060.0,
        prev_speech_ms=0.0,
        next_speech_ms=180.0,
        sim=0.3,
        pause_trust_sim=0.50,
    )


def test_long_pause_start_after_speech_candidate_requires_speech_then_silence():
    assert _long_pause_start_after_speech_candidate(
        pause_duration_ms=1840.0,
        prev_speech_ms=7220.0,
        next_speech_ms=0.0,
        sim=0.509,
        min_sim=0.34,
    )
    assert not _long_pause_start_after_speech_candidate(
        pause_duration_ms=1840.0,
        prev_speech_ms=7220.0,
        next_speech_ms=800.0,
        sim=0.509,
        min_sim=0.34,
    )
    assert not _long_pause_start_after_speech_candidate(
        pause_duration_ms=1200.0,
        prev_speech_ms=7220.0,
        next_speech_ms=0.0,
        sim=0.509,
        min_sim=0.34,
    )


def test_long_pause_start_balanced_speech_candidate_requires_text_span_quality():
    kwargs = {
        "pause_duration_ms": 5540.0,
        "prev_speech_ms": 4880.0,
        "next_speech_ms": 5140.0,
        "prev_quality": 0.865,
        "next_quality": 0.736,
        "sim": 0.371,
        "min_sim": 0.34,
        "max_sim": 0.50,
        "action": "pause_fallback",
    }

    assert _long_pause_start_balanced_speech_candidate(**kwargs)
    assert not _long_pause_start_balanced_speech_candidate(
        **{**kwargs, "next_quality": 0.245}
    )
    assert not _long_pause_start_balanced_speech_candidate(
        **{**kwargs, "action": "opening_run_repair"}
    )


def test_compact_dsp_start_candidate_requires_balanced_recovered_pause():
    kwargs = {
        "pause_reason": "dsp",
        "pause_duration_ms": 1060.0,
        "prev_quality": 0.931,
        "next_quality": 0.799,
        "sim": 0.471,
        "min_sim": 0.34,
        "action": "recovered",
    }

    assert _compact_dsp_start_candidate(**kwargs)
    assert not _compact_dsp_start_candidate(**{**kwargs, "next_quality": 0.45})
    assert not _compact_dsp_start_candidate(**{**kwargs, "pause_duration_ms": 1300.0})
    assert not _compact_dsp_start_candidate(**{**kwargs, "action": "pause_fallback"})


def test_compact_dsp_end_candidate_requires_short_balanced_recovered_pause():
    kwargs = {
        "pause_reason": "dsp",
        "pause_duration_ms": 880.0,
        "prev_speech_ms": 3860.0,
        "next_speech_ms": 3700.0,
        "prev_quality": 0.909,
        "next_quality": 0.863,
        "sim": 0.415,
        "action": "recovered",
    }

    assert _compact_dsp_end_candidate(**kwargs)
    assert not _compact_dsp_end_candidate(**{**kwargs, "next_speech_ms": 4440.0})
    assert not _compact_dsp_end_candidate(**{**kwargs, "next_quality": 0.772})
    assert not _compact_dsp_end_candidate(**{**kwargs, "sim": 0.50})
    assert not _compact_dsp_end_candidate(**{**kwargs, "action": "confirmed"})


def test_dsp_edge_timing_candidate_uses_neighbor_span_balance():
    start_kwargs = {
        "pause_reason": "dsp",
        "pause_duration_ms": 1320.0,
        "prev_speech_ms": 5360.0,
        "next_speech_ms": 4340.0,
        "prev_quality": 0.624,
        "next_quality": 0.454,
        "sim": 0.576,
        "min_sim": 0.34,
        "action": "weak_pause_recovery",
    }
    end_kwargs = {
        "pause_reason": "dsp",
        "pause_duration_ms": 1700.0,
        "prev_speech_ms": 2560.0,
        "next_speech_ms": 4820.0,
        "prev_quality": 0.462,
        "next_quality": 0.724,
        "sim": 0.477,
        "min_sim": 0.34,
        "action": "recovered",
    }

    assert _dsp_edge_timing_candidate(**start_kwargs) == "start"
    assert _dsp_edge_timing_candidate(**end_kwargs) == "end"
    assert _dsp_edge_timing_candidate(**{**start_kwargs, "next_quality": 0.55}) is None
    assert _dsp_edge_timing_candidate(**{**end_kwargs, "prev_quality": 0.35}) is None
    assert _dsp_edge_timing_candidate(**{**end_kwargs, "action": "confirmed"}) is None

    medium_waqf_end = {
        "pause_reason": "dsp",
        "pause_duration_ms": 1420.0,
        "prev_speech_ms": 4520.0,
        "next_speech_ms": 5780.0,
        "prev_quality": 0.222,
        "next_quality": 0.436,
        "waqf_before": 1,
        "waqf_after": 0,
        "sim": 0.417,
        "min_sim": 0.34,
        "action": "recovered",
    }
    assert _dsp_edge_timing_candidate(**medium_waqf_end) == "end"
    assert _dsp_edge_timing_candidate(
        **{**medium_waqf_end, "waqf_before": 0}
    ) is None
    assert _dsp_edge_timing_candidate(
        **{**medium_waqf_end, "next_quality": 0.78}
    ) is None


def test_preword_micro_timing_candidate_requires_confirmed_pre_pause_word():
    kwargs = {
        "pause_reason": "dsp",
        "pause_start_ms": 1_145_560.0,
        "refined_onset_ms": 1_144_451.0,
        "refined_conf": 0.452,
        "conf_thresh": 0.40,
        "sim": 0.333,
        "min_sim": 0.31,
    }

    assert _preword_micro_timing_candidate(**kwargs)
    assert not _preword_micro_timing_candidate(
        **{**kwargs, "refined_onset_ms": 1_145_200.0}
    )
    assert not _preword_micro_timing_candidate(**{**kwargs, "refined_conf": 0.30})
    assert not _preword_micro_timing_candidate(**{**kwargs, "pause_reason": "weak-dsp"})


def test_weak_dsp_prior_recovery_candidate_requires_short_internal_blip():
    kwargs = {
        "pause_reason": "weak-dsp",
        "action": "recovered",
        "prev_speech_ms": 320.0,
        "next_speech_ms": 7240.0,
        "next_quality": 0.861,
    }

    assert _weak_dsp_prior_recovery_candidate(**kwargs)
    assert not _weak_dsp_prior_recovery_candidate(
        **{**kwargs, "prev_speech_ms": 900.0}
    )
    assert not _weak_dsp_prior_recovery_candidate(
        **{**kwargs, "action": "confirmed"}
    )
    assert not _weak_dsp_prior_recovery_candidate(
        **{**kwargs, "next_quality": 0.55}
    )


def test_weak_dsp_prior_trusted_recovery_candidate_requires_tail_shape_and_waqf():
    kwargs = {
        "pause_reason": "weak-dsp",
        "action": "recovered",
        "prev_speech_ms": 1080.0,
        "next_speech_ms": 5480.0,
        "prev_quality": 0.0,
        "next_quality": 0.878,
        "waqf_after": 1,
    }

    assert _weak_dsp_prior_trusted_recovery_candidate(**kwargs)
    assert not _weak_dsp_prior_trusted_recovery_candidate(
        **{**kwargs, "waqf_after": 0}
    )
    assert not _weak_dsp_prior_trusted_recovery_candidate(
        **{**kwargs, "prev_quality": 0.20}
    )
    assert not _weak_dsp_prior_trusted_recovery_candidate(
        **{**kwargs, "prev_speech_ms": 2500.0}
    )


def test_rejected_micro_valley_candidate_requires_quiet_valley_and_span_quality():
    kwargs = {
        "pause_reason": "rejected",
        "action": "recovered",
        "valley_ratio": 0.145,
        "shift_ms": 981.0,
        "selected_prev_quality": 0.954,
        "selected_next_quality": 0.802,
        "candidate_prev_quality": 0.804,
        "candidate_next_quality": 0.941,
    }

    assert _rejected_micro_valley_candidate(**kwargs)
    assert not _rejected_micro_valley_candidate(
        **{**kwargs, "pause_reason": "weak-dsp"}
    )
    assert not _rejected_micro_valley_candidate(
        **{**kwargs, "valley_ratio": 0.25}
    )
    assert not _rejected_micro_valley_candidate(
        **{**kwargs, "candidate_prev_quality": 0.60}
    )


def test_weak_dsp_expected_micro_valley_requires_expected_gain_and_quiet_shelf():
    accepted = {
        "pause_reason": "weak-dsp",
        "action": "recovered",
        "pause_sim": 0.293,
        "valley_ratio": 0.117,
        "shift_ms": 1424.0,
        "expected_gain_ms": 726.0,
        "selected_prev_quality": 0.793,
        "selected_next_quality": 0.967,
        "candidate_prev_quality": 0.940,
        "candidate_next_quality": 0.767,
    }
    late_wrong = {
        **accepted,
        "pause_sim": 0.314,
        "valley_ratio": 0.051,
        "shift_ms": 1979.0,
        "expected_gain_ms": -359.0,
        "selected_prev_quality": 0.789,
        "selected_next_quality": 0.456,
        "candidate_prev_quality": 0.753,
        "candidate_next_quality": 0.774,
    }

    assert _weak_dsp_expected_micro_valley_candidate(**accepted)
    assert not _weak_dsp_expected_micro_valley_candidate(**late_wrong)
    assert not _weak_dsp_expected_micro_valley_candidate(
        **{**accepted, "pause_reason": "rejected"}
    )
    assert not _weak_dsp_expected_micro_valley_candidate(
        **{**accepted, "valley_ratio": 0.20}
    )
    assert not _weak_dsp_expected_micro_valley_candidate(
        **{**accepted, "candidate_next_quality": 0.60}
    )


def test_weak_dsp_post_selected_micro_valley_requires_two_sided_span_gain():
    accepted = {
        "pause_reason": "weak-dsp",
        "action": "recovered",
        "pause_sim": 0.271,
        "valley_ratio": 0.118,
        "forward_shift_ms": 1532.0,
        "selected_prev_quality": 0.629,
        "selected_next_quality": 0.337,
        "candidate_prev_quality": 0.776,
        "candidate_next_quality": 0.696,
    }

    assert _weak_dsp_post_selected_micro_valley_candidate(**accepted)
    assert not _weak_dsp_post_selected_micro_valley_candidate(
        **{**accepted, "forward_shift_ms": 300.0}
    )
    assert not _weak_dsp_post_selected_micro_valley_candidate(
        **{**accepted, "selected_next_quality": 0.90}
    )
    assert not _weak_dsp_post_selected_micro_valley_candidate(
        **{**accepted, "candidate_prev_quality": 0.65}
    )
    assert not _weak_dsp_post_selected_micro_valley_candidate(
        **{**accepted, "valley_ratio": 0.20}
    )


def test_weak_dsp_wasl_recovery_candidate_requires_mixed_tail_and_current():
    kwargs = {
        "pause_reason": "weak-dsp",
        "action": "recovered",
        "prev_speech_ms": 2520.0,
        "next_speech_ms": 2340.0,
        "prev_quality": 0.343,
        "next_quality": 0.984,
        "sim": 0.314,
        "min_sim": 0.31,
    }

    assert _weak_dsp_wasl_recovery_candidate(**kwargs)
    assert not _weak_dsp_wasl_recovery_candidate(
        **{**kwargs, "prev_speech_ms": 320.0}
    )
    assert not _weak_dsp_wasl_recovery_candidate(
        **{**kwargs, "sim": 0.417}
    )
    assert not _weak_dsp_wasl_recovery_candidate(
        **{**kwargs, "prev_quality": 0.70}
    )


def test_long_dsp_fractional_start_candidate_is_narrow():
    opening = {
        "pause_reason": "dsp",
        "pause_duration_ms": 3620.0,
        "prev_quality": 0.239,
        "next_quality": 0.245,
        "waqf_before": 0,
        "waqf_after": 0,
        "sim": 0.335,
        "min_sim": 0.34,
        "action": "opening_run_repair",
    }
    balanced = {
        "pause_reason": "dsp",
        "pause_duration_ms": 5540.0,
        "prev_quality": 0.865,
        "next_quality": 0.736,
        "waqf_before": 0,
        "waqf_after": 0,
        "sim": 0.371,
        "min_sim": 0.34,
        "action": "pause_fallback",
    }

    assert _long_dsp_fractional_start_candidate(**opening) is None
    assert _long_dsp_fractional_start_candidate(**{**opening, "min_sim": 0.31}) == 0.30
    assert _long_dsp_fractional_start_candidate(**balanced) == 0.12
    assert _long_dsp_fractional_start_candidate(
        **{**balanced, "next_quality": 0.55}
    ) is None
    assert (
        _long_dsp_fractional_start_candidate(
            **{
                **balanced,
                "prev_quality": 0.378,
                "next_quality": 0.361,
                "sim": 0.403,
                "action": "observed_current",
            }
        )
        == 0.12
    )
    assert (
        _long_dsp_fractional_start_candidate(
            **{
                **balanced,
                "pause_duration_ms": 3260.0,
                "prev_quality": 0.126,
                "next_quality": 0.853,
                "sim": 0.466,
                "action": "observed_current",
            }
        )
        == 0.85
    )
    assert (
        _long_dsp_fractional_start_candidate(
            **{
                **balanced,
                "pause_duration_ms": 2440.0,
                "prev_quality": 0.821,
                "next_quality": 0.532,
                "sim": 0.442,
                "action": "recovered",
            }
        )
        == 0.85
    )
    assert (
        _long_dsp_fractional_start_candidate(
            **{
                **balanced,
                "pause_duration_ms": 880.0,
                "prev_quality": 0.909,
                "next_quality": 0.863,
                "sim": 0.415,
                "action": "recovered",
            }
        )
        is None
    )
    assert (
        _long_dsp_fractional_start_candidate(
            **{
                **balanced,
                "pause_duration_ms": 2640.0,
                "prev_quality": 0.694,
                "next_quality": 0.0,
                "waqf_before": 1,
                "waqf_after": 1,
                "sim": 0.351,
                "action": "recovered",
            }
        )
        == 0.85
    )
    assert (
        _long_dsp_fractional_start_candidate(
            **{
                **balanced,
                "pause_duration_ms": 2640.0,
                "prev_quality": 0.694,
                "next_quality": 0.0,
                "waqf_before": 1,
                "waqf_after": 0,
                "sim": 0.351,
                "action": "recovered",
            }
        )
        is None
    )
    assert (
        _long_dsp_fractional_start_candidate(
            **{
                **balanced,
                "pause_duration_ms": 1420.0,
                "prev_quality": 0.041,
                "next_quality": 0.777,
                "sim": 0.342,
                "action": "recovered",
            }
        )
        is None
    )


def test_short_low_sim_dsp_start_candidate_requires_internal_tail_shape():
    kwargs = {
        "pause_reason": "dsp",
        "pause_duration_ms": 500.0,
        "prev_quality": 0.858,
        "next_speech_ms": 680.0,
        "sim": 0.048,
        "action": "weak_pause_recovery",
    }

    assert _short_low_sim_dsp_start_candidate(**kwargs)
    assert not _short_low_sim_dsp_start_candidate(**{**kwargs, "sim": 0.25})
    assert not _short_low_sim_dsp_start_candidate(
        **{**kwargs, "next_speech_ms": 1200.0}
    )
    assert not _short_low_sim_dsp_start_candidate(
        **{**kwargs, "action": "confirmed"}
    )


def test_late_repeated_dsp_prior_candidate_requires_internal_repeat_shape():
    kwargs = {
        "pause_reason": "dsp",
        "pause_duration_ms": 4920.0,
        "prev_speech_ms": 6840.0,
        "next_speech_ms": 1380.0,
        "sim": 0.711,
        "action": "observed_current",
        "min_sim": 0.50,
    }

    assert _late_repeated_dsp_prior_candidate(**kwargs)
    assert not _late_repeated_dsp_prior_candidate(
        **{**kwargs, "next_speech_ms": 2500.0}
    )
    assert not _late_repeated_dsp_prior_candidate(
        **{**kwargs, "prev_speech_ms": 3000.0}
    )
    assert not _late_repeated_dsp_prior_candidate(
        **{**kwargs, "action": "pause_fallback"}
    )


def test_left_edge_repeated_opening_candidate_requires_strong_left_edge_inside_shelf():
    assert _left_edge_repeated_opening_candidate(
        [
            (571_392.9, 0.127, False),
            (569_441.2, 0.605, True),
            (569_851.3, 0.510, True),
            (573_268.1, 0.190, False),
        ],
        pause_start_ms=568_920.0,
        pause_center_ms=571_200.0,
        min_conf=0.45,
    ) == (569_851.3, 0.510)
    assert _left_edge_repeated_opening_candidate(
        [
            (607_012.4, 0.023, True),
            (609_468.0, 0.055, False),
        ],
        pause_start_ms=606_060.0,
        pause_center_ms=607_870.0,
        min_conf=0.45,
    ) is None


def test_prior_close_dsp_before_micro_stop_picks_immediate_strong_pause():
    assert (
        _prior_close_dsp_before_micro_stop(
            146_598.6,
            [137_050.0, 146_280.0, 156_230.0],
        )
        == 146_280.0
    )
    assert _prior_close_dsp_before_micro_stop(146_598.6, [145_000.0]) is None


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


def test_opening_internal_repeats_finds_previous_ayah_suffix():
    openings = [
        "waqālalladhīnaāmanū",
        "innalladhīnaāmanūwaʿamilūṣāliḥāti",
    ]
    previous_bodies = [
        "previousbody",
        "qālalmalauinnalladhīnaāmanūthummaakhbarū",
    ]

    repeats = _opening_internal_repeats(openings, previous_bodies, prefix_chars=8)

    assert 0 not in repeats
    assert repeats[1][0] >= 10
    assert repeats[1][1].startswith(_collapse("innalladhīnaāmanū"))


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


def test_sequential_pause_match_recovery_prefers_early_observed_current_over_late_micro():
    exp = [23_000.0, 29_264.0, 37_150.0]
    ons = [23_110.0, 27_390.0, 31_556.0, 37_150.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (1, 1): 0.257,
            (1, 2): 0.342,
            (2, 3): 0.90,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
        lookahead=1,
    )

    assert matched[1] == 27_390.0
    assert matched[1] != 31_556.0
    assert any(e["action"] == "recovered" and e["boundary"] == 1 for e in events)


def test_sequential_pause_match_observed_current_prefers_acoustic_sim_over_prior():
    exp = [10_171.8, 15_112.2, 18_641.0, 23_581.5]
    ons = [10_920.0, 14_270.0, 16_790.0, 20_400.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (1, 1): 0.386,
            (1, 2): 0.352,
            (1, 3): 0.200,
            (2, 2): 0.430,
            (2, 3): 0.100,
            (3, 3): 0.532,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.34,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
        lookahead=1,
    )

    assert matched[1] == 14_270.0
    assert matched[1] != 16_790.0
    assert matched[2] == 16_790.0
    assert any(
        e["action"] in {"observed_current", "recovered"}
        and e["boundary"] == 1
        and e["pause"] == 1
        for e in events
    )


def test_sequential_pause_match_observed_current_ignores_tiny_far_sim_gain():
    exp = [16_402.9]
    ons = [17_910.0, 23_490.0, 26_880.0, 33_690.0]

    def sim(i: int, j: int) -> float:
        return [0.426, 0.344, 0.364, 0.451][j]

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
    )

    assert matched == [17_910.0]
    assert any(
        e["action"] == "observed_current"
        and e["boundary"] == 0
        and e["pause"] == 0
        for e in events
    )


def test_sequential_pause_match_keeps_near_current_before_multi_ayah_future_jump():
    exp = [22_908.3, 27_916.6, 32_014.3]
    ons = [21_990.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.315,
            (2, 0): 0.442,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.31,
        confirm_min_sim=0.50,
        future_margin=0.12,
        prior_window_ms=15_000.0,
        lookahead=2,
    )

    assert matched[0] == 21_990.0
    assert matched[2] is None
    assert not any(
        e["action"] == "future_gap" and e["boundary"] == 0
        for e in events
    )


def test_sequential_pause_match_recovery_keeps_near_current_future_echo():
    exp = [18_030.7, 22_908.3, 27_916.6, 32_014.3]
    ons = [17_900.0, 21_990.0, 26_530.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.453,
            (0, 1): 0.163,
            (1, 1): 0.298,
            (1, 2): 0.213,
            (2, 2): 0.500,
            (3, 1): 0.404,
        }
        return scores.get((i, j), 0.02)

    def score_bonus(i: int, j: int) -> float:
        if (i, j) == (1, 1):
            return 0.063
        return 0.0

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        future_margin=0.12,
        prior_window_ms=15_000.0,
        lookahead=2,
        score_bonus=score_bonus,
    )

    assert matched[1] == 21_990.0
    assert any(
        e["action"] == "recovered"
        and e["boundary"] == 1
        and e["pause"] == 1
        for e in events
    )
    assert not any(
        e["action"] == "skip_recovery_observed_mismatch"
        and e["boundary"] == 1
        and e["pause"] == 1
        for e in events
    )


def test_sequential_pause_match_recovers_chain_before_future_gap():
    exp = [1000.0, 2000.0, 3000.0]
    ons = [980.0, 2010.0, 3005.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.35,
            (0, 2): 0.10,
            (1, 1): 0.36,
            (1, 2): 0.10,
            (2, 2): 0.80,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=1500.0,
        lookahead=2,
    )

    assert matched == [980.0, 2010.0, 3005.0]
    assert any(e["action"] == "recovered" and e["boundary"] == 0 for e in events)
    assert any(e["action"] == "recovered" and e["boundary"] == 1 for e in events)


def test_sequential_pause_match_keeps_far_observed_current_candidate():
    exp = [1000.0]
    ons = [1100.0, 1400.0]

    def sim(_i: int, j: int) -> float:
        return [0.39, 0.40][j]

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=1000.0,
        reject=lambda _i, _j, _s: "far_from_expected",
    )

    assert matched == [1100.0]
    assert any(e["action"] == "observed_current_candidate" for e in events)


def test_sequential_pause_match_rejects_far_weak_observed_current():
    exp = [1000.0, 2000.0]
    ons = [11_000.0, 12_000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.34,
            (1, 1): 0.80,
        }
        return scores.get((i, j), 0.01)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
    )

    assert matched[0] is None
    assert not any(e["action"] == "observed_current" for e in events)


def test_sequential_pause_match_recovery_rejects_far_weak_tempo_bonus():
    exp = [1000.0, 2000.0]
    ons = [11_000.0, 12_000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.30,
            (1, 1): 0.80,
        }
        return scores.get((i, j), 0.01)

    def bonus(i: int, j: int) -> float:
        return 0.08 if (i, j) == (0, 0) else 0.0

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
        score_bonus=bonus,
    )

    assert matched[0] is None
    assert any(
        e["action"] == "skip_recovery_low_sim"
        and e["boundary"] == 0
        and e["pause"] == 0
        for e in events
    )


def test_sequential_pause_match_prefers_early_current_before_late_confirmed_repeat():
    exp = [1000.0]
    ons = [1000.0, 10_000.0]

    def sim(_i: int, j: int) -> float:
        return [0.475, 0.527][j]

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
    )

    assert matched == [1000.0]
    assert any(e["action"] == "observed_current" for e in events)


def test_sequential_pause_match_uses_pause_order_before_far_confirmed_repeat():
    exp = [470_000.0, 480_000.0, 490_000.0]
    ons = [470_000.0, 480_150.0, 505_000.0, 515_000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (1, 1): 0.29,  # weak text, but the pause is at the current boundary
            (1, 3): 0.55,  # stronger repeated opening far ahead
            (2, 2): 0.20,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
    )

    assert matched[1] == 480_150.0
    assert matched[1] != 515_000.0
    assert any(e["action"] == "pause_order_current_candidate" for e in events)
    assert any(
        e["action"] in {"weak_observed_current", "observed_current", "recovered"}
        and e["boundary"] == 1
        and e["pause"] == 1
        for e in events
    )


def test_sequential_pause_match_recovery_keeps_prior_closer_immediate_future_echo():
    exp = [90_432.8, 95_659.0, 99_051.0]
    ons = [89_920.0, 99_210.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.287,
            (1, 0): 0.354,
            (2, 1): 0.850,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        future_margin=0.12,
        prior_window_ms=15_000.0,
        lookahead=3,
    )

    assert matched[0] == 89_920.0
    assert any(
        e["action"] == "recovered"
        and e["boundary"] == 0
        and e["pause"] == 0
        for e in events
    )


def test_sequential_pause_match_accepts_final_weak_current():
    exp = [111_622.9]
    ons = [114_270.0]

    def sim(_i: int, _j: int) -> float:
        return 0.191

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
    )

    assert matched == [114_270.0]
    assert any(e["action"] == "final_weak_current" for e in events)


def test_sequential_pause_match_rejects_final_weak_current_against_tempo():
    exp = [166_676.6]
    ons = [164_288.0]

    def sim(_i: int, _j: int) -> float:
        return 0.191

    def score_bonus(_i: int, _j: int) -> float:
        return -0.08

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
        score_bonus=score_bonus,
    )

    assert matched == [None]
    assert not any(e["action"] == "final_weak_current" for e in events)


def test_sequential_pause_match_accepts_final_weak_current_near_neutral_tempo():
    exp = [242_465.1]
    ons = [241_350.0]

    def sim(_i: int, _j: int) -> float:
        return 0.214

    def score_bonus(_i: int, _j: int) -> float:
        return -0.008

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
        score_bonus=score_bonus,
    )

    assert matched == [241_350.0]
    assert any(e["action"] == "final_weak_current" for e in events)


def test_sequential_pause_match_accepts_near_weak_observed_current():
    exp = [204_332.0, 208_307.0]
    ons = [202_120.0, 207_570.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.279,
            (1, 1): 0.300,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
    )

    assert matched[0] == 202_120.0
    assert any(e["action"] == "weak_observed_current" for e in events)


def test_sequential_pause_match_recovery_keeps_local_current_over_global_false_match():
    exp = [1000.0, 2000.0, 3000.0]
    ons = [1000.0, 2000.0, 3000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.90,
            (0, 1): 0.53,  # stronger global false match for the recovery pause
            (1, 1): 0.48,
            (1, 2): 0.10,
            (2, 2): 0.90,
        }
        return scores.get((i, j), 0.02)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.34,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=1500.0,
        lookahead=1,
    )

    assert matched == ons
    assert any(e["action"] == "recovered" and e["boundary"] == 1 for e in events)
    assert not any(
        e["action"] == "skip_recovery_observed_mismatch"
        and e["boundary"] == 1
        and e["pause"] == 1
        for e in events
    )


def test_sequential_pause_match_keeps_local_current_over_global_stale():
    exp = [1_000.0, 2_000.0, 3_000.0]
    ons = [1_000.0, 2_050.0, 3_000.0]

    def sim(i: int, j: int) -> float:
        scores = {
            (0, 0): 0.70,
            (0, 1): 0.50,
            (1, 1): 0.27,
            (1, 2): 0.10,
            (2, 2): 0.70,
        }
        return scores.get((i, j), 0.0)

    matched, events = _sequential_pause_match(
        exp,
        ons,
        sim,
        min_sim=0.31,
        recover_min_sim=0.34,
        confirm_min_sim=0.50,
        prior_window_ms=15_000.0,
    )

    assert matched == [1_000.0, 2_050.0, 3_000.0]
    assert any(
        e["action"] == "observed_current_candidate"
        and e.get("reason") == "local_current_over_global_stale"
        for e in events
    )


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
    assert any(
        e["action"] == "wasl"
        and e["boundary"] == 0
        and e.get("reason") == "unresolved_current"
        for e in events
    )
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
