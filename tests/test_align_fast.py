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
    PauseReadPlan,
    TempoEstimate,
    TempoObservation,
    TextDisambiguation,
    _accept_anchored_full_span_onset,
    _accept_fallback_pause_candidate,
    _accept_later_pause_grid_repair,
    _accept_pause_backed_repair,
    _adjacent_repeated_owners_start_shifted_section,
    _ambiguous_multiscale_owner_requires_full_span,
    _ambiguous_wasl_run_requires_bracket_extension,
    _confirmed_weak_pause_repair_candidate,
    _pre_owner_ctc_overrides_weak_micro_pause,
    _confirmed_trusted_pause_repair_candidate,
    _confirmed_trusted_pause_refine_candidate,
    _ambiguous_openings,
    _ambiguous_observed_pause_starts_shifted_section,
    _assign_opening_block,
    _ayah_waqf_counts,
    _collapse,
    _build_pause_read_plan,
    _build_text_disambiguation,
    _bounded_local_repair_spans,
    _ctc_prior_pause_replaces_weak_future_owner,
    _common_prefix_len,
    _compact_dsp_start_candidate,
    _compact_dsp_end_candidate,
    _context_block_pause_onset_candidate,
    _count_waqf_marks,
    _competing_prior_pause_requires_full_span,
    _dense_prior_opening_candidate,
    _direct_owner_conflicts_with_anchor_geometry,
    _dense_later_opening_candidate,
    _dense_later_final_opening_candidate,
    _dsp_edge_timing_candidate,
    _find_micro_pause_valley,
    _final_weak_current_activation_window,
    _first_ayah_content_start,
    _full_span_pause_agrees_with_local_read,
    _full_span_following_support_is_eligible,
    _full_span_opening_support_confidence,
    _full_span_right_bracket_ms,
    _full_wasl_read_classification,
    _insufficient_audio_ayahs,
    _requires_low_verify_ctc_fallback,
    _full_span_pause_candidate_is_supported,
    _full_span_pause_owner_is_consistent,
    _full_span_unique_next_pause_candidate,
    _full_span_weak_pause_is_jointly_supported,
    _first_opening_owner_conflicts_with_anchor_geometry,
    _multiscale_ctc_in_owner_candidate,
    _multiscale_pause_repair_is_eligible,
    _owner_interval_snap_is_allowed,
    _preserve_multiscale_ctc_owner_timing,
    _stable_multiscale_pause_onset,
    _stable_multiscale_onset_cluster,
    _geometry_invalidated_owner_requires_full_span,
    _geometry_invalidated_full_span_weak_pause_is_supported,
    _geometry_invalidated_repair_can_compete,
    _geometry_repair_is_already_consistent,
    _recovered_owner_conflicts_with_anchor_geometry,
    _repeated_low_trust_left_anchor_extends_atomic_block,
    _recovered_later_best_pause_candidate,
    _internal_stop_repair_candidate,
    _insufficient_wasl_read_invalidates_right_anchor,
    _interp_wasl,
    _left_edge_repeated_opening_candidate,
    _late_repeated_dsp_prior_candidate,
    _long_dsp_fractional_start_candidate,
    _late_long_dsp_fallback_start_candidate,
    _overlapping_dsp_replaces_micro_owner,
    _long_pause_context_read_candidate,
    _long_pause_start_after_speech_candidate,
    _long_pause_start_balanced_speech_candidate,
    _long_pause_start_timing_candidate,
    _shifted_geometry_long_pause_start_candidate,
    _match_boundaries,
    _missing_wasl_pause_candidate,
    _onset_prefix_len,
    _opening_internal_repeats,
    _opening_shared_prefix_chars,
    _micro_shelf_start_candidate,
    _micro_owner_prior_prefix_candidate,
    _prefer_expected_micro_valley,
    _pause_backed_refine_start,
    _pause_center_start,
    _pause_center_with_refined_onset,
    _pause_match_min_keep_sim,
    _pause_owner_geometry_is_unreliable,
    _pause_owner_geometry_needs_full_span,
    _pause_refine_confirms_pause,
    _post_pause_repair_confirms_prior_dsp,
    _post_selected_micro_context_read_candidate,
    _preword_micro_timing_candidate,
    _allow_prior_internal_stop_replacement,
    _prior_pause_before_internal_stop,
    _prior_pause_between_anchors,
    _prefer_competing_prior_pause_extended_decode,
    _prefer_competing_prior_pause_pre_context,
    _prior_close_dsp_before_micro_stop,
    _repeated_opening_pause_anchor_is_ambiguous,
    _repeated_wasl_prior_pause_candidate,
    _single_wasl_prior_pause_candidate,
    _revised_wasl_right_anchor_requires_predecessor_reread,
    _restore_owner_timing_from_internal_repair,
    _restore_trusted_owner_after_predecessor_repair,
    _rejected_micro_valley_candidate,
    _reject_low_conf_off_grid_repair,
    _sequential_pause_match,
    _section_repair_member,
    _follows_unresolved_cascade,
    _section_long_pause_start_candidate,
    _enforce_strictly_increasing_starts,
    _estimate_tempo,
    _fast_boundary_diagnostics,
    _skips_dense_pause_block,
    _sim,
    _snap_pause_backed_start_to_owner,
    _short_low_sim_dsp_start_candidate,
    _span_ratio_quality,
    _speech_ms_excluding_pauses,
    _text_disambiguation_requires_atomic,
    _unit_len,
    _unclassified_owner_requires_full_span,
    _unclassified_repeated_pause_run_requires_atomic_block,
    _very_long_pause_repair_context_candidate,
    _very_long_context_owner_still_valid,
    _weak_dsp_prior_recovery_candidate,
    _weak_dsp_prior_trusted_recovery_candidate,
    _weak_dsp_later_prefix_candidate,
    _weak_dsp_expected_micro_valley_candidate,
    _weak_dsp_post_selected_micro_valley_candidate,
    _wasl_revision_invalidates_fallback_right_anchor,
    _weak_dsp_wasl_recovery_candidate,
    _weak_recovered_compact_pause_center_candidate,
    _weak_recovered_long_pause_start_candidate,
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


def test_text_disambiguation_reads_through_first_different_word():
    plan = _build_text_disambiguation(
        2,
        {
            2: ["innā", "aʿṭaynāka", "alkawthar"],
            3: ["innā", "anzalnāhu"],
            4: ["qul", "huwa"],
        },
        competitor_ayahs=[3, 4],
    )

    assert isinstance(plan, TextDisambiguation)
    assert plan.reason == "shared_opening"
    assert plan.required_words == 2
    assert plan.distinguishing_word == "aʿṭaynāka"
    assert plan.required_text_units == _unit_len("innā") + _unit_len("aʿṭaynāka")
    assert plan.competitors == (3, 4)


def test_text_disambiguation_marks_internal_repeat_and_unresolved_text():
    repeated = _build_text_disambiguation(
        5,
        {5: ["fa-bi-ayyi", "ālāʾi", "rabbikumā"]},
        internal_sequences=[["fa-bi-ayyi", "ālāʾi", "qabla"]],
    )
    unresolved = _build_text_disambiguation(
        5,
        {5: ["qul"], 6: ["qul", "aʿūdhu"]},
        competitor_ayahs=[6],
    )

    assert repeated.reason == "internal_repeat"
    assert repeated.required_words == 3
    assert unresolved.reason == "unresolved"
    assert unresolved.distinguishing_word is None
    assert _text_disambiguation_requires_atomic(unresolved)
    assert not _text_disambiguation_requires_atomic(repeated)


def test_speech_duration_subtracts_and_merges_internal_pauses():
    assert _speech_ms_excluding_pauses(
        1000.0,
        5000.0,
        [
            (500.0, 1200.0),
            (2000.0, 2500.0),
            (2400.0, 2800.0),
            (4900.0, 5200.0),
        ],
    ) == 2900.0


def test_tempo_estimate_ignores_weak_and_extreme_observations():
    estimate = _estimate_tempo(
        [
            TempoObservation(10.0, 1000.0, pause_ms=400.0, confidence=0.9),
            TempoObservation(20.0, 2000.0, pause_ms=600.0, confidence=0.8),
            TempoObservation(12.0, 1200.0, pause_ms=500.0, confidence=0.95),
            TempoObservation(100.0, 1000.0, pause_ms=9000.0, confidence=0.9),
            TempoObservation(1.0, 5000.0, confidence=0.2),
        ],
        fallback_rate_units_per_ms=0.02,
    )

    assert isinstance(estimate, TempoEstimate)
    assert estimate.rate_units_per_ms == 0.01
    assert estimate.conservative_rate < estimate.rate_units_per_ms
    assert estimate.observation_count == 3
    assert estimate.pause_median_ms == 500.0
    assert not estimate.fallback_used


def test_tempo_estimate_blends_small_history_with_fallback():
    estimate = _estimate_tempo(
        [TempoObservation(10.0, 1000.0, confidence=1.0)],
        fallback_rate_units_per_ms=0.02,
        minimum_observations=3,
    )

    assert 0.01 < estimate.rate_units_per_ms < 0.02
    assert estimate.fallback_used
    assert estimate.confidence == 1 / 3


def test_pause_read_plan_keeps_marker_center_and_speech_after_long_pause():
    disambiguation = TextDisambiguation(
        required_text_units=20.0,
        required_words=2,
        competitors=(3,),
        reason="shared_opening",
        target_text="innāaʿṭaynāka",
        distinguishing_word="aʿṭaynāka",
    )
    tempo = TempoEstimate(
        rate_units_per_ms=0.01,
        conservative_rate=0.008,
        spread_units_per_ms=0.001,
        pause_median_ms=500.0,
        pause_iqr_ms=200.0,
        confidence=1.0,
        observation_count=4,
        fallback_used=False,
    )

    plan = _build_pause_read_plan(
        pause_start_ms=1000.0,
        pause_end_ms=3000.0,
        total_ms=20_000.0,
        disambiguation=disambiguation,
        tempo=tempo,
        minimum_window_ms=2500.0,
        maximum_window_ms=8500.0,
        safety_margin_ms=500.0,
    )

    assert isinstance(plan, PauseReadPlan)
    assert plan.marker_ms == 2000.0
    assert plan.decode_start_ms == 1000.0
    assert plan.decode_end_ms == 6000.0
    assert plan.speech_read_ms == 3000.0
    assert plan.reason == "shared_opening:local-tempo"

    long_pause_plan = _build_pause_read_plan(
        pause_start_ms=1000.0,
        pause_end_ms=12_000.0,
        total_ms=30_000.0,
        disambiguation=disambiguation,
        tempo=tempo,
        minimum_window_ms=2500.0,
        maximum_window_ms=8500.0,
        safety_margin_ms=500.0,
    )
    assert long_pause_plan.decode_end_ms == 15_000.0
    assert long_pause_plan.speech_read_ms == 3000.0


def test_span_ratio_quality_prefers_matching_duration():
    assert _span_ratio_quality(1000.0, 1000.0) == 1.0
    assert _span_ratio_quality(1000.0, 2500.0) == 0.0
    assert _span_ratio_quality(0.0, 1000.0) == 0.0


def test_weak_recovered_long_pause_start_requires_local_plausibility():
    kwargs = dict(
        pause_reason="dsp",
        pause_duration_ms=960.0,
        sim=0.37,
        action="recovered",
        expected_delta_ms=-2477.0,
        next_quality=0.95,
    )
    assert _weak_recovered_long_pause_start_candidate(**kwargs)
    assert not _weak_recovered_long_pause_start_candidate(
        **{**kwargs, "expected_delta_ms": -3001.0}
    )
    assert not _weak_recovered_long_pause_start_candidate(
        **{**kwargs, "expected_delta_ms": 3001.0}
    )
    assert not _weak_recovered_long_pause_start_candidate(
        **{**kwargs, "pause_duration_ms": 599.0}
    )
    assert not _weak_recovered_long_pause_start_candidate(
        **{**kwargs, "action": "observed_current"}
    )
    assert not _weak_recovered_long_pause_start_candidate(
        **{**kwargs, "next_quality": 0.49}
    )


def test_weak_recovered_compact_pause_center_requires_unclear_next_span():
    kwargs = dict(
        pause_reason="dsp",
        pause_duration_ms=840.0,
        sim=0.39,
        action="recovered",
        next_quality=0.435,
    )
    assert _weak_recovered_compact_pause_center_candidate(**kwargs)
    assert not _weak_recovered_compact_pause_center_candidate(
        **{**kwargs, "next_quality": 0.50}
    )
    assert not _weak_recovered_compact_pause_center_candidate(
        **{**kwargs, "pause_duration_ms": 900.0}
    )


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


def test_pause_center_start_is_independent_of_ctc_word_onset():
    assert _pause_center_start(253_700.0, 256_700.0) == 255_200.0


def test_ctc_rolls_back_a_weak_future_pause_to_the_preceding_dsp_pause():
    kwargs = dict(
        original_pause_reason="weak-dsp",
        original_pause_sim=0.289,
        repair_onset_ms=254_002.9,
        repair_conf=0.501,
        candidate_pause_end_ms=253_900.0,
        candidate_pause_reason="dsp",
        opening_shared_prefix_chars=0,
        conf_reliable=0.45,
        pause_trust_sim=0.42,
    )
    assert _ctc_prior_pause_replaces_weak_future_owner(**kwargs)
    assert not _ctc_prior_pause_replaces_weak_future_owner(
        **{**kwargs, "candidate_pause_end_ms": 253_500.0}
    )
    assert not _ctc_prior_pause_replaces_weak_future_owner(
        **{**kwargs, "original_pause_reason": "dsp"}
    )


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


def test_follows_unresolved_cascade_requires_all_predecessors():
    unresolved = {70, 71, 72, 73}
    assert _follows_unresolved_cascade(74, unresolved)
    assert not _follows_unresolved_cascade(75, unresolved)
    assert not _follows_unresolved_cascade(73, unresolved)
    assert not _follows_unresolved_cascade(74, {70, 71, 73})


def test_single_wasl_prior_pause_requires_unique_opening_and_ctc_evidence():
    candidates = [(734_280.0, 0.310, 0.186)]
    kwargs = {
        "previous_time": 727_812.0,
        "full_read_onset": 742_622.0,
        "candidates": candidates,
        "opening_shared_prefix_chars": 0,
        "full_read_confidence": 0.744,
        "reliable_confidence": 0.45,
    }
    assert _single_wasl_prior_pause_candidate(**kwargs) == 734_280.0
    assert _single_wasl_prior_pause_candidate(
        **{**kwargs, "opening_shared_prefix_chars": 1}
    ) is None
    assert _single_wasl_prior_pause_candidate(
        **{**kwargs, "full_read_confidence": 0.20}
    ) is None


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


def test_prior_internal_stop_replacement_requires_heard_opening_when_compact():
    assert _allow_prior_internal_stop_replacement(
        0.327,
        selected_pause_duration_ms=2140.0,
    )
    assert not _allow_prior_internal_stop_replacement(
        0.116,
        selected_pause_duration_ms=2140.0,
    )
    assert _allow_prior_internal_stop_replacement(
        0.116,
        selected_pause_duration_ms=7260.0,
    )


def test_post_pause_repair_confirms_prior_dsp_not_late_micro_owner():
    kwargs = {
        "original_pause_reason": "micro-tempo",
        "original_pause_sim": 0.339,
        "repair_onset_ms": 689_072.8,
        "repair_conf": 0.579,
        "candidate_pause_end_ms": 688_860.0,
        "candidate_pause_reason": "dsp",
        "candidate_prefix": 4,
        "selected_prefix": 0,
        "conf_reliable": 0.45,
        "pause_trust_sim": 0.50,
    }

    assert _post_pause_repair_confirms_prior_dsp(**kwargs)
    assert not _post_pause_repair_confirms_prior_dsp(
        **{**kwargs, "repair_onset_ms": 689_260.0}
    )
    assert not _post_pause_repair_confirms_prior_dsp(
        **{**kwargs, "selected_prefix": 3}
    )


def test_confirmed_trusted_pause_refine_replaces_weak_owner():
    kwargs = {
        "from_pause": True,
        "original_pause_sim": 0.367,
        "refined_conf": 0.706,
        "refined_inside_different_trusted_pause": True,
        "conf_reliable": 0.45,
        "pause_trust_sim": 0.50,
    }

    assert _confirmed_trusted_pause_refine_candidate(**kwargs)
    assert not _confirmed_trusted_pause_refine_candidate(
        **{**kwargs, "refined_conf": 0.40}
    )
    assert not _confirmed_trusted_pause_refine_candidate(
        **{**kwargs, "refined_inside_different_trusted_pause": False}
    )


def test_anchored_repair_can_replace_a_geometry_invalidated_owner():
    kwargs = {
        "from_pause": True,
        "original_pause_sim": 0.384,
        "geometry_invalidated_owner": True,
        "repair_conf": 0.831,
        "repaired_inside_different_trusted_pause": True,
        "conf_reliable": 0.45,
        "pause_trust_sim": 0.50,
    }
    assert _confirmed_trusted_pause_repair_candidate(**kwargs)
    assert not _confirmed_trusted_pause_repair_candidate(
        **{**kwargs, "geometry_invalidated_owner": False}
    )
    assert not _confirmed_trusted_pause_repair_candidate(
        **{**kwargs, "repaired_inside_different_trusted_pause": False}
    )
    assert not _confirmed_trusted_pause_repair_candidate(
        **{**kwargs, "repair_conf": 0.449}
    )


def test_micro_owner_prior_prefix_candidate_requires_text_and_pair_geometry():
    kwargs = {
        "previous_time": 1_376_702.5,
        "selected_time": 1_387_282.8,
        "next_time": 1_406_470.0,
        "expected_prev_gap_ms": 7203.6,
        "expected_next_gap_ms": 19_433.0,
        "candidates": [(1_383_660.0, 0.237, 2)],
        "pause_reason": "micro-tempo",
        "action": "weak_pause_recovery",
        "selected_sim": 0.325,
        "selected_prefix": 0,
    }

    assert _micro_owner_prior_prefix_candidate(**kwargs) == 1_383_660.0
    assert _micro_owner_prior_prefix_candidate(
        **{**kwargs, "candidates": [(1_383_660.0, 0.237, 1)]}
    ) is None
    assert _micro_owner_prior_prefix_candidate(
        **{**kwargs, "selected_prefix": 2}
    ) is None


def test_repeated_wasl_prior_pause_candidate_requires_text_and_spacing():
    kwargs = {
        "previous_time": 1_232_617.5,
        "full_read_onset": 1_258_970.6,
        "expected_gap_ms": 18_047.8,
        "candidates": [(1_251_110.0, 0.413, 0.338)],
        "opening_shared_prefix_chars": 19,
        "full_read_confidence": 0.672,
        "reliable_confidence": 0.45,
    }

    assert _repeated_wasl_prior_pause_candidate(**kwargs) == 1_251_110.0
    assert _repeated_wasl_prior_pause_candidate(
        **{**kwargs, "candidates": [(1_251_110.0, 0.36, 0.20)]}
    ) is None
    assert _repeated_wasl_prior_pause_candidate(
        **{**kwargs, "opening_shared_prefix_chars": 3}
    ) is None


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


def test_confirmed_weak_pause_repair_requires_strong_local_evidence():
    kwargs = {
        "from_pause": True,
        "original_pause_sim": 0.325,
        "repair_conf": 0.688,
        "conf_reliable": 0.45,
        "repaired_inside_weak_pause": True,
        "pause_trust_sim": 0.50,
    }
    assert _confirmed_weak_pause_repair_candidate(**kwargs)
    assert not _confirmed_weak_pause_repair_candidate(
        **{**kwargs, "repaired_inside_weak_pause": False}
    )
    assert not _confirmed_weak_pause_repair_candidate(
        **{**kwargs, "repair_conf": 0.44}
    )
    assert not _confirmed_weak_pause_repair_candidate(
        **{**kwargs, "original_pause_sim": 0.56}
    )
    assert _confirmed_weak_pause_repair_candidate(
        **{**kwargs, "from_pause": False, "original_pause_sim": 0.0}
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
        == 5352.0
    )
    assert (
        _first_ayah_content_start(
            5600.0,
            basmala=True,
            intro_pause_start_ms=4568.0,
            intro_pause_end_ms=6136.0,
            raw_confidence=0.60,
            reliable_confidence=0.45,
            min_content_start_ms=6000.0,
        )
        == 5352.0
    )
    assert (
        _first_ayah_content_start(
            5200.0,
            basmala=True,
            intro_pause_start_ms=4568.0,
            intro_pause_end_ms=6136.0,
            raw_confidence=0.60,
            reliable_confidence=0.45,
            min_content_start_ms=6000.0,
        )
        == 5352.0
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
    assert not _accept_fallback_pause_candidate(
        pause_reason="micro-tempo",
        sim=0.26,
        min_match_sim=0.31,
        expected_delta_ms=1799.7,
        prev_quality=0.878,
        next_quality=0.0,
        next_speech_ms=1172.0,
    )
    assert not _accept_fallback_pause_candidate(
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
    assert not _accept_fallback_pause_candidate(
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


def test_long_pause_context_read_requires_contradictory_windows():
    candidates = [
        (156_695.5, 0.0, False),
        (156_500.3, 0.159, False),
        (155_463.0, 0.433, True),
        (155_454.4, 0.391, False),
    ]
    assert _long_pause_context_read_candidate(
        candidates,
        pause_reason="dsp",
        pause_start_ms=155_780.0,
        pause_end_ms=157_360.0,
        refined_onset_ms=155_454.4,
        refined_conf=0.391,
        conf_thresh=0.25,
    )
    assert not _long_pause_context_read_candidate(
        [(155_454.4, 0.391, False)],
        pause_reason="dsp",
        pause_start_ms=155_780.0,
        pause_end_ms=157_360.0,
        refined_onset_ms=155_454.4,
        refined_conf=0.391,
        conf_thresh=0.25,
    )
    assert not _long_pause_context_read_candidate(
        candidates,
        pause_reason="dsp",
        pause_start_ms=155_780.0,
        pause_end_ms=157_360.0,
        refined_onset_ms=155_454.4,
        refined_conf=0.20,
        conf_thresh=0.25,
    )
    assert not _long_pause_context_read_candidate(
        [
            (751_143.0, 0.60, False),
            (748_600.0, 0.50, False),
        ],
        pause_reason="dsp",
        pause_start_ms=748_560.0,
        pause_end_ms=751_180.0,
        refined_onset_ms=748_300.0,
        refined_conf=0.50,
        conf_thresh=0.25,
    )


def test_pause_backed_start_snaps_only_near_its_owner_interval():
    assert (
        _snap_pause_backed_start_to_owner(
            313_905.5,
            pause_start_ms=313_979.5,
            pause_end_ms=314_201.5,
        )
        == 313_979.5
    )
    assert (
        _snap_pause_backed_start_to_owner(
            314_100.0,
            pause_start_ms=313_979.5,
            pause_end_ms=314_201.5,
        )
        == 314_100.0
    )
    assert (
        _snap_pause_backed_start_to_owner(
            369_040.0,
            pause_start_ms=372_040.0,
            pause_end_ms=373_900.0,
        )
        is None
    )


def test_predecessor_repair_restores_strong_owner_after_provisional_push():
    params = {
        "current_start_ms": 822_190.0,
        "owner_timing_ms": 812_780.0,
        "pause_start_ms": 812_580.0,
        "pause_end_ms": 812_980.0,
        "repaired_predecessor_ms": 794_030.0,
        "from_pause": True,
        "pause_similarity": 0.635,
        "pause_trust_similarity": 0.50,
    }
    assert _restore_trusted_owner_after_predecessor_repair(**params) == 812_780.0
    assert _restore_trusted_owner_after_predecessor_repair(
        **{**params, "pause_similarity": 0.49},
    ) is None
    assert _restore_trusted_owner_after_predecessor_repair(
        **{**params, "repaired_predecessor_ms": 812_770.0},
    ) is None


def test_internal_repair_restores_owner_after_stale_monotonic_push():
    params = {
        "current_start_ms": 445_950.0,
        "owner_timing_ms": 438_220.0,
        "pause_start_ms": 437_220.0,
        "pause_end_ms": 439_220.0,
        "repaired_onset_ms": 439_180.0,
        "repaired_confidence": 0.560,
        "repaired_predecessor_ms": 423_790.0,
        "from_pause": True,
        "acceptance_confidence": 0.55,
    }
    assert _restore_owner_timing_from_internal_repair(**params) == 438_220.0
    assert _restore_owner_timing_from_internal_repair(
        **{**params, "repaired_confidence": 0.549},
    ) is None
    assert _restore_owner_timing_from_internal_repair(
        **{**params, "repaired_onset_ms": 439_221.0},
    ) is None
    assert _restore_owner_timing_from_internal_repair(
        **{**params, "current_start_ms": 438_220.0},
    ) is None
    assert _restore_owner_timing_from_internal_repair(
        **{**params, "repaired_predecessor_ms": 438_200.0},
    ) is None


def test_context_block_onset_must_be_confident_and_inside_owned_pause():
    assert (
        _context_block_pause_onset_candidate(
            157_222.0,
            0.573,
            pause_start_ms=155_780.0,
            pause_end_ms=157_360.0,
            left_anchor_ms=140_208.0,
            right_anchor_ms=190_350.0,
            min_confidence=0.45,
        )
        == 157_222.0
    )
    assert (
        _context_block_pause_onset_candidate(
            157_500.0,
            0.80,
            pause_start_ms=155_780.0,
            pause_end_ms=157_360.0,
            left_anchor_ms=140_208.0,
            right_anchor_ms=190_350.0,
            min_confidence=0.45,
        )
        is None
    )
    assert (
        _context_block_pause_onset_candidate(
            157_222.0,
            0.30,
            pause_start_ms=155_780.0,
            pause_end_ms=157_360.0,
            left_anchor_ms=140_208.0,
            right_anchor_ms=190_350.0,
            min_confidence=0.45,
        )
        is None
    )


def test_very_long_pause_opposite_edge_retry_requires_context_block():
    params = {
        "pause_reason": "dsp",
        "pause_start_ms": 777_140.0,
        "pause_end_ms": 784_780.0,
        "selected_onset_ms": 780_960.0,
        "repaired_onset_ms": 784_983.5,
        "repaired_confidence": 0.340,
        "acceptance_confidence": 0.25,
    }
    assert _very_long_pause_repair_context_candidate(**params)
    assert not _very_long_pause_repair_context_candidate(
        **{**params, "repaired_onset_ms": 784_700.0},
    )
    assert not _very_long_pause_repair_context_candidate(
        **{**params, "repaired_confidence": 0.24},
    )
    assert not _very_long_pause_repair_context_candidate(
        **{**params, "selected_onset_ms": 783_500.0},
    )
    assert _very_long_context_owner_still_valid(
        pause_start_ms=777_140.0,
        pause_end_ms=784_780.0,
    )
    assert not _very_long_context_owner_still_valid(
        pause_start_ms=885_240.0,
        pause_end_ms=886_380.0,
    )


def test_full_span_onset_uses_the_actual_fallback_right_anchor():
    assert _accept_anchored_full_span_onset(
        8450.0,
        0.10,
        left_anchor_ms=1000.0,
        right_anchor_ms=9000.0,
    )
    assert not _accept_anchored_full_span_onset(
        8990.0,
        0.80,
        left_anchor_ms=1000.0,
        right_anchor_ms=9000.0,
    )


def test_full_span_pause_must_agree_with_the_independent_local_read():
    assert _full_span_pause_agrees_with_local_read(176_596.9, 178_556.7)
    assert not _full_span_pause_agrees_with_local_read(546_316.0, 558_315.4)
    assert _full_span_pause_owner_is_consistent(
        2_877_063.1,
        2_883_072.3,
        pause_duration_ms=680.0,
    )
    assert not _full_span_pause_owner_is_consistent(
        546_316.0,
        558_315.4,
        pause_duration_ms=6760.0,
    )


def test_full_span_unique_next_pause_requires_no_intervening_trusted_pause():
    assert _full_span_unique_next_pause_candidate(
        local_onset_ms=2_877_063.1,
        full_pause_start_ms=2_880_740.0,
        target_occurrences=1,
        full_confidence=0.899,
        confidence_threshold=0.45,
        intervening_trusted_pauses=0,
    )
    assert not _full_span_unique_next_pause_candidate(
        local_onset_ms=497_109.2,
        full_pause_start_ms=502_720.0,
        target_occurrences=1,
        full_confidence=0.872,
        confidence_threshold=0.45,
        intervening_trusted_pauses=1,
    )
    assert not _full_span_unique_next_pause_candidate(
        local_onset_ms=546_316.0,
        full_pause_start_ms=551_780.0,
        target_occurrences=2,
        full_confidence=0.663,
        confidence_threshold=0.45,
        intervening_trusted_pauses=0,
    )


def test_impossible_owner_geometry_requires_a_full_span_read():
    assert _pause_owner_geometry_needs_full_span(
        from_pause=True,
        old_onset_ms=3_015_525.0,
        local_onset_ms=3_068_223.9,
        previous_quality=0.0,
        next_quality=0.0,
    )
    assert not _pause_owner_geometry_needs_full_span(
        from_pause=True,
        old_onset_ms=156_570.0,
        local_onset_ms=157_222.0,
        previous_quality=0.90,
        next_quality=0.75,
    )


def test_geometry_invalidated_owner_cannot_be_revalidated_by_local_read():
    assert _geometry_invalidated_owner_requires_full_span(
        geometry_unreliable=True,
        action="weak_pause_recovery",
    )
    assert _geometry_invalidated_owner_requires_full_span(
        geometry_unreliable=True,
        action="observed_current",
    )
    assert _geometry_invalidated_owner_requires_full_span(
        geometry_unreliable=True,
        action="post_selected_micro_recovery",
        pause_similarity=0.186,
        min_match_similarity=0.25,
    )
    assert not _geometry_invalidated_owner_requires_full_span(
        geometry_unreliable=True,
        action="post_selected_micro_recovery",
        pause_similarity=0.335,
        min_match_similarity=0.25,
    )
    assert not _geometry_invalidated_owner_requires_full_span(
        geometry_unreliable=False,
        action="weak_pause_recovery",
    )
    assert not _geometry_repair_is_already_consistent(
        current_ms=6_445_610.8,
        predicted_ms=6_445_941.3,
        geometry_invalidated_owner=True,
    )


def test_low_similarity_unclassified_pause_requires_full_context():
    params = {
        "from_pause": True,
        "action": "",
        "pause_similarity": 0.24,
        "min_match_similarity": 0.31,
    }
    assert _unclassified_owner_requires_full_span(**params)
    assert not _unclassified_owner_requires_full_span(
        **{**params, "action": "observed_current"},
    )
    assert not _unclassified_owner_requires_full_span(
        **{**params, "pause_similarity": 0.0},
    )
    assert not _unclassified_owner_requires_full_span(
        **{**params, "pause_similarity": 0.40},
    )
    assert _geometry_repair_is_already_consistent(
        current_ms=6_445_610.8,
        predicted_ms=6_445_941.3,
        geometry_invalidated_owner=False,
    )
    assert not _geometry_repair_is_already_consistent(
        current_ms=6_440_316.0,
        predicted_ms=6_445_941.3,
        geometry_invalidated_owner=False,
    )


def test_unsupported_recovered_owner_defers_to_distant_anchor_geometry():
    params = {
        "from_pause": True,
        "action": "recovered",
        "pause_similarity": 0.254,
        "previous_quality": 0.0,
        "next_quality": 0.0,
        "current_ms": 2_960_035.0,
        "predicted_ms": 2_971_775.0,
        "pause_trust_similarity": 0.50,
    }
    assert _recovered_owner_conflicts_with_anchor_geometry(**params)
    assert not _recovered_owner_conflicts_with_anchor_geometry(
        **{**params, "action": "observed_current"},
    )
    assert not _recovered_owner_conflicts_with_anchor_geometry(
        **{**params, "next_quality": 0.20},
    )
    assert not _recovered_owner_conflicts_with_anchor_geometry(
        **{**params, "predicted_ms": 2_969_000.0},
    )


def test_unsupported_first_opening_owner_defers_to_anchor_geometry():
    params = {
        "from_pause": True,
        "action": "first_opening_repair",
        "pause_similarity": 0.289,
        "previous_quality": 0.0,
        "next_quality": 0.0,
        "current_ms": 669_600.0,
        "predicted_ms": 730_128.1,
        "pause_trust_similarity": 0.50,
    }
    assert _first_opening_owner_conflicts_with_anchor_geometry(**params)
    assert not _first_opening_owner_conflicts_with_anchor_geometry(
        **{**params, "previous_quality": 0.959},
    )
    assert not _first_opening_owner_conflicts_with_anchor_geometry(
        **{**params, "predicted_ms": 685_000.0},
    )
    assert _first_opening_owner_conflicts_with_anchor_geometry(
        **{
            **params,
            "current_ms": 3_156_145.0,
            "predicted_ms": 3_180_203.5,
        },
    )


def test_unsupported_direct_owner_defers_to_leave_one_out_anchor_geometry():
    params = {
        "from_pause": True,
        "action": "observed_current",
        "pause_similarity": 0.463,
        "previous_quality": 0.0,
        "next_quality": 0.043,
        "current_ms": 576_160.0,
        "predicted_ms": 568_604.0,
        "pause_trust_similarity": 0.50,
    }
    assert _direct_owner_conflicts_with_anchor_geometry(**params)
    assert not _direct_owner_conflicts_with_anchor_geometry(
        **{**params, "next_quality": 0.20},
    )
    assert not _direct_owner_conflicts_with_anchor_geometry(
        **{**params, "predicted_ms": 572_000.0},
    )
    assert not _direct_owner_conflicts_with_anchor_geometry(
        **{**params, "action": "first_opening_repair"},
    )


def test_recovered_later_best_pause_needs_sim_and_prefix_gains():
    params = {
        "selected_time": 1_997_890.0,
        "next_time": 2_015_670.0,
        "candidates": [(2_004_720.0, 0.389, 5)],
        "pause_reason": "dsp",
        "selected_sim": 0.340,
        "selected_prefix": 3,
        "action": "recovered",
    }
    assert _recovered_later_best_pause_candidate(**params) == 2_004_720.0
    assert _recovered_later_best_pause_candidate(
        **{**params, "candidates": [(2_004_720.0, 0.37, 5)]},
    ) is None
    assert _recovered_later_best_pause_candidate(
        **{**params, "candidates": [(2_004_720.0, 0.389, 4)]},
    ) is None
    assert _recovered_later_best_pause_candidate(
        **{**params, "action": "observed_current"},
    ) is None
    assert _recovered_later_best_pause_candidate(
        **{**params, "candidates": [(2_008_000.0, 0.50, 8)]},
    ) is None


def test_competing_prior_pause_requires_full_span_read():
    params = {
        "from_pause": True,
        "action": "observed_current",
        "pause_reason": "dsp",
        "selected_similarity": 0.489,
        "selected_time_ms": 367_420.0,
        "previous_anchor_ms": 335_430.0,
        "candidates": [(356_230.0, "weak-dsp", 0.413)],
        "pause_trust_similarity": 0.50,
    }
    assert _competing_prior_pause_requires_full_span(**params)
    assert not _competing_prior_pause_requires_full_span(
        **{**params, "candidates": [(356_230.0, "weak-dsp", 0.38)]},
    )
    assert not _competing_prior_pause_requires_full_span(
        **{**params, "candidates": [(350_000.0, "weak-dsp", 0.45)]},
    )
    assert not _competing_prior_pause_requires_full_span(
        **{**params, "selected_similarity": 0.51},
    )
    assert not _competing_prior_pause_requires_full_span(
        **{**params, "action": "recovered"},
    )


def test_extended_decode_must_clearly_prefer_competing_prior_pause():
    params = {
        "candidate_reason": "weak-dsp",
        "candidate_similarity": 0.488,
        "selected_similarity": 0.463,
    }
    assert _prefer_competing_prior_pause_extended_decode(**params)
    assert not _prefer_competing_prior_pause_extended_decode(
        **{**params, "candidate_similarity": 0.47},
    )
    assert not _prefer_competing_prior_pause_extended_decode(
        **{**params, "selected_similarity": 0.475},
    )
    assert not _prefer_competing_prior_pause_extended_decode(
        **{**params, "candidate_reason": "rejected"},
    )


def test_pre_context_must_show_selected_pause_is_inside_current_ayah():
    params = {
        "candidate_previous_tail_similarity": 0.634,
        "selected_previous_tail_similarity": 0.216,
        "candidate_current_opening_similarity": 0.190,
        "selected_current_opening_similarity": 0.281,
    }
    assert _prefer_competing_prior_pause_pre_context(**params)
    assert not _prefer_competing_prior_pause_pre_context(
        **{**params, "candidate_previous_tail_similarity": 0.54},
    )
    assert not _prefer_competing_prior_pause_pre_context(
        **{**params, "selected_previous_tail_similarity": 0.40},
    )
    assert not _prefer_competing_prior_pause_pre_context(
        **{**params, "selected_current_opening_similarity": 0.23},
    )


def test_following_word_can_confirm_short_full_span_opening():
    params = {
        "opening_confidence": 0.0,
        "following_confidence": 0.470,
        "opening_onset_ms": 2_975_970.0,
        "following_onset_ms": 2_976_210.0,
        "same_ayah": True,
    }
    assert _full_span_opening_support_confidence(**params) == 0.470
    assert _full_span_opening_support_confidence(
        **{**params, "same_ayah": False},
    ) == 0.0
    assert _full_span_opening_support_confidence(
        **{**params, "following_onset_ms": 2_977_210.0},
    ) == 0.0
    assert _full_span_opening_support_confidence(
        **{**params, "opening_confidence": 0.80},
    ) == 0.80


def test_geometry_invalidated_full_span_can_use_following_word_without_pause():
    assert _full_span_following_support_is_eligible(
        geometry_invalidated_owner=True,
        multiscale_full_span_required=False,
        impossible_owner_geometry=False,
    )
    assert not _full_span_following_support_is_eligible(
        geometry_invalidated_owner=False,
        multiscale_full_span_required=False,
        impossible_owner_geometry=False,
    )


def test_weak_full_wasl_read_is_insufficient_and_rejects_weak_closer():
    assert _full_wasl_read_classification(
        0.168,
        acceptance_confidence=0.25,
        audio_bracket_proven=False,
    ) == "insufficient_audio"
    assert _full_wasl_read_classification(
        0.25,
        acceptance_confidence=0.25,
        audio_bracket_proven=False,
    ) == "wasl"
    assert _full_wasl_read_classification(
        0.168,
        acceptance_confidence=0.25,
        audio_bracket_proven=True,
    ) == "wasl"
    params = {
        "classification": "insufficient_audio",
        "bracketed_on_right": True,
        "right_anchor_from_pause": True,
        "right_anchor_similarity": 0.351,
        "pause_trust_similarity": 0.50,
        "opening_shared_prefix_chars": 8,
    }
    assert _insufficient_wasl_read_invalidates_right_anchor(**params)
    assert not _insufficient_wasl_read_invalidates_right_anchor(
        **{**params, "right_anchor_similarity": 0.50},
    )
    assert not _insufficient_wasl_read_invalidates_right_anchor(
        **{**params, "classification": "wasl"},
    )
    assert not _insufficient_wasl_read_invalidates_right_anchor(
        **{**params, "opening_shared_prefix_chars": 3},
    )


def test_recording_end_brackets_only_a_trailing_full_span_read():
    assert _full_span_right_bracket_ms(
        right_anchor_ms=None,
        total_ms=1_053_700.0,
        final_ayah=True,
    ) == 1_053_700.0
    assert _full_span_right_bracket_ms(
        right_anchor_ms=1_025_860.0,
        total_ms=1_053_700.0,
        final_ayah=False,
    ) == 1_025_860.0
    assert _full_span_right_bracket_ms(
        right_anchor_ms=None,
        total_ms=1_053_700.0,
        final_ayah=False,
    ) is None


def test_repeated_weak_wasl_run_requires_reliable_bracket_extension():
    params = {
        "run_length": 2,
        "opening_shared_prefix_chars": 4,
        "minimum_confidence": 0.216,
        "acceptance_confidence": 0.25,
        "right_anchor_from_pause": True,
        "right_anchor_similarity": 0.314,
        "pause_trust_similarity": 0.50,
    }
    assert _ambiguous_wasl_run_requires_bracket_extension(**params)
    assert not _ambiguous_wasl_run_requires_bracket_extension(
        **{**params, "run_length": 1},
    )
    assert not _ambiguous_wasl_run_requires_bracket_extension(
        **{**params, "minimum_confidence": 0.25},
    )
    assert not _ambiguous_wasl_run_requires_bracket_extension(
        **{**params, "right_anchor_similarity": 0.50},
    )


def test_unclassified_repeated_pause_run_requires_atomic_block():
    params = {
        "from_pause": True,
        "action": "",
        "pause_similarity": 0.333,
        "previous_quality": 0.038,
        "next_quality": 0.0,
        "opening_shared_prefix_chars": 6,
        "previous_from_pause": True,
        "previous_similarity": 0.371,
        "next_from_pause": True,
        "next_similarity": 0.381,
        "pause_trust_similarity": 0.50,
    }
    assert _unclassified_repeated_pause_run_requires_atomic_block(**params)
    assert not _unclassified_repeated_pause_run_requires_atomic_block(
        **{**params, "action": "observed_current"},
    )
    assert not _unclassified_repeated_pause_run_requires_atomic_block(
        **{**params, "opening_shared_prefix_chars": 3},
    )
    assert not _unclassified_repeated_pause_run_requires_atomic_block(
        **{**params, "next_similarity": 0.50},
    )
    assert _repeated_low_trust_left_anchor_extends_atomic_block(
        pause_similarity=0.371,
        previous_quality=0.0,
        opening_shared_prefix_chars=3,
        pause_trust_similarity=0.50,
    )
    assert not _repeated_low_trust_left_anchor_extends_atomic_block(
        pause_similarity=0.371,
        previous_quality=0.0,
        opening_shared_prefix_chars=2,
        pause_trust_similarity=0.50,
    )
    assert not _repeated_low_trust_left_anchor_extends_atomic_block(
        pause_similarity=0.50,
        previous_quality=0.0,
        opening_shared_prefix_chars=3,
        pause_trust_similarity=0.50,
    )


def test_reliable_full_read_can_confirm_exact_weak_pause_edge():
    params = {
        "onset_ms": 1_951_020.0,
        "pause_start_ms": 1_950_780.0,
        "pause_end_ms": 1_951_000.0,
        "pause_tier_name": "weak",
        "pause_reason": "weak-dsp",
        "confidence": 0.577,
        "reliable_confidence": 0.45,
    }
    assert _full_span_pause_candidate_is_supported(**params)
    assert not _full_span_pause_candidate_is_supported(
        **{**params, "onset_ms": 1_951_100.0},
    )
    assert not _full_span_pause_candidate_is_supported(
        **{**params, "confidence": 0.44},
    )
    assert _full_span_pause_candidate_is_supported(
        **{
            **params,
            "onset_ms": 1_950_900.0,
            "pause_tier_name": "trusted",
            "pause_reason": "dsp",
            "confidence": 0.0,
        },
    )


def test_local_and_full_reads_can_jointly_confirm_a_weak_pause_edge():
    params = {
        "full_onset_ms": 3_385_012.7,
        "full_confidence": 0.169,
        "local_onset_ms": 3_384_288.2,
        "local_confidence": 0.375,
        "pause_start_ms": 3_384_540.0,
        "pause_end_ms": 3_385_000.0,
        "pause_reason": "weak-dsp",
        "acceptance_confidence": 0.25,
    }
    assert _full_span_weak_pause_is_jointly_supported(**params)
    assert not _full_span_weak_pause_is_jointly_supported(
        **{**params, "full_confidence": 0.09},
    )
    assert not _full_span_weak_pause_is_jointly_supported(
        **{**params, "local_confidence": 0.24},
    )
    assert not _full_span_weak_pause_is_jointly_supported(
        **{**params, "local_onset_ms": 3_384_100.0},
    )
    assert not _full_span_weak_pause_is_jointly_supported(
        **{**params, "pause_reason": "dsp"},
    )


def test_unique_anchored_full_read_can_confirm_invalidated_weak_pause():
    params = {
        "geometry_invalidated_owner": True,
        "full_onset_ms": 45_741.9,
        "full_confidence": 0.337,
        "target_occurrences": 1,
        "pause_start_ms": 45_620.0,
        "pause_end_ms": 45_880.0,
        "pause_reason": "weak-dsp",
        "acceptance_confidence": 0.25,
    }
    assert _geometry_invalidated_full_span_weak_pause_is_supported(**params)
    assert not _geometry_invalidated_full_span_weak_pause_is_supported(
        **{**params, "target_occurrences": 2}
    )
    assert not _geometry_invalidated_full_span_weak_pause_is_supported(
        **{**params, "geometry_invalidated_owner": False}
    )
    assert not _geometry_invalidated_full_span_weak_pause_is_supported(
        **{**params, "full_onset_ms": 45_881.0}
    )


def test_multiscale_consensus_requires_direct_non_repeated_pause_claim():
    params = {
        "action": "observed_current",
        "pause_similarity": 0.309,
        "pause_trust_similarity": 0.50,
        "opening_shared_prefix_chars": 2,
        "owner_prediction_shift_ms": 7788.5,
    }
    assert _multiscale_pause_repair_is_eligible(**params)
    assert _multiscale_pause_repair_is_eligible(
        **{**params, "opening_shared_prefix_chars": 5},
    )
    assert not _multiscale_pause_repair_is_eligible(
        **{**params, "action": "first_opening_repair"},
    )
    assert not _multiscale_pause_repair_is_eligible(
        **{**params, "opening_shared_prefix_chars": 6},
    )
    assert not _multiscale_pause_repair_is_eligible(
        **{**params, "owner_prediction_shift_ms": 2999.0},
    )


def test_eligible_multiscale_reads_must_cluster_inside_one_pause():
    candidates = [
        (3_377_399.6, 0.027, False),
        (3_377_388.8, 0.103, False),
        (3_377_398.6, 0.153, False),
        (3_377_408.4, 0.055, False),
        (3_377_398.2, 0.098, False),
    ]
    assert _stable_multiscale_pause_onset(
        candidates,
        pause_start_ms=3_377_240.0,
        pause_end_ms=3_377_560.0,
    ) == 3_377_398.6
    assert _stable_multiscale_pause_onset(
        [*candidates[:2], (3_377_520.0, 0.20, False)],
        pause_start_ms=3_377_240.0,
        pause_end_ms=3_377_560.0,
    ) is None


def test_multiscale_ctc_in_owner_requires_unique_text_and_two_sided_tempo():
    params = {
        "from_pause": True,
        "action": "observed_current",
        "pause_reason": "dsp",
        "pause_similarity": 0.357,
        "pause_trust_similarity": 0.50,
        "opening_shared_prefix_chars": 0,
        "prev_quality": 0.995,
        "next_quality": 0.956,
        "clustered_onset_ms": 510_428.0,
        "owner_onset_ms": 509_880.0,
        "pause_start_ms": 509_880.0,
        "pause_end_ms": 510_620.0,
        "previous_start_ms": 503_320.0,
        "next_start_ms": 516_860.0,
    }
    assert _multiscale_ctc_in_owner_candidate(**params)
    assert not _multiscale_ctc_in_owner_candidate(
        **{**params, "opening_shared_prefix_chars": 6}
    )
    assert not _multiscale_ctc_in_owner_candidate(
        **{**params, "next_quality": 0.79}
    )
    assert not _multiscale_ctc_in_owner_candidate(
        **{**params, "clustered_onset_ms": 510_615.0}
    )


def test_multiscale_ctc_consensus_is_not_replaced_by_owner_edge_snap():
    assert not _owner_interval_snap_is_allowed("multiscale_ctc_in_owner")
    assert _owner_interval_snap_is_allowed("post_micro_context_ctc")
    assert _owner_interval_snap_is_allowed(None)


def test_late_section_read_keeps_same_owner_multiscale_ctc_onset():
    params = {
        "timing_strategy": "multiscale_ctc_in_owner",
        "current_owner_center_ms": 509_880.0,
        "proposed_owner_center_ms": 509_880.0,
        "current_start_ms": 510_428.0,
        "pause_start_ms": 509_880.0,
        "pause_end_ms": 510_620.0,
    }
    assert _preserve_multiscale_ctc_owner_timing(**params)
    assert not _preserve_multiscale_ctc_owner_timing(
        **{**params, "proposed_owner_center_ms": 509_900.0}
    )
    assert not _preserve_multiscale_ctc_owner_timing(
        **{**params, "current_start_ms": 510_700.0}
    )


def test_repeated_opening_multiscale_conflict_requires_full_context():
    candidates = [
        (1_950_656.0, 0.004, False),
        (1_950_665.0, 0.005, False),
        (1_950_654.0, 0.110, False),
        (1_950_302.0, 0.119, False),
        (1_951_195.0, 0.150, False),
    ]
    cluster = _stable_multiscale_onset_cluster(candidates)
    assert cluster == 1_950_656.0
    params = {
        "action": "observed_current",
        "pause_similarity": 0.370,
        "pause_trust_similarity": 0.50,
        "opening_shared_prefix_chars": 9,
        "owner_onset_ms": 1_959_078.0,
        "predicted_onset_ms": 1_950_423.0,
        "clustered_onset_ms": cluster,
    }
    assert _ambiguous_multiscale_owner_requires_full_span(**params)
    assert not _ambiguous_multiscale_owner_requires_full_span(
        **{**params, "opening_shared_prefix_chars": 5},
    )
    assert not _ambiguous_multiscale_owner_requires_full_span(
        **{**params, "clustered_onset_ms": 1_952_000.0},
    )
    assert not _ambiguous_multiscale_owner_requires_full_span(
        **{**params, "action": "first_opening_repair"},
    )


def test_multiscale_cluster_rejects_edge_and_spread_out_candidates():
    assert _stable_multiscale_onset_cluster([
        (1000.0, 0.30, True),
        (1010.0, 0.20, False),
        (1020.0, 0.20, False),
    ]) is None


def test_large_wasl_revision_invalidates_low_trust_fallback_right_anchor():
    params = {
        "full_wasl_repair": True,
        "full_onset_ms": 1_373_435.2,
        "local_onset_ms": 1_384_616.0,
        "right_anchor_from_pause": True,
        "right_anchor_action": "pause_fallback",
        "right_anchor_similarity": 0.317,
        "pause_trust_similarity": 0.50,
    }
    assert _wasl_revision_invalidates_fallback_right_anchor(**params)
    assert not _wasl_revision_invalidates_fallback_right_anchor(
        **{**params, "right_anchor_action": "observed_current"},
    )
    assert not _wasl_revision_invalidates_fallback_right_anchor(
        **{**params, "right_anchor_similarity": 0.55},
    )
    assert not _wasl_revision_invalidates_fallback_right_anchor(
        **{**params, "local_onset_ms": 1_375_000.0},
    )
    assert _stable_multiscale_onset_cluster([
        (1000.0, 0.20, False),
        (1100.0, 0.20, False),
        (1200.0, 0.20, False),
    ]) is None


def test_forward_right_anchor_revision_rereads_weak_predecessor_wasl():
    params = {
        "old_anchor_ms": 1_286_597.5,
        "new_anchor_ms": 1_325_599.0,
        "run_confidence": 0.15,
        "acceptance_confidence": 0.25,
    }
    assert _revised_wasl_right_anchor_requires_predecessor_reread(**params)
    assert not _revised_wasl_right_anchor_requires_predecessor_reread(
        **{**params, "run_confidence": 0.25},
    )
    assert not _revised_wasl_right_anchor_requires_predecessor_reread(
        **{**params, "new_anchor_ms": 1_288_000.0},
    )


def test_invalidated_owner_cannot_block_reliable_repair_by_stale_confidence():
    assert _geometry_invalidated_repair_can_compete(
        geometry_invalidated_owner=True,
        repair_confidence=0.831,
        reliable_confidence=0.45,
    )
    assert not _geometry_invalidated_repair_can_compete(
        geometry_invalidated_owner=False,
        repair_confidence=0.831,
        reliable_confidence=0.45,
    )
    assert not _geometry_invalidated_repair_can_compete(
        geometry_invalidated_owner=True,
        repair_confidence=0.449,
        reliable_confidence=0.45,
    )


def test_low_similarity_owner_with_impossible_spans_is_not_an_anchor():
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="recovered",
        pause_reason="micro-tempo",
        pause_similarity=0.423,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-16_229.7,
        previous_expected_delta_ms=-15_000.0,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_reason="weak-dsp",
        pause_similarity=0.311,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-38_130.8,
        previous_expected_delta_ms=-35_000.0,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_reason="weak-dsp",
        pause_similarity=0.35,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-38_130.8,
        previous_expected_delta_ms=-35_000.0,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="recovered",
        pause_similarity=0.274,
        previous_quality=0.287,
        next_quality=0.0,
        expected_delta_ms=7_451.0,
        best_pause_is_later=True,
        best_pause_similarity=0.433,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="recovered",
        pause_similarity=0.274,
        previous_quality=0.287,
        next_quality=0.0,
        expected_delta_ms=7_451.0,
        best_pause_is_later=True,
        best_pause_similarity=0.373,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.401,
        previous_quality=0.0,
        next_quality=0.018,
        expected_delta_ms=-7_775.6,
        best_pause_is_later=True,
        best_pause_similarity=0.502,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_reason="micro-tempo",
        pause_similarity=0.406,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-21_814.5,
        previous_expected_delta_ms=-20_000.0,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_reason="dsp",
        pause_similarity=0.406,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-21_814.5,
        previous_expected_delta_ms=-20_000.0,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.469,
        previous_quality=0.511,
        next_quality=0.0,
        expected_delta_ms=-34_038.3,
        opening_self_repeat_chars=8,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.469,
        previous_quality=0.511,
        next_quality=0.0,
        expected_delta_ms=-34_038.3,
        opening_self_repeat_chars=0,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.483,
        previous_quality=0.0,
        next_quality=0.489,
        expected_delta_ms=25_971.8,
        opening_self_repeat_chars=6,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="recovered",
        pause_similarity=0.242,
        previous_quality=0.199,
        next_quality=0.0,
        expected_delta_ms=5_642.8,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.384,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-9_438.2,
        best_pause_is_later=True,
        best_pause_similarity=0.517,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.384,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-9_438.2,
        best_pause_is_later=False,
        best_pause_similarity=0.517,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="recovered",
        pause_similarity=0.26,
        previous_quality=0.199,
        next_quality=0.0,
        expected_delta_ms=5_642.8,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="first_opening_repair",
        pause_similarity=0.315,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-81_597.0,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="first_opening_repair",
        pause_similarity=0.58,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-81_597.0,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="first_opening_repair",
        pause_similarity=0.409,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-24_058.5,
        previous_expected_delta_ms=11_388.6,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="first_opening_repair",
        pause_similarity=0.409,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-5_000.0,
        previous_expected_delta_ms=11_388.6,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="weak_pause_recovery",
        pause_similarity=0.393,
        previous_quality=0.692,
        next_quality=0.0,
        expected_delta_ms=60_262.3,
        previous_expected_delta_ms=49_254.0,
        opening_shared_prefix_chars=5,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="recovered",
        pause_similarity=0.283,
        previous_quality=0.0,
        next_quality=0.06,
        expected_delta_ms=-52_619.2,
        previous_expected_delta_ms=-34_830.4,
        opening_shared_prefix_chars=13,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="weak_pause_recovery",
        pause_similarity=0.253,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-9_466.4,
        previous_expected_delta_ms=-20_425.9,
        opening_shared_prefix_chars=0,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="weak_pause_recovery",
        pause_similarity=0.253,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-11_000.0,
        previous_expected_delta_ms=-20_000.0,
        opening_shared_prefix_chars=0,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="weak_pause_recovery",
        pause_similarity=0.362,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=34_914.8,
        previous_expected_delta_ms=3_372.4,
        opening_shared_prefix_chars=0,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="weak_pause_recovery",
        pause_similarity=0.362,
        previous_quality=0.20,
        next_quality=0.0,
        expected_delta_ms=34_914.8,
        previous_expected_delta_ms=3_372.4,
        opening_shared_prefix_chars=0,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.460,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=42_315.8,
        previous_expected_delta_ms=57_614.3,
        opening_shared_prefix_chars=3,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.553,
        previous_quality=0.0,
        next_quality=0.022,
        expected_delta_ms=-23_637.0,
        previous_expected_delta_ms=-7_390.4,
        opening_shared_prefix_chars=0,
        opening_internal_repeat_chars=12,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.553,
        previous_quality=0.0,
        next_quality=0.022,
        expected_delta_ms=-23_637.0,
        previous_expected_delta_ms=-7_390.4,
        opening_shared_prefix_chars=0,
        opening_internal_repeat_chars=0,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="weak_pause_recovery",
        pause_similarity=0.256,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-1_771.9,
        previous_expected_delta_ms=-114_469.8,
        opening_shared_prefix_chars=20,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="post_selected_micro_recovery",
        pause_similarity=0.542,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=3_753.5,
        previous_expected_delta_ms=-34.3,
        opening_shared_prefix_chars=6,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="preword_micro_recovery",
        pause_similarity=0.313,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-38_704.1,
        previous_expected_delta_ms=-45_678.1,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="first_opening_repair",
        pause_similarity=0.315,
        previous_quality=0.0,
        next_quality=0.20,
        expected_delta_ms=74_000.0,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="opening_run_repair",
        pause_similarity=0.315,
        previous_quality=0.0,
        next_quality=0.0,
        expected_delta_ms=-81_597.0,
        pause_trust_similarity=0.50,
    )
    assert _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="opening_run_repair",
        pause_similarity=0.352,
        previous_quality=0.0,
        next_quality=0.276,
        expected_delta_ms=41_589.4,
        previous_expected_delta_ms=24_918.1,
        pause_trust_similarity=0.50,
    )
    assert not _pause_owner_geometry_is_unreliable(
        from_pause=True,
        action="opening_run_repair",
        pause_similarity=0.352,
        previous_quality=0.20,
        next_quality=0.276,
        expected_delta_ms=41_589.4,
        previous_expected_delta_ms=24_918.1,
        pause_trust_similarity=0.50,
    )
    assert not _accept_anchored_full_span_onset(
        8450.0,
        0.09,
        left_anchor_ms=1000.0,
        right_anchor_ms=9000.0,
    )


def test_repeated_opening_with_impossible_previous_span_is_not_an_anchor():
    assert _repeated_opening_pause_anchor_is_ambiguous(
        from_pause=True,
        pause_similarity=0.380,
        previous_quality=0.042,
        current_opening="وَلَقَدۡ",
        next_opening="وَلَقَدۡ",
        pause_trust_similarity=0.50,
    )
    assert not _repeated_opening_pause_anchor_is_ambiguous(
        from_pause=True,
        pause_similarity=0.380,
        previous_quality=0.042,
        current_opening="وَلَقَدۡ",
        next_opening="فَٱنظُرۡ",
        pause_trust_similarity=0.50,
    )


def test_ambiguous_observed_pause_detects_shifted_section_start():
    assert _ambiguous_observed_pause_starts_shifted_section(
        from_pause=True,
        action="observed_current",
        pause_similarity=0.461,
        opening_shared_prefix_chars=6,
        current_expected_delta_ms=44_244.3,
        previous_expected_delta_ms=19_558.1,
        pause_trust_similarity=0.50,
    )
    assert not _ambiguous_observed_pause_starts_shifted_section(
        from_pause=True,
        action="repair_pause_recovery",
        pause_similarity=0.315,
        opening_shared_prefix_chars=8,
        current_expected_delta_ms=-81_597.3,
        previous_expected_delta_ms=-46_567.5,
        pause_trust_similarity=0.50,
    )
    assert not _repeated_opening_pause_anchor_is_ambiguous(
        from_pause=True,
        pause_similarity=0.580,
        previous_quality=0.042,
        current_opening="وَلَقَدۡ",
        next_opening="وَلَقَدۡ",
        pause_trust_similarity=0.50,
    )


def test_adjacent_repeated_direct_owners_trigger_atomic_section_read():
    kwargs = {
        "current_from_pause": True,
        "current_action": "observed_current",
        "current_similarity": 0.358,
        "current_previous_quality": 0.0,
        "current_expected_delta_ms": 6_158.9,
        "previous_from_pause": True,
        "previous_action": "observed_current",
        "previous_similarity": 0.456,
        "previous_previous_quality": 0.0,
        "previous_expected_delta_ms": 215.5,
        "opening_shared_prefix_chars": 8,
        "pause_trust_similarity": 0.50,
    }
    assert _adjacent_repeated_owners_start_shifted_section(**kwargs)
    assert not _adjacent_repeated_owners_start_shifted_section(
        **{**kwargs, "opening_shared_prefix_chars": 5}
    )
    assert not _adjacent_repeated_owners_start_shifted_section(
        **{**kwargs, "previous_previous_quality": 0.11}
    )
    assert not _adjacent_repeated_owners_start_shifted_section(
        **{**kwargs, "current_expected_delta_ms": 5_000.0}
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


def test_shifted_geometry_long_pause_start_requires_untouched_impossible_prior():
    kwargs = {
        "from_pause": True,
        "action": "recovered",
        "pause_reason": "dsp",
        "pause_duration_ms": 2000.0,
        "current_start_ms": 318860.0,
        "pause_center_ms": 318860.0,
        "expected_delta_ms": -17397.7,
        "prev_quality": 0.0,
        "next_quality": 0.0,
        "timing_adjusted": False,
    }
    assert _shifted_geometry_long_pause_start_candidate(**kwargs)
    assert not _shifted_geometry_long_pause_start_candidate(
        **{**kwargs, "expected_delta_ms": -4376.1}
    )
    assert not _shifted_geometry_long_pause_start_candidate(
        **{**kwargs, "prev_quality": 0.11}
    )
    assert not _shifted_geometry_long_pause_start_candidate(
        **{**kwargs, "next_quality": 0.11}
    )
    assert not _shifted_geometry_long_pause_start_candidate(
        **{**kwargs, "current_start_ms": 319737.0}
    )
    assert not _shifted_geometry_long_pause_start_candidate(
        **{**kwargs, "timing_adjusted": True}
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


def test_post_selected_micro_context_read_requires_strong_in_owner_ctc():
    accepted = {
        "pause_reason": "dsp",
        "action": "post_selected_micro_recovery",
        "pause_duration_ms": 2720.0,
        "pause_sim": 0.467,
        "prev_quality": 0.0,
        "next_quality": 0.564,
        "candidate_ms": 457180.0,
        "candidate_conf": 0.82,
        "pause_start_ms": 454760.0,
        "pause_end_ms": 457480.0,
        "conf_reliable": 0.70,
    }
    assert _post_selected_micro_context_read_candidate(**accepted)
    assert not _post_selected_micro_context_read_candidate(
        **{**accepted, "candidate_conf": 0.69}
    )
    assert not _post_selected_micro_context_read_candidate(
        **{**accepted, "candidate_ms": 454800.0}
    )
    assert not _post_selected_micro_context_read_candidate(
        **{**accepted, "next_quality": 0.49}
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
    assert _late_long_dsp_fallback_start_candidate(
        pause_reason="dsp",
        pause_duration_ms=2820.0,
        action="pause_fallback",
        sim=0.559,
        prev_speech_ms=0.0,
        next_quality=0.756,
    )
    assert not _late_long_dsp_fallback_start_candidate(
        pause_reason="dsp",
        pause_duration_ms=2820.0,
        action="pause_fallback",
        sim=0.559,
        prev_speech_ms=600.0,
        next_quality=0.756,
    )

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


def test_overlapping_dsp_replaces_only_impossible_micro_owner():
    kwargs = {
        "action": "observed_current",
        "pause_reason": "micro-tempo",
        "expected_delta_ms": -57_495.0,
        "prev_speech_ms": 0.0,
        "next_quality": 0.744,
        "overlap_ms": 140.0,
    }
    assert _overlapping_dsp_replaces_micro_owner(**kwargs)
    assert not _overlapping_dsp_replaces_micro_owner(
        **{**kwargs, "prev_speech_ms": 200.0}
    )
    assert not _overlapping_dsp_replaces_micro_owner(
        **{**kwargs, "overlap_ms": 50.0}
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
    assert not any(e["action"] == "confirmed" and e["boundary"] == 2 for e in events)
    assert any(e["action"] == "deferred_future_anchor" and e["boundary"] == 2 for e in events)
    assert any(
        e["action"] == "wasl" and e["boundary"] == 1
        and e["reason"] == "missing_before_future"
        for e in events
    )
    assert any(
        e["action"] == "unresolved_current" and e["boundary"] == 1
        and e["reason"] == "missing_before_future"
        for e in events
    )


def test_sequential_pause_match_marks_future_anchor_deferred_after_unrecovered_gap():
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
    assert not any(e["action"] == "future_anchor" for e in events)
    assert not any(e["action"] == "confirmed" and e["boundary"] == 2 for e in events)
    assert any(e["action"] == "deferred_future_anchor" and e["boundary"] == 2 for e in events)
    assert any(
        e["action"] == "unresolved_current" and e["boundary"] == 1
        and e["reason"] == "missing_before_future"
        for e in events
    )


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


def test_final_start_guard_keeps_first_and_last_ayah_strictly_ordered():
    rows, repairs = _enforce_strictly_increasing_starts([
        (1, 0, 0.9),
        (2, 0, 0.8),
        (3, -20, 0.7),
        (4, 4, 0.6),
    ])

    assert [row[1] for row in rows] == [0, 1, 2, 4]
    assert [row[2] for row in rows] == [0.9, 0.0, 0.0, 0.6]
    assert repairs == [
        {
            "ayah": 2,
            "reason": "non_increasing_timestamp",
            "requested_ms": 0,
            "corrected_ms": 1,
        },
        {
            "ayah": 3,
            "reason": "non_increasing_timestamp",
            "requested_ms": 0,
            "corrected_ms": 2,
        },
    ]


def test_boundary_diagnostics_marks_rollback_merge_and_review_reason():
    diagnostics = _fast_boundary_diagnostics(
        [
            {
                "ayah": 11,
                "action": "backtrack_before_future",
                "heard_ayah": 13,
            },
            {"ayah": 11, "action": "future_gap", "reason": "future_match"},
        ],
        [
            {"ayah": 12, "from_pause": False, "match_action": "wasl"},
            {
                "ayah": 13,
                "from_pause": True,
                "timing_adjusted": True,
                "timing_strategy": "context_block_edge_recovery",
            },
        ],
        [{"ayah": 14, "kind": "owner-anchor-conflict"}],
        [
            {
                "ayah": 15,
                "reason": "non_increasing_timestamp",
                "requested_ms": 500,
                "corrected_ms": 501,
            }
        ],
    )

    assert diagnostics["counts"] == {
        "rollback": 1,
        "merged": 1,
        "suspicious": 3,
        "corrections": 1,
        "monotonic_repairs": 1,
    }
    assert diagnostics["rollbacks"] == [
        {"ayah": 11, "reason": "future_gap_local_rollback", "heard_ayah": 13}
    ]
    assert diagnostics["merged_intervals"] == [
        {"ayah": 12, "reason": "wasl_interval"}
    ]


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


def test_strong_unique_ctc_can_reject_weak_internal_micro_pause():
    assert _pre_owner_ctc_overrides_weak_micro_pause(
        pause_reason="micro-tempo",
        pause_similarity=0.34,
        pause_trust_similarity=0.45,
        refined_onset_ms=313_432.0,
        refined_confidence=0.74,
        confidence_reliable=0.45,
        pause_start_ms=313_980.0,
        previous_start_ms=302_000.0,
        next_start_ms=323_000.0,
        opening_shared_prefix_chars=0,
    )
    assert not _pre_owner_ctc_overrides_weak_micro_pause(
        pause_reason="micro-tempo",
        pause_similarity=0.34,
        pause_trust_similarity=0.45,
        refined_onset_ms=313_432.0,
        refined_confidence=0.74,
        confidence_reliable=0.45,
        pause_start_ms=313_980.0,
        previous_start_ms=302_000.0,
        next_start_ms=323_000.0,
        opening_shared_prefix_chars=8,
    )


def test_only_explicit_insufficient_audio_is_selected_for_local_repair():
    assert _insufficient_audio_ayahs([
        {"ayah": 42, "match_action": "wasl"},
        {"ayah": 45, "match_action": "insufficient_audio"},
        {"ayah": 46, "match_action": "recovered", "pause_sim": 0.29},
        {"ayah": 47, "match_action": "wasl", "pause_sim": 0.29},
    ]) == [45]


def test_bounded_local_repair_spans_include_shifted_run_and_tail():
    ayahs = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
    starts = {
        67: 1_270_000.0,
        68: 1_306_233.0,
        69: 1_350_905.0,
        70: 1_350_906.0,
        71: 1_373_010.0,
        72: 1_416_330.0,
        73: 1_393_650.0,
        74: 1_422_266.0,
        75: 1_447_815.0,
        76: 1_470_000.0,
    }

    assert _bounded_local_repair_spans(
        ayahs,
        [70],
        {67, 72, 76},
        starts,
    ) == [(0, 6), (6, 9)]


def test_bounded_local_repair_spans_need_two_anchors():
    ayahs = [67, 68, 69, 70]
    starts = {67: 100.0, 68: 200.0, 69: 300.0, 70: 400.0}

    assert _bounded_local_repair_spans(ayahs, [67], {69}, starts) == []
    assert _bounded_local_repair_spans(ayahs, [70], {67}, starts) == []
    assert _bounded_local_repair_spans(ayahs, [99], {67, 70}, starts) == []


def test_low_verify_fallback_requires_both_independent_failure_signals():
    assert _requires_low_verify_ctc_fallback(0.35, 0.65, False)
    assert not _requires_low_verify_ctc_fallback(0.44, 0.65, False)
    assert not _requires_low_verify_ctc_fallback(0.35, 0.92, False)
    assert not _requires_low_verify_ctc_fallback(0.35, 0.65, True)
