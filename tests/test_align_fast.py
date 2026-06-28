"""Unit tests for the pure (model-free) helpers of the fast aligner.

Only the top-level functions are exercised — they import nothing heavy, so these
run without the ONNX model or any audio. The DP matcher ``_match_boundaries`` is
the off-by-one-prone heart of Phase 1c; the rest guard the small primitives the
char prior and acoustic cost are built on.
"""
from __future__ import annotations

from types import SimpleNamespace

from surah_cutter.core.align_fast import (
    _ambiguous_openings,
    _collapse,
    _common_prefix_len,
    _count_waqf_marks,
    _interp_wasl,
    _match_boundaries,
    _sequential_pause_match,
    _sim,
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


def test_sim_identical_and_collapsed():
    assert _sim("kafaru", "kafaru") == 1.0
    # Recitation stretches vowels; collapse makes the stretched onset match.
    assert _sim("kafaaaaruuu", "kafaru") == 1.0
    assert _sim("xyz", "kafaru") < 0.3


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
        exp, ons, sim, min_sim=0.4, prior_window_ms=300.0, lookahead=2
    )

    assert matched == [1500.0, None, 6000.0]
    assert any(e["action"] == "recovered" and e["boundary"] == 0 for e in events)
    assert any(e["action"] == "future_anchor" and e["boundary"] == 2 for e in events)


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
    assert any(e["action"] == "future_anchor" and e["boundary"] == 2 for e in events)


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


def test_count_waqf_marks_from_mushaf_words():
    words = [
        SimpleNamespace(text="عَلَيۡهِۚ"),
        SimpleNamespace(text="ثُمَّ"),
        SimpleNamespace(text="قَلِيلٗاۖ"),
    ]

    assert _count_waqf_marks(words) == 2
