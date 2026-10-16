"""Review ordering cues never masquerade as calibrated timing probabilities."""

import copy

import pytest

from surah_cutter.data.timing_accuracy import timing_accuracy


def assess(*, start=1000, end=2600, time=2350, level="high", pauses=True):
    check = dict(ayah=2, full_ctc_first_ms=end)
    if pauses:
        check["pause_local_decisions"] = [dict(
            start_ms=start, end_ms=end, accepted=True, owned=True)]
    trace = dict(ayahs=[dict(ayah=1, full_ctc_end_ms=start), check])
    before = copy.deepcopy(trace)
    result = timing_accuracy(check, time, trace, dict(level=level))
    assert trace == before
    return result


def test_high_acoustic_support_cannot_hide_wide_cut_range():
    r = assess()
    assert r["probability"] is None
    assert r["cut_span_ms"] == 1350
    assert r["cut_span_source"] == "local-pause"
    assert r["review_priority"] == "suspect"
    assert r["review_reasons"] == ["wide-cut-range"]


def test_narrow_supported_pause_does_not_invent_accuracy():
    r = assess(start=2100)
    assert r["review_priority"] == "supported"
    assert r["probability"] is None
    assert r["calibration"] == "unavailable"


def test_ctc_gap_is_labelled_separately_from_silence():
    r = assess(pauses=False)
    assert r["cut_span_source"] == "ctc-gap"
    assert r["review_priority"] == "suspect"


@pytest.mark.parametrize("start,end,time", [
    (None, 2600, 2350), (float("nan"), 2600, 2350),
    (True, 2600, 2350), (2600, 1000, 2350), (-100, 2600, 2350),
    (1000, 2600, 9000),
])
def test_bad_or_unrelated_interval_does_not_create_review_span(start, end, time):
    r = assess(start=start, end=end, time=time)
    assert r["cut_span_ms"] is None
    assert r["probability"] is None


def test_conflict_is_suspect_even_with_narrow_pause():
    assert assess(start=2100, level="low")["review_priority"] == "suspect"


def test_unknown_without_geometry_is_unassessed():
    assert assess(start=None, level="unknown")["review_priority"] == "unassessed"


def test_exact_tolerance_is_not_flagged_as_over_tolerance():
    assert assess(start=1000, end=2000, time=1500)["review_priority"] == "supported"
    assert assess(start=1000, end=2001, time=1500)["review_priority"] == "suspect"
