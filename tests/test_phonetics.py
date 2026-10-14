"""Shared phonetic helpers used outside the removed legacy FAST."""
from surah_cutter.core.phonetics import (
    collapse as _collapse, common_prefix_len as _common_prefix_len,
    unit_len as _unit_len, onset_similarity as _sim,
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


def test_sim_penalizes_previous_tail_before_opening():
    clean = _sim("innalladhīnaāmanū", "innalladhīnaāmanū", chars=24)
    with_tail = _sim("qālainnaalladhīnaāmanū", "innalladhīnaāmanū", chars=24)

    assert clean == 1.0
    assert with_tail < clean - 0.20


def test_sim_can_compare_past_default_prefix_for_repeated_starts():
    dec = "innalladhīnaāmanū"
    kafaru = "innalladhīnakafarū"
    amanu = "innalladhīnaāmanū"

    assert _sim(dec, kafaru, chars=10) == _sim(dec, amanu, chars=10)
    assert _sim(dec, amanu, chars=24) > _sim(dec, kafaru, chars=24)
