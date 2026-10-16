"""Bounded recovery after no DSP candidate, without inventing the missed onset."""
from surah_cutter.core.fast_executor import Policy
from test_fast_executor import Fake, observation, pause, run


def responder(fake, start, duration):
    if start == 1000:
        return observation(fake.phrase(1), positions=False)
    if start == 6000:
        return observation(fake.phrase(2), positions=False)
    if start == 15000:
        return observation(fake.phrase(4), positions=False)
    return observation((), positions=False)


def test_next_sparse_anchor_leaves_missing_current_for_review():
    fake = Fake(responder)
    fake.timing = False
    rows, trace = run(fake, [pause(1000), pause(6000), pause(15000)],
                      policy=Policy(sparse_anchors=True))
    assert trace["boundaries"][2]["confidence"] == 0 and 3 in trace["review_required_ayahs"]
    assert rows[-1] == (4, 15100, .99)
    assert trace["boundaries"][3]["proof"]["unresolved_ayahs"] == [3]
    probes = [e for e in trace["transitions"] if e.get("reason") == "empty_tempo_region_next_dsp_pause"]
    assert len(probes) == 1
    assert probes[0]["lookahead_region_ms"][0] < 15000 <= probes[0]["lookahead_region_ms"][1]


def test_unheard_owner_never_becomes_a_confirmation():
    fake = Fake(lambda f, start, duration: observation(("unknown",), positions=False)
                if start == 15000 else responder(f, start, duration))
    fake.timing = False
    rows, trace = run(fake, [pause(1000), pause(6000), pause(15000)],
                      policy=Policy(sparse_anchors=True))
    assert trace["boundaries"][2]["confidence"] == trace["boundaries"][3]["confidence"] == 0
    assert len([r for r in trace["acoustic_reads"] if r["window_ms"][0] == 15000]) == 1


def test_next_pause_beyond_horizon_is_not_read():
    fake = Fake(responder)
    fake.timing = False
    rows, trace = run(fake, [pause(1000), pause(6000), pause(18000)],
                      policy=Policy(sparse_anchors=True, independent_search=False))
    assert trace["boundaries"][2]["confidence"] == trace["boundaries"][3]["confidence"] == 0
    assert not any(r["window_ms"][0] == 18000 for r in trace["acoustic_reads"])


def test_supported_wasl_recovery_keeps_its_existing_read():
    fake = Fake(lambda f, start, duration:
                observation((f.rendered[5][0], *f.phrase(3))) if start == 6100
                else responder(f, start, duration))
    rows, trace = run(fake, [pause(1000), pause(6000), pause(15000)],
                      policy=Policy(sparse_anchors=True))
    assert trace["boundaries"][2]["confidence"] > 0
    assert trace["boundaries"][2]["evidence_kind"] == "wasl"
    assert not any(e.get("reason") == "empty_tempo_region_next_dsp_pause" for e in trace["transitions"])
