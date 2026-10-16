"""Timing evidence for the bounded next-pause recovery path."""
import pytest
from unittest.mock import patch

from surah_cutter.core.fast_executor import Boundary, Policy, _review_addressed_recovery_onsets
from surah_cutter.data import quran_text
from test_fast_empty_region import responder
from test_fast_executor import Fake, observation, pause, run


@pytest.mark.parametrize('stationary', [False, True])
def test_recovered_wide_interval_keeps_identity_for_review(stationary):
    earlier = Boundary(1, 1000, 'pause', .99, read_id=0, proof={'pause_id': 0})
    recovered = Boundary(2, 2000, 'pause', .99, read_id=1,
                         proof={'pause_id': 1, 'bounded_following_origin_ayah': 1})
    downstream = Boundary(3, 3000, 'pause', .99, read_id=2, proof={'pause_id': 2})
    narrow = Boundary(4, 4000, 'pause', .99, read_id=3, proof={'pause_id': 3})
    geometry = [{'quiet_stationarity': stationary, 'midpoint_interval_radius_ms': x}
                for x in (1000, 200, 800, 500)]
    rows, reviews = _review_addressed_recovery_onsets(
        [earlier, recovered, downstream, narrow], geometry, Policy())
    assert rows[:2] == [earlier, recovered] and rows[3] == narrow
    assert rows[2].start_ms == downstream.start_ms and rows[2].confidence == 0
    assert rows[2].evidence_kind == 'estimate' and rows[2].proof is None
    assert reviews[0]['identity_proof'] == downstream.proof
    assert reviews[0]['internal_anchor_ms'] == downstream.start_ms
    assert reviews[0]['reason'] == 'recovered_wide_midpoint_unproved'
    assert _review_addressed_recovery_onsets([earlier, downstream], geometry, Policy()) == (
        [earlier, downstream], [])


def test_first_recovered_anchor_is_also_checked_for_temporal_uncertainty():
    anchor = Boundary(1, 1000, 'pause', .99, read_id=0,
                      proof={'pause_id': 0, 'bounded_following_origin_ayah': 1})
    geometry = [{'quiet_stationarity': True, 'midpoint_interval_radius_ms': 800}]
    rows, reviews = _review_addressed_recovery_onsets([anchor], geometry, Policy())
    assert rows[0].confidence == 0 and rows[0].start_ms == anchor.start_ms
    assert reviews[0]['identity_proof'] == anchor.proof


def test_weak_recovery_anchor_cannot_hide_downstream_timing_uncertainty():
    fake = Fake(responder, words=quran_text.surah_words(114))
    fake.timing, fake.model_id = False, 'whisper-tarteel-base'

    def respond(f, start, duration):
        if start == 15000:
            return observation(f.phrase(4), quality=.4 if duration < 2000 else .99, positions=False)
        if start == 21400:
            return observation(f.phrase(5), positions=False)
        return responder(f, start, duration)

    fake.responder = respond

    def geometry(pcm, sr, interval):
        return {'accepted': True, 'quiet_stationarity': False,
                'midpoint_interval_radius_ms': (interval.end_ms - interval.start_ms) / 2}

    with patch('surah_cutter.core.fast_executor.pause_geometry', side_effect=geometry):
        rows, trace = run(fake, [pause(1000), pause(6000), pause(10000), pause(15000),
                                pause(20000, 1600)], policy=Policy(sparse_anchors=True))
    assert rows[3][2] == rows[4][2] == 0
    assert trace['boundaries'][3]['review_reason'] == 'weak_complete_sparse_prefix_unproved_time'
    assert trace['boundaries'][4]['review_reason'] == 'recovered_wide_midpoint_unproved'
