"""A nonempty failed tempo region may try one bounded next DSP candidate."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from surah_cutter.core.fast_executor import (
    Boundary, Policy, _review_addressed_recovery_onsets, validate_boundaries,
)
from surah_cutter.data import quran_text
from test_fast_empty_region import responder
from test_fast_executor import Fake, observation, pause, run


def probe_fake(model='whisper-tarteel-base'):
    fake = Fake(responder)
    fake.timing = False
    fake.model_id = model
    return fake


def test_failed_nonempty_region_gets_one_independent_next_anchor():
    rows, trace = run(probe_fake(), [pause(1000), pause(6000), pause(10000), pause(15000)],
                      policy=Policy(sparse_anchors=True))
    assert rows[2][2] == 0 and rows[3] == (4, 15100, .99)
    proof = trace['boundaries'][3]['proof']
    assert proof['bounded_following_origin_ayah'] == 3
    events = [e for e in trace['transitions'] if e.get('reason') == 'optional_after_unproved_tempo_candidates']
    assert len(events) == 1 and events[0]['lookahead_region_ms'][0] < 15000
    assert len([r for r in trace['acoustic_reads'] if r['window_ms'][0] == 15000]) == 1


def test_base_probe_does_not_change_tiny_or_exceed_candidate_cap():
    for fake, policy in [(probe_fake('whisper-tarteel-tiny'), Policy(sparse_anchors=True)),
                         (probe_fake(), Policy(sparse_anchors=True, max_candidates=1))]:
        rows, trace = run(fake, [pause(1000), pause(6000), pause(10000), pause(15000)], policy=policy)
        assert rows[2][2] == rows[3][2] == 0
        assert not any(r['window_ms'][0] == 15000 for r in trace['acoustic_reads'])


def test_an_existing_strong_incomplete_prefix_blocks_the_optional_later_pause():
    fake = probe_fake()
    rendered = list(fake.rendered)
    first3 = next(i for i, w in enumerate(fake.words) if w.ayah == 3)
    rendered[-1] = rendered[first3]
    fake.rendered = tuple(rendered)
    fake.responder = lambda f, s, d: observation(f.phrase(3)[:1], positions=False) if s == 10000 else responder(f, s, d)
    rows, trace = run(fake, [pause(1000), pause(6000), pause(10000), pause(15000)],
                      policy=Policy(sparse_anchors=True))
    assert rows[2][2] == rows[3][2] == 0
    assert not any(r['window_ms'][0] == 15000 for r in trace['acoustic_reads'])


def test_optional_following_owner_cannot_jump_beyond_two_ayahs():
    fake = Fake(responder, words=quran_text.surah_words(114))
    fake.timing, fake.model_id = False, 'whisper-tarteel-base'
    fake.responder = lambda f, s, d: observation(f.phrase(6), positions=False) if s == 15000 else responder(f, s, d)
    rows, trace = run(fake, [pause(1000), pause(6000), pause(10000), pause(15000)],
                      policy=Policy(sparse_anchors=True))
    assert rows[5][2] == 0
    assert any(r['window_ms'][0] == 15000 for r in trace['acoustic_reads'])
    assert all(e.get('ayah', 0) != 6 for e in trace['transitions'] if e['state'] == 'CONFIRMED')


def test_validator_rejects_fabricated_following_owner_origin():
    _, trace = run(probe_fake(), [pause(1000), pause(6000), pause(10000), pause(15000)],
                   policy=Policy(sparse_anchors=True))
    boundary = Boundary(**trace['boundaries'][3])
    corrupt = replace(boundary, proof={**boundary.proof, 'bounded_following_origin_ayah': 1})
    with pytest.raises(ValueError, match='unbounded_following_owner'):
        validate_boundaries([corrupt], [4], 40000, trace['acoustic_reads'])


def test_recovery_origin_must_be_a_positive_ayah_number():
    _, trace = run(probe_fake(), [pause(1000), pause(6000), pause(10000), pause(15000)],
                   policy=Policy(sparse_anchors=True))
    boundary = Boundary(**trace['boundaries'][3])
    corrupt = replace(boundary, ayah=1, proof={**boundary.proof, 'bounded_following_origin_ayah': 0})
    with pytest.raises(ValueError, match='unbounded_following_owner'):
        validate_boundaries([corrupt], [1], 40000, trace['acoustic_reads'])


def test_existing_body_cue_head_retry_also_retains_next_pause_origin():
    fake = probe_fake()
    rendered = list(fake.rendered)
    rendered[6:10] = [('head',), ('wanted',), ('body',), ('suffix',)]
    rendered[10:12] = [('head',), ('rival',)]
    fake.rendered = tuple(rendered)

    def respond(f, start, duration):
        if start == 10000:
            return observation(('body', 'suffix'), positions=False)
        if start == 15000:
            if duration < 2000:
                return observation(('headx',), positions=False)
            return observation(('head', 'wrong' if duration < 3000 else 'wanted'), positions=False)
        return responder(f, start, duration)

    fake.responder = respond
    rows, trace = run(fake, [pause(1000), pause(6000), pause(10000), pause(15000)],
                      policy=Policy(sparse_anchors=True))
    assert rows[2][1:] == (15100, .99)
    proof = trace['boundaries'][2]['proof']
    assert proof['head_boundary_recheck'] and proof['bounded_following_origin_ayah'] == 3


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
