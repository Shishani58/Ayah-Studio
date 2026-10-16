"""Behavioral contracts for finite ASR spelling cues; actual canonical plans."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from surah_cutter.core.fast_executor import (
    Boundary, Policy, _asr_head_competitors, _incomplete_speech_edge, _partial_asr_head,
    _review_addressed_recovery_onsets, validate_boundaries,
)
from test_fast_executor import Fake, observation, pause, run


@pytest.mark.parametrize('heard,expected', [('اولى', 'اولئك'), ('اانت', 'اانتم'),
                                           ('فمنشر', 'فمن'), ('جزاؤه', 'جزاؤهم')])
def test_clipped_and_glued_spelling_is_only_a_search_cue(heard, expected):
    assert _partial_asr_head(observation((heard,), positions=False), (expected,), .6)
    assert not _partial_asr_head(observation((heard,), quality=.59, positions=False), (expected,), .6)
    assert not _partial_asr_head(observation((heard,)), (expected,), .6)
    assert not _partial_asr_head(replace(observation((heard,), positions=False), truncated=True),
                                 (expected,), .6)


@pytest.mark.parametrize('heard', [(), ('واولى',), ('wrong',), ('اولى', 'wrong'), ('او',)])
def test_wrong_short_and_internal_words_cannot_request_this_opening(heard):
    assert not _partial_asr_head(observation(heard, positions=False), ('اولئك', 'على'), .6)


def test_cue_treats_starts_and_internal_words_symmetrically():
    groups = {1: ('اولئك', 'على'), 2: ('body', 'اولئك', 'هم')}
    ends = {1: [1, 2], 2: [1, 2, 3]}
    obs = observation(('اولى',), positions=False)
    assert _asr_head_competitors(obs, groups, ends, .6) == [(1, 0), (2, 1)]
    pcm = np.ones(16000 * 5, dtype=np.float32) * .1
    assert _incomplete_speech_edge(pcm, 16000, 2000, obs, ('اولئك',), groups, ends, 1, .6,
                                   asr_cue=True)
    pcm[32000:] = 0
    assert not _incomplete_speech_edge(pcm, 16000, 2000, obs, ('اولئك',), groups, ends, 1, .6,
                                       asr_cue=True)


def clipped_fake(kind):
    fake = Fake(lambda *args: observation((), positions=False))
    fake.model_id = 'whisper-tarteel-base'
    fake.timing = False
    rendered = list(fake.rendered)
    rendered[0] = ('اولئك',)
    fake.rendered = tuple(rendered)

    def respond(f, start, duration):
        if start == 1000:
            if duration < 2000:
                return observation(('اولى',), positions=False)
            if kind == 'good':
                return observation(f.phrase(1), positions=False)
            if kind == 'weak':
                return observation(f.phrase(1), quality=.59, positions=False)
            if kind == 'internal':
                return observation(('body',) + f.phrase(1), positions=False)
            if kind == 'truncated':
                return replace(observation(f.phrase(1), positions=False), truncated=True)
            return observation(('wrong',), positions=False)
        if start == 6000:
            return observation(f.phrase(1), positions=False)
        return observation((), positions=False)
    fake.responder = respond
    return fake


def test_executor_requires_new_full_proof_and_accounts_every_crop():
    rows, trace = run(clipped_fake('good'), [pause(1000)])
    assert rows[0] == (1, 1100, .99)
    boundary = trace['boundaries'][0]
    assert boundary['proof']['partial_asr_head_cue_id'] == 0
    assert boundary['read_id'] == 1
    assert [r['window_ms'] for r in trace['acoustic_reads'][:2]] == [[1000., 2500.], [1000., 3700.]]
    assert trace['audio_ms'] == sum(b-a for a, b in [r['window_ms'] for r in trace['acoustic_reads']])
    assert trace['local_read_limits']['max_coverage'] == .65


@pytest.mark.parametrize('kind', ['wrong', 'weak', 'internal', 'truncated'])
def test_failed_optional_extension_does_not_hide_later_independent_pause(kind):
    rows, trace = run(clipped_fake(kind), [pause(1000), pause(6000)])
    assert rows[0] == (1, 6100, .99)
    assert trace['boundaries'][0]['proof'].get('partial_asr_head_cue_id') is None
    assert any(e.get('reason') == 'short_word_extension_unresolved_resume_candidates'
               for e in trace['transitions'])


def test_partial_word_cannot_override_rejected_temporal_geometry():
    with patch('surah_cutter.core.fast_executor.pause_geometry', return_value={
        'accepted': False, 'reason': 'decaying_tail_midpoint_unproved'}):
        rows, trace = run(clipped_fake('good'), [pause(1000)])
    assert all(row[2] == 0 for row in rows)
    assert not any(e['state'] == 'SEARCH_CUE_ONLY' for e in trace['transitions'])


def test_disabled_sparse_recovery_cannot_confirm_future_from_cue():
    fake = clipped_fake('wrong')
    fake.model_id = 'whisper-tarteel-base'
    original = fake.responder
    fake.responder = lambda f, s, d: observation(f.phrase(2), positions=False) if s == 6000 else original(f, s, d)
    rows, _ = run(fake, [pause(1000), pause(6000)], policy=Policy(sparse_anchors=False))
    assert rows[0][2] == 0 and rows[1][2] == 0


def test_new_recovery_keeps_owner_chain_but_publishes_wide_decay_for_review():
    old = Boundary(1, 1000, 'pause', .99, read_id=0, proof={'pause_id': 0})
    recovery = Boundary(2, 2000, 'pause', .99, read_id=2,
                        proof={'pause_id': 1, 'partial_asr_head_cue_id': 1})
    downstream = Boundary(3, 3000, 'pause', .99, read_id=3, proof={'pause_id': 2})
    narrow = Boundary(4, 4000, 'pause', .99, read_id=4, proof={'pause_id': 3})
    geometry = [{'quiet_stationarity': False, 'midpoint_interval_radius_ms': x}
                for x in (800, 200, 530, 500)]
    rows, reviews = _review_addressed_recovery_onsets([old, recovery, downstream, narrow], geometry, Policy())
    assert rows[:2] == [old, recovery] and rows[3] == narrow
    assert rows[2].start_ms == downstream.start_ms and rows[2].confidence == 0
    assert rows[2].evidence_kind == 'estimate' and rows[2].proof is None
    assert reviews[0]['identity_proof'] == downstream.proof
    assert reviews[0]['internal_anchor_ms'] == downstream.start_ms
    assert _review_addressed_recovery_onsets([old, downstream], geometry, Policy()) == ([old, downstream], [])


def test_stationary_quiet_interval_does_not_prove_a_precise_midpoint():
    anchor = Boundary(1, 1000, 'pause', .99, read_id=1,
                      proof={'pause_id': 0, 'partial_asr_head_cue_id': 0})
    geometry = [{'quiet_stationarity': True, 'midpoint_interval_radius_ms': 800}]
    result, reviews = _review_addressed_recovery_onsets([anchor], geometry, Policy())
    assert result[0].confidence == 0 and result[0].start_ms == anchor.start_ms
    assert reviews[0]["reason"] == "recovered_wide_midpoint_unproved"


def test_validator_rejects_mutated_cue_id_and_wrong_crop_provenance():
    _, trace = run(clipped_fake('good'), [pause(1000)])
    boundary = Boundary(**trace['boundaries'][0])
    with pytest.raises(ValueError, match='invalid_partial_head_cue'):
        validate_boundaries([replace(boundary, proof={**boundary.proof,
            'partial_asr_head_cue_id': boundary.read_id})], [1], 40000, trace['acoustic_reads'])
    calls = [dict(r) for r in trace['acoustic_reads']]
    calls[0]['window_ms'] = [1100, 2500]
    with pytest.raises(ValueError, match='unrelated_partial_head_cue'):
        validate_boundaries([boundary], [1], 40000, calls)


def test_validator_rejects_strong_contradiction_after_new_opening():
    fake = clipped_fake('good')
    _, trace = run(fake, [pause(1000)])
    boundary = Boundary(**trace['boundaries'][0])
    calls = [dict(r) for r in trace['acoustic_reads']]
    final = calls[boundary.read_id]
    final['observed'] = {**final['observed'], 'units': ['اولئك', 'wrong']}
    group = tuple(u for w, units in zip(fake.words, fake.rendered) if w.ayah == 1 for u in units)
    with pytest.raises(ValueError, match='contradictory_partial_head_continuation'):
        validate_boundaries([boundary], [1], 40000, calls, groups={1: group},
                            word_ends={1: list(range(1, len(group)+1))}, threshold=.6)


def test_unmeasured_tiny_rollout_keeps_the_previous_search():
    fake = clipped_fake('good')
    fake.model_id = 'whisper-tarteel-tiny'
    rows, trace = run(fake, [pause(1000)])
    assert all(row[2] == 0 for row in rows)
    assert not any(e['state'] == 'SEARCH_CUE_ONLY' for e in trace['transitions'])


def test_new_optional_route_at_eof_never_repeats_identical_window():
    fake = clipped_fake('wrong')
    rendered = list(fake.rendered)
    other = next(i for i, w in enumerate(fake.words) if w.ayah == 2)
    rendered[other] = ('اولئك',)
    fake.rendered = tuple(rendered)
    fake.responder = lambda f, s, d: observation(
        ('اولى',) if d < 1700 else ('اولئك',), positions=False
    ) if s == 8000 else observation((), positions=False)
    rows, trace = run(fake, [pause(8000)], duration=10000)
    assert all(row[2] == 0 for row in rows)
    assert trace['transitions'][-1]['state'] == 'RESULT_READY'
    assert any(e.get('reason') == 'addressed_context_exhausted_at_audio_end'
               for e in trace['transitions'])
    windows = [r['window_ms'] for r in trace['acoustic_reads']]
    assert len(windows) == len({tuple(w) for w in windows})


def test_failed_future_cue_cannot_create_a_sticky_current_prefix():
    fake = clipped_fake('wrong')
    rendered = list(fake.rendered)
    rendered[0] = ('currentword',)
    next_ayah = next(i for i, w in enumerate(fake.words) if w.ayah == 2)
    rendered[next_ayah] = ('futureword',)
    rendered[-1] = ('currentword',)
    fake.rendered = tuple(rendered)
    fake.responder = lambda f, s, d: (
        observation(('futurx',) if d < 2000 else ('currentword',), positions=False) if s == 1000
        else observation(f.phrase(1), positions=False) if s == 6000
        else observation((), positions=False))
    rows, trace = run(fake, [pause(1000), pause(6000)], policy=Policy(sparse_anchors=True))
    assert rows[0] == (1, 6100, .99)
    assert any(e.get('possible_ayah') == 2 and e['state'] == 'SEARCH_CUE_ONLY'
               for e in trace['transitions'])
