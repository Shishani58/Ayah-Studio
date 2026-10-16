"""Prepared behavioral checks; execute only in the final testing phase."""
from dataclasses import replace
import json

import pytest

from surah_cutter.core.fast_executor import Boundary, Policy, _expansion_evidence, validate_boundaries
from test_fast_executor import Fake, observation, pause, run


def vocative_fake(response):
    fake = Fake(lambda *args: observation(()))
    fake.timing = False
    rendered = list(fake.rendered)
    rendered[0] = ('يا', 'ايها')
    rival = next(i for i, word in enumerate(fake.words) if word.ayah == 2)
    rendered[rival] = ('يا', 'اولي')
    fake.rendered = tuple(rendered)
    fake.responder = lambda f, start, duration: (
        observation(response(duration), positions=False) if start == 1000 else observation((), positions=False))
    return fake


def test_complete_short_spoken_word_can_request_rest_of_canonical_word():
    fake = vocative_fake(lambda duration: ('يا',) if duration < 2000 else ('يا', 'ايها'))
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][2] > 0 and rows[0][1] == 1100
    assert len([r for r in trace['acoustic_reads'] if r['reason'].startswith('ayah=1:pause')]) == 2
    assert any(e.get('acoustic_anchor') == 'whole_spoken_prefix_incomplete_canonical_word'
               for e in trace['transitions'])
    assert trace['boundaries'][0]['proof']['phrase'] == ('يا', 'ايها')


@pytest.mark.parametrize('later', [('يا',), ('يا', 'اولي'), ('wrong',), ()])
def test_cue_never_confirms_missing_or_wrong_rest(later):
    fake = vocative_fake(lambda duration: ('يا',) if duration < 2000 else later)
    rows, _ = run(fake, [pause(1000)])
    assert rows[0][2] == 0


def test_weak_initial_unit_does_not_authorize_new_read():
    fake = vocative_fake(lambda duration: ('يا',))
    fake.responder = lambda f, start, duration: observation(('يا',), quality=.59, positions=False)
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][2] == 0
    assert not any(e.get('acoustic_anchor') == 'whole_spoken_prefix_incomplete_canonical_word'
                   for e in trace['transitions'])


def test_strong_foreign_continuation_does_not_authorize_new_read():
    fake = vocative_fake(lambda duration: ('يا', 'اولي'))
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][2] == 0
    assert not any(e.get('acoustic_anchor') == 'whole_spoken_prefix_incomplete_canonical_word'
                   for e in trace['transitions'])


def test_full_proof_roundtrip_keeps_all_essential_units():
    fake = vocative_fake(lambda duration: ('يا',) if duration < 2000 else ('يا', 'ايها'))
    _, trace = run(fake, [pause(1000)])
    boundary = Boundary(**json.loads(json.dumps(trace['boundaries'][0])))
    groups, ends = {}, {}
    for word, units in zip(fake.words, fake.rendered):
        groups[word.ayah] = groups.get(word.ayah, ()) + units
        ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
    validate_boundaries([boundary], [1], 40000, trace['acoustic_reads'],
                        groups=groups, word_ends=ends, threshold=.6)
    with pytest.raises(ValueError):
        validate_boundaries([replace(boundary, proof={**boundary.proof, 'phrase': ['يا']})],
                            [1], 40000, trace['acoustic_reads'], groups=groups, word_ends=ends, threshold=.6)


def test_wide_pause_keeps_cue_without_spending_new_read_or_confirming_time():
    fake = vocative_fake(lambda duration: ('يا',))
    fake.responder = lambda f, start, duration: (
        observation(('يا',), positions=False) if start == 2000 else observation((), positions=False))
    rows, trace = run(fake, [pause(1000, width=1200)])
    assert rows[0][2] == 0
    assert any(e['state'] == 'SEARCH_CUE_ONLY' and e['reason'] == 'short_word_pause_time_uncertain'
               for e in trace['transitions'])
    assert len([r for r in trace['acoustic_reads'] if r['reason'].startswith('ayah=1:pause')]) == 1


def test_weak_second_essential_unit_cannot_be_averaged_away():
    fake = vocative_fake(lambda duration: ('يا',))
    fake.responder = lambda f, start, duration: (
        observation(('يا',), positions=False) if duration < 2000
        else replace(observation(('يا', 'ايها'), positions=False), unit_quality=(.99, .59)))
    rows, _ = run(fake, [pause(1000)])
    assert rows[0][2] == 0


def test_single_character_ctc_units_do_not_get_asr_short_word_permission():
    groups, ends = {1: tuple('abc')}, {1: [3]}
    assert _expansion_evidence(observation(('a',)), groups[1], ends[1], .25,
                               groups, 1, ends) is None


def test_failed_new_short_word_reads_preserve_later_independent_sparse_anchor():
    fake = vocative_fake(lambda duration: ('يا',))
    fake.responder = lambda f, start, duration: (
        observation(('يا',), positions=False) if start == 1000
        else observation(('يا', 'اولي'), positions=False) if start == 6000
        else observation((), positions=False))
    rows, trace = run(fake, [pause(1000), pause(6000)], policy=Policy(sparse_anchors=True))
    assert rows[0][2] == 0 and rows[1][2] > 0 and rows[1][1] == 6100
    assert any(e.get('reason') == 'short_word_extension_unresolved_resume_candidates'
               for e in trace['transitions'])


def test_new_short_cue_cannot_erase_a_preexisting_supported_opening_block():
    fake = vocative_fake(lambda duration: ())
    later = fake.rendered[1][0]
    rendered = list(fake.rendered)
    rival = next(i for i, word in enumerate(fake.words) if word.ayah == 2)
    rendered[rival] = ('يا', 'ايها')
    fake.rendered = tuple(rendered)
    def respond(f, start, duration):
        if start == 1000:
            units = ('wrong', 'wrong', later) if duration < 2000 else ('يا',) if duration < 3500 else ()
        elif start == 6000:
            units = ('يا', 'ايها', later)
        else:
            units = ()
        return observation(units, positions=False)
    fake.responder = respond
    rows, trace = run(fake, [pause(1000), pause(6000)])
    assert rows[0][2] == 0
    assert any(e.get('acoustic_anchor') == 'exact_later_word_noisy_opening'
               for e in trace['transitions'])
    assert not any(e.get('reason') == 'short_word_extension_unresolved_resume_candidates'
                   and e.get('candidate') == 0 for e in trace['transitions'])


@pytest.mark.parametrize('stationary,initial_quality,reliable', [
    (False, .5, False), (True, .5, True), (False, .99, True),
])
def test_sparse_identity_can_advance_while_unsupported_time_requires_review(
        stationary, initial_quality, reliable):
    from unittest.mock import patch
    fake = Fake(lambda *args: observation(()))
    fake.timing = False
    def respond(f, start, duration):
        if start == 1000:
            return observation(f.phrase(2)[:1], quality=initial_quality if duration < 2000 else .99,
                               positions=False)
        if start == 6000:
            return observation(f.phrase(3)[:1], positions=False)
        return observation((), positions=False)
    fake.responder = respond
    geometry = {'accepted': True, 'reason': 'measured_dsp_interval', 'quiet_stationarity': stationary}
    with patch('surah_cutter.core.fast_executor.pause_geometry', return_value=geometry):
        rows, trace = run(fake, [pause(1000), pause(6000)], policy=Policy(sparse_anchors=True))
    assert bool(rows[1][2]) is reliable
    assert rows[2][2] > 0  # The heard identity still enables the following ayah.
    if not reliable:
        assert trace['boundaries'][1]['review_reason'] == 'weak_complete_sparse_prefix_unproved_time'
        assert any(e['state'] == 'IDENTITY_ONLY' and e['ayah'] == 2 for e in trace['transitions'])


def test_weak_later_word_does_not_remove_a_strong_initial_localization():
    from unittest.mock import patch
    fake = Fake(lambda *args: observation(()))
    fake.timing = False
    rendered = list(fake.rendered)
    a2 = [i for i, w in enumerate(fake.words) if w.ayah == 2]
    a4 = [i for i, w in enumerate(fake.words) if w.ayah == 4]
    rendered[a2[0]], rendered[a2[1]] = ('future',), ('unique',)
    rendered[a4[0]], rendered[a4[1]] = ('future',), ('other',)
    fake.rendered = tuple(rendered)
    def respond(f, start, duration):
        if start != 1000:
            return observation((), positions=False)
        obs = observation(('future', 'unique'), positions=False)
        return replace(obs, unit_quality=(.99, .5)) if duration < 2000 else obs
    fake.responder = respond
    geometry = {'accepted': True, 'reason': 'measured_dsp_interval', 'quiet_stationarity': False}
    with patch('surah_cutter.core.fast_executor.pause_geometry', return_value=geometry):
        rows, _ = run(fake, [pause(1000)], policy=Policy(sparse_anchors=True))
    assert rows[1][2] > 0
