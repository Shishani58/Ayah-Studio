"""Confidence combines independent evidence without changing the stored cut."""

import copy

import pytest

from surah_cutter.data.onset_evidence import onset_evidence


def local_trace():
    previous = dict(ayah=1, onset_ms=1000, full_ctc_end_ms=1800,
                    full_ctc_tail_conf=.8)
    current = dict(ayah=2, onset_ms=2100, raw_onset_ms=2100, conf=.01,
                   first_token_conf=.9, token_confs=[.9, .0001],
                   reliable=False, reason='weak-word', full_ctc_first_ms=2300,
                   pause_center_reason='positioned', center_pause_start_ms=1900,
                   center_pause_end_ms=2300, pause_lead_ms=200,
                   pause_local_decisions=[dict(start_ms=1900, end_ms=2300,
                                               accepted=True, owned=True)])
    following = dict(ayah=3, onset_ms=4000)
    return dict(ayahs=[previous, current, following], align_lag_ms=200)


def assess(trace):
    c = trace['ayahs'][1]
    return onset_evidence(c, c['onset_ms'], c['conf'], trace=trace)['boundary']


def test_local_pause_and_both_owners_outweigh_weak_word_average():
    trace = local_trace()
    before = copy.deepcopy(trace)
    assert assess(trace) == dict(level='high', reason='local-ctc-pause')
    assert trace == before


@pytest.mark.parametrize('field,value', [
    ('center_pause_start_ms', 2200), ('center_pause_end_ms', 2500),
    ('pause_lead_ms', 250), ('onset_ms', 2200), ('full_ctc_first_ms', 2800),
    ('first_token_conf', .01), ('full_ctc_first_ms', float('nan')),
])
def test_local_flag_alone_cannot_confirm_conflicting_geometry(field, value):
    trace = local_trace()
    trace['ayahs'][1][field] = value
    assert assess(trace)['level'] != 'high'


@pytest.mark.parametrize('target,field,value', [
    (0, 'full_ctc_end_ms', 2200), (0, 'full_ctc_tail_conf', .1),
    (0, 'onset_ms', 2150), (0, 'ayah', 9), (2, 'onset_ms', 2000),
])
def test_neighbors_must_support_current_cut(target, field, value):
    trace = local_trace()
    trace['ayahs'][target][field] = value
    assert assess(trace)['level'] != 'high'


@pytest.mark.parametrize('change', ['rejected', 'not-owned', 'duplicate', 'missing'])
def test_local_pause_must_be_unique_accepted_and_owned(change):
    trace = local_trace()
    decisions = trace['ayahs'][1]['pause_local_decisions']
    if change == 'rejected':
        decisions[0]['accepted'] = False
    elif change == 'not-owned':
        decisions[0]['owned'] = False
    elif change == 'duplicate':
        decisions.append(dict(decisions[0]))
    else:
        decisions.clear()
    assert assess(trace)['level'] == 'low'


def test_long_pause_is_supported_when_cut_is_at_accepted_lead():
    trace = local_trace()
    current = trace['ayahs'][1]
    current.update(center_pause_end_ms=3500, onset_ms=3250, pause_lead_ms=250,
                   full_ctc_first_ms=3550)
    current['pause_local_decisions'][0]['end_ms'] = 3500
    assert assess(trace)['level'] == 'high'


def test_first_ayah_uses_intro_tail_not_missing_previous_ayah():
    trace = local_trace()
    c = trace['ayahs'][1]
    c.update(ayah=1, full_intro_end_ms=1800, full_intro_tail_conf=.8)
    trace['ayahs'] = [c, dict(ayah=2, onset_ms=4000)]
    assert onset_evidence(c, 2100, .01, trace=trace)['boundary']['level'] == 'high'


def test_weak_first_phoneme_uses_nearby_sounds_and_temporal_context():
    trace = local_trace()
    c = trace['ayahs'][1]
    c.update(reason='weak-onset', pause_center_reason='weak-evidence',
             first_token_conf=.01, token_confs=[.01, .99, .8])
    assert assess(trace) == dict(level='medium', reason='contextual-onset')
    # A wholly unsupported or distant prefix remains uncertain (e.g. 2:220).
    c['token_confs'] = [.0002, 0, .001, .99]
    assert assess(trace)['level'] == 'low'


def test_global_pause_and_adjacent_sounds_support_weak_onset_across_long_breath():
    trace = local_trace()
    c = trace['ayahs'][1]
    c.update(reason='weak-onset', pause_center_reason='weak-evidence',
             first_token_conf=.01, token_confs=[.01, .99])
    trace['ayahs'][0]['full_ctc_end_ms'] = 1100
    pause = dict(accepted=True, start_ms=1300, end_ms=2250)
    trace['full_windows'] = dict(pause_decisions=[pause])
    assert assess(trace) == dict(level='medium', reason='contextual-pause')
    # A pause owned by the previous word or a later word cannot corroborate it.
    for changes in [dict(start_ms=900), dict(end_ms=2800), dict(accepted=False)]:
        trace['full_windows']['pause_decisions'] = [{**pause, **changes}]
        assert assess(trace)['level'] == 'low'
    trace['full_windows']['pause_decisions'] = [pause, dict(pause)]
    assert assess(trace)['level'] == 'low'


@pytest.mark.parametrize('change', ['overlap', 'long-gap', 'wrong-lag', 'moved', 'order'])
def test_strong_suffix_does_not_override_temporal_disagreement(change):
    trace = local_trace()
    c = trace['ayahs'][1]
    c.update(reason='weak-onset', pause_center_reason='weak-evidence',
             first_token_conf=.01, token_confs=[.01, .99])
    if change == 'overlap':
        trace['ayahs'][0]['full_ctc_end_ms'] = 2400
    elif change == 'long-gap':
        trace['ayahs'][0]['full_ctc_end_ms'] = 1200
    elif change == 'wrong-lag':
        trace['align_lag_ms'] = 0
    elif change == 'moved':
        c['onset_ms'] += 1
    else:
        trace['ayahs'][2]['onset_ms'] = 2050
    assert assess(trace)['level'] == 'low'


@pytest.mark.parametrize('reason', ['ambiguous-pause', 'order-conflict', 'unbounded-gap'])
def test_explicit_pause_conflict_cannot_be_overridden_by_strong_word(reason):
    trace = local_trace()
    trace['ayahs'][1].update(pause_center_reason=reason, reliable=True, reason='ctc')
    assert assess(trace) == dict(level='low', reason='boundary-context-conflict')


def test_explicit_recheck_warning_still_wins_over_pause_support():
    trace = local_trace()
    trace['ayahs'][1]['reason'] = 'late-onset-in-pause'
    assert assess(trace) == dict(level='low', reason='late-onset-in-pause')
