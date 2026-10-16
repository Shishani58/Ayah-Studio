"""Shared first words must not turn a fully read previous body into a fatal cue."""
from test_fast_executor import Fake, observation, pause, run


def scenario(rival='strong', distinct=False):
    def respond(fake, start, duration):
        ayah = {1000: 1, 6000: 2, 12000: 3, 20000: 4}.get(start)
        if ayah:
            return observation(fake.phrase(ayah))
        if start == 14000:
            if rival == 'partial':
                return observation(('shared',))
            return observation(('shared', 'other'), quality=.5 if rival == 'weak' else .99)
        return observation(())

    fake = Fake(respond)
    rendered = list(fake.rendered)
    # Canonical112:3 contains «ولم يولد»,112:4 begins «ولم يكن».
    # Canonical words are untouched; only the fake acoustic encoding is shared.
    assert fake.words[8].text == fake.words[10].text
    rendered[8], rendered[9] = ('shared',), ('other',)
    rendered[10], rendered[11] = ('different' if distinct else 'shared',), ('wanted',)
    fake.rendered = tuple(rendered)
    return run(fake, [pause(t) for t in [1000, 6000, 12000, 14000, 20000]])


def test_complete_previous_body_is_declined_before_reading_true_opening():
    rows, trace = scenario()
    assert rows[3][1:] == (20100, .99)
    declines = [e for e in trace['transitions'] if e['state'] == 'DECLINE_CANDIDATE']
    assert len(declines) == 1 and declines[0]['rival']['ayah'] == 3
    assert declines[0]['reason'] == 'decoded_previous_ayah_body'
    assert len([r for r in trace['acoustic_reads'] if r['window_ms'][0] == 14000]) == 1


def test_partial_rival_keeps_the_unresolved_opening_protected():
    rows, trace = scenario('partial')
    assert trace["boundaries"][3]["confidence"] == 0
    assert not any(e['state'] == 'DECLINE_CANDIDATE' for e in trace['transitions'])
    assert not any(r['window_ms'][0] == 20000 for r in trace['acoustic_reads'])


def test_weak_rival_cannot_authorize_a_later_replacement():
    rows, trace = scenario('weak')
    assert trace["boundaries"][3]["confidence"] == 0
    assert not any(e['state'] == 'DECLINE_CANDIDATE' for e in trace['transitions'])


def test_distinct_first_word_does_not_use_the_shared_word_rule():
    rows, trace = scenario(distinct=True)
    assert rows[3][1:] == (20100, .99)
    assert not any(e['state'] == 'DECLINE_CANDIDATE' for e in trace['transitions'])
