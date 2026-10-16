"""Isolated short-prototype contract checks; run only in concluding M7 phase."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

import surah_cutter.core.fast_executor as executor
from surah_cutter.core.fast_executor import Boundary, _short_context_cue, validate_boundaries
from test_fast_executor import Fake, observation, pause, run


def test_frozen_prototype_import():
    p = Path(__file__).parent
    root = Path(json.loads((p / 'snapshot-paths.json').read_text())['prototype']).resolve()
    assert Path(executor.__file__).resolve().is_relative_to(root)


def test_short_remainder_is_cue_only():
    groups, ends = {1: tuple('qulhuwa'), 2: tuple('other')}, {1: [3, 7], 2: [5]}
    assert _short_context_cue(observation(tuple('ulbad')), tuple('qul'), groups, ends, 1, .25)
    assert not _short_context_cue(observation(tuple('ulbad'), quality=.2), tuple('qul'), groups, ends, 1, .25)
    assert not _short_context_cue(replace(observation(tuple('ulbad')), truncated=True), tuple('qul'), groups, ends, 1, .25)
    assert not _short_context_cue(observation(tuple('ql')), tuple('qul'), groups, ends, 1, .25)


@pytest.mark.parametrize('rival,ends', [('xul', [3]), ('ul', [2]), ('abcdxul', [4, 7])])
def test_symmetric_wrong_owner_and_internal_rival(rival, ends):
    assert not _short_context_cue(observation(tuple('ulbad')), tuple('qul'),
        {1: tuple('qulhuwa'), 2: tuple(rival)}, {1: [3, 7], 2: ends}, 1, .25)


def short_fake(short_units, quality=.99):
    fake = Fake(lambda *a: observation(()))
    fake.metric = 'ctc_greedy_probability'
    rendered = list(fake.rendered)
    rendered[0] = tuple('qul')
    fake.rendered = tuple(rendered)
    fake.responder = lambda f, start, duration: (
        observation(tuple(short_units), quality=quality) if start == 1000 and duration < 1000
        else observation(tuple('ulbad')) if start == 1000 else observation(())
    )
    return fake


def test_full_new_proof_required_and_roundtrip():
    fake = short_fake('qul')
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][2] > 0 and rows[0][1] == 1100
    assert len([e for e in trace['transitions'] if e['state'] == 'SHORT_CONTEXT_RECHECK']) == 1
    b = Boundary(**json.loads(json.dumps(trace['boundaries'][0])))
    groups, ends = {}, {}
    for w, units in zip(fake.words, fake.rendered):
        groups[w.ayah] = groups.get(w.ayah, ()) + units
        ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
    validate_boundaries([b], [1], 40000, trace['acoustic_reads'], groups=groups, word_ends=ends, threshold=.25)
    with pytest.raises(ValueError):
        validate_boundaries([replace(b, confidence=1.1)], [1], 40000,
                            trace['acoustic_reads'], groups=groups, word_ends=ends, threshold=.25)


@pytest.mark.parametrize('units,quality', [('ul', .99), ('wrong', .99), ('qul', .2), ('', .99)])
def test_failed_short_keeps_review_and_original_observation(units, quality):
    fake = short_fake(units, quality)
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][2] == 0
    originals = [r for r in trace['acoustic_reads'] if r['reason'] == 'ayah=1:pause:minimal_phrase']
    assert tuple(originals[0]['observed']['units']) == tuple('ulbad')
