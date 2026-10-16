"""Byte/mark ownership for a search cue must never replace whole-word proof quality."""
import math

import numpy as np
import pytest

from surah_cutter.core.fast_adapters import Observation, attributed_asr_units
from surah_cutter.core.fast_executor import _asr_head_cue_quality, _partial_asr_head
from regression_draft import clipped_fake
from test_fast_executor import observation, pause, run


def result(parts, scores):
    payload = [p.encode() if isinstance(p, str) else p for p in parts]
    return {'text': b''.join(payload).decode(), 'selected_tokens': [
        {'id': i, 'bytes': list(part), 'log_probability': math.log(q)}
        for i, (part, q) in enumerate(zip(payload, scores))]}


def decoded(parts, scores):
    units, q, owners = attributed_asr_units(result(parts, scores))
    return Observation(units, None, min(q), unit_quality=q, attribution=owners)


def test_weak_word_ending_only_authorizes_more_evidence_from_strong_prefix():
    obs = decoded(['ا', 'ا', 'ن', 'ت'], [.99, .98, .97, .2])
    assert obs.unit_quality == (.2,)
    np.testing.assert_allclose(obs.attribution[0]['letter_quality'], [.99, .98, .97, .2])
    assert _asr_head_cue_quality(obs)['quality'] == .97
    assert _partial_asr_head(obs, ('اانتم',), .6)


def test_split_utf8_and_combining_mark_scores_cannot_be_discarded():
    raw = 'اانت'.encode()
    split = decoded([raw[:1], raw[1:]], [.1, .99])
    assert split.attribution[0]['letter_quality'][0] == pytest.approx(.1)
    assert not _partial_asr_head(split, ('اانتم',), .6)
    initial_mark = decoded(['ا', 'َ', 'انت'], [.99, .1, .99])
    assert initial_mark.attribution[0]['letter_quality'][0] == pytest.approx(.1)
    assert not _partial_asr_head(initial_mark, ('اانتم',), .6)
    tail_mark = decoded(['اان', 'ت', 'َ'], [.99, .99, .1])
    assert tail_mark.unit_quality == pytest.approx((.1,))
    assert _partial_asr_head(tail_mark, ('اانتم',), .6)


def test_cross_word_token_remains_shared_without_promoting_weak_letters():
    obs = decoded(['اولى عل', 'ى'], [.9, .1])
    assert obs.units == ('اولى', 'على') and obs.unit_quality == pytest.approx((.9, .1))
    np.testing.assert_allclose(obs.attribution[0]['letter_quality'], [.9]*4)
    np.testing.assert_allclose(obs.attribution[1]['letter_quality'], [.9, .9, .1])
    assert obs.attribution[0]['selected_indices'] == [0]
    assert obs.attribution[1]['selected_indices'] == [0, 1]


def test_ligature_normalization_preserves_shared_byte_quality():
    obs = decoded(['ﻻ'], [.8])
    assert obs.units == ('لا',) and obs.unit_quality == (.8,)
    np.testing.assert_allclose(obs.attribution[0]['letter_quality'], [.8, .8])


def test_strong_letter_cue_cannot_supply_quality_to_the_final_whole_word():
    fake = clipped_fake('wrong')
    cue = decoded(['ا', 'و', 'ل', 'ى'], [.99, .99, .99, .1])
    weak_complete = decoded(['ا', 'و', 'ل', 'ئ', 'ك'], [.99, .99, .99, .1, .99])
    fake.responder = lambda f, s, d: (
        cue if d < 2000 else weak_complete) if s == 1000 else observation((), positions=False)
    rows, trace = run(fake, [pause(1000)])
    assert all(row[2] == 0 for row in rows)
    assert any(e['state'] == 'SEARCH_CUE_ONLY' for e in trace['transitions'])
    assert len([r for r in trace['acoustic_reads'] if r['window_ms'][0] == 1000]) >= 2
