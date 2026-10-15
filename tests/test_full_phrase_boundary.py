from types import SimpleNamespace
import threading

import numpy as np
import pytest

from surah_cutter.core import align, candidates, full_boundary
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope
from surah_cutter.data import transliteration


def fixture(monkeypatch):
    trs = ['al-ḥamdu', 'lillahi']
    vocab = {c: i + 1 for i, c in enumerate(dict.fromkeys(''.join(trs) + '|'))}
    words = [SimpleNamespace(surah=1, ayah=2, index=i) for i in range(2)]
    bundle = SimpleNamespace(vocab=vocab, blank_id=0, delim_id=vocab['|'], lag_ms=200)
    targets, word_spans = align._build_targets(words, trs, bundle)
    matrix = np.full((90, len(vocab) + 1), -12, np.float32)
    matrix[:, 0] = -.1
    for i, token in enumerate(targets):
        matrix[14 + i * 3, 0] = -12
        matrix[14 + i * 3, token] = -.1
    calls = []

    def emit(pcm):
        calls.append(len(pcm))
        return matrix

    bundle.emitter = SimpleNamespace(emissions=emit)
    # A strong vowel before a pause; all other first-word letters on the right.
    spans = [(10, 10)] + [(55 + i * 2, 55 + i * 2) for i in range(len(targets) - 1)]
    scores = np.full(100, -.1, np.float32)
    check = {'ayah': 2, 'raw_onset_ms': 0, 'onset_ms': 0, 'conf': .9,
             'first_token_conf': .9, 'token_confs': [.9] * len(trs[0]),
             'reliable': True, 'repaired': False, 'reason': 'ctc'}
    checks = {2: check}
    trace = {}
    pcm = np.zeros(48_000, np.float32)
    monkeypatch.setattr(transliteration, 'surah_word_tr', lambda _: trs)
    monkeypatch.setattr(candidates, '_dsp_accepts_pause', lambda _: True)
    monkeypatch.setattr(candidates, '_dsp_pause_metrics', lambda *_: {})
    return bundle, pcm, words, word_spans, spans, scores, checks, trace, calls


def run(data, pauses=None):
    b, pcm, words, ws, spans, scores, checks, trace, _ = data
    full_boundary.recheck_split_words(b, pcm, 16000, words, ws, spans, scores, 20,
                                    checks, pauses or [candidates.Pause(500, 1000, 500)], trace=trace)


def test_strong_vowel_does_not_hide_split_word_when_whole_phrase_is_confirmed(monkeypatch):
    data = fixture(monkeypatch)
    run(data)
    check = data[6][2]
    assert check['repaired']
    assert 750 <= check['onset_ms'] <= 1200
    assert check['reason'] == 'split-word-local-recheck'
    assert check['original_first_token_conf'] == .9
    read = data[7]['full_boundary_reads'][0]
    assert read['outcome'] == 'completed' and read['accepted']
    assert read['end_ms'] - read['start_ms'] <= 6000


def test_strong_consonant_before_pause_cannot_be_discarded(monkeypatch):
    data = fixture(monkeypatch)
    data[4][1] = (15, 15)  # l belongs to the left, supported by scores.
    run(data)
    assert not data[6][2]['repaired']
    assert not data[-1]


def test_greedy_phrase_disagreement_prevents_forced_alignment_repair(monkeypatch):
    data = fixture(monkeypatch)
    b = data[0]
    matrix = np.zeros((90, len(b.vocab) + 1), np.float32)
    b.emitter.emissions = lambda _: matrix  # independent decode is empty.
    run(data)
    assert not data[6][2]['repaired']
    assert data[7]['full_boundary_reads'][0]['reason'] == 'greedy-opening-mismatch'


def test_ambiguous_pauses_do_not_trigger_a_model_scan(monkeypatch):
    data = fixture(monkeypatch)
    run(data, [candidates.Pause(300, 600, 300), candidates.Pause(700, 1000, 300)])
    assert not data[6][2]['repaired']
    assert not data[-1]


def test_matching_consonants_with_wrong_internal_vowel_is_not_phrase_evidence(monkeypatch):
    data = fixture(monkeypatch)
    b = data[0]
    matrix = b.emitter.emissions(data[1]).copy()
    # The a after ḥ is required; consonant-only matching would hide its failure.
    matrix[14 + 4 * 3, b.vocab['a']] = -3
    matrix[14 + 4 * 3, 0] = -.1
    b.emitter.emissions = lambda _: matrix
    run(data)
    assert not data[6][2]['repaired']
    read = data[7]['full_boundary_reads'][0]
    assert read['reason'] == 'insufficient-local-word-evidence'
    assert read['local_check']['conf'] >= .25


def test_boundary_candidate_cannot_cross_next_ayah(monkeypatch):
    data = fixture(monkeypatch)
    data[6][3] = {'onset_ms': 500}
    run(data)
    assert not data[6][2]['repaired']
    assert data[7]['full_boundary_reads'][0]['reason'] == 'conflicting-ayah-order'


def test_failed_inference_is_recorded_and_does_not_write_a_new_boundary(monkeypatch):
    data = fixture(monkeypatch)

    def fail(_):
        raise RuntimeError('inference failed')

    data[0].emitter.emissions = fail
    with pytest.raises(RuntimeError, match='inference failed'):
        run(data)
    assert not data[6][2]['repaired']
    assert data[7]['full_boundary_reads'][0]['outcome'] == 'failed'


def test_cancelled_inference_is_recorded_before_any_result_is_applied(monkeypatch):
    data = fixture(monkeypatch)
    event = threading.Event()
    emitter = data[0].emitter.emissions

    def cancel(pcm):
        result = emitter(pcm)
        event.set()
        return result

    data[0].emitter.emissions = cancel
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        run(data)
    assert not data[6][2]['repaired']
    assert data[7]['full_boundary_reads'][0]['outcome'] == 'cancelled'


def test_cancellation_before_dispatch_does_not_record_a_model_call(monkeypatch):
    data = fixture(monkeypatch)
    event = threading.Event()
    event.set()
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        run(data)
    assert not data[-1]
