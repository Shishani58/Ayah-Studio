from types import SimpleNamespace
import threading

import numpy as np
import pytest

from surah_cutter.core import full_boundary, model_runtime, model_store
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope
from surah_cutter.data import quran_text
from surah_cutter.data.alignment_result import result_source
from surah_cutter.core.candidates import Pause


def fixture(monkeypatch):
    current = [SimpleNamespace(surah=98, ayah=7, index=i + 3, text=str(i)) for i in range(3)]
    rivals = [SimpleNamespace(surah=98, ayah=6, index=i, text=str(i)) for i in range(3)]
    matrix = np.full((100, 11), -12, np.float32)
    matrix[:, 0] = -.1
    for i, token in enumerate(range(1, 7)):
        matrix[20 + i * 6, 0] = -12
        matrix[20 + i * 6, token] = -.1

    left = matrix.copy()
    left[:] = -12
    left[:, 0] = -.1
    for i, token in enumerate([7, 8, 9, 10]):
        left[20 + i * 6, 0] = -12
        left[20 + i * 6, token] = -.1
    calls = []

    def emissions(pcm):
        calls.append(len(pcm))
        return matrix if len(calls) == 1 else left

    def targets(words, intro=False):
        if len(words) == 2:
            return ([7, 8, 9, 10] if words[0].ayah == 6 else [3, 4, 5, 6], [(0, 2), (2, 4)])
        return ([1, 2, 3, 4, 5, 6] if words[0].ayah == 7 else [1, 2, 3, 4, 9, 10],
                [(0, 2), (2, 4), (4, 6)])

    emitter = SimpleNamespace(emissions=emissions, build_targets=targets,
                              runtime_identity={'model': 'muaalem', 'weights_sha256': 'test'})
    bundle = SimpleNamespace(emitter=emitter, vocab={}, blank_id=0, delim_id=None, lag_ms=0)
    monkeypatch.setattr(full_boundary, '_muaalem_verifier', lambda: (bundle, 'muaalem@fixed'))
    monkeypatch.setattr(quran_text, 'surah_words', lambda _: rivals + current)
    check = {'ayah': 7, 'onset_ms': 800, 'raw_onset_ms': 800, 'conf': .01,
             'first_token_conf': .001, 'reliable': False, 'repaired': False}
    return current, bundle, matrix, check, [], {}


def run(data):
    words, _, _, check, reads, trace = data
    full_boundary._recover_long_split(np.zeros(112000, np.float32), 16000, words,
                                     Pause(1000, 5000, 4000), 4800, 5900,
                                     {6: {'onset_ms': 600}, 7: check, 8: {'onset_ms': 7000}},
                                     check, reads, trace)


def test_auxiliary_proof_disambiguates_shared_words_and_uses_safe_dsp_cut(monkeypatch):
    data = fixture(monkeypatch)
    selected = model_store.ACTIVE_ID
    run(data)
    check, reads, trace = data[3:]
    assert check['onset_ms'] == 4900
    assert check['raw_onset_ms'] == 800
    assert check['reason'] == 'split-word-muaalem-proof'
    assert not check['reliable']  # Text probabilities are not calibrated timing confidence.
    assert reads[0]['required_tokens'] == 5  # First two words alone are ambiguous.
    assert reads[0]['accepted'] and reads[0]['outcome'] == 'completed'
    assert reads[1]['proof_supported']
    assert reads[1]['role'] == 'closing-proof'
    assert trace['full_boundary_verifier_source'] == 'muaalem@fixed'
    assert model_store.ACTIVE_ID == selected
    assert 'full_boundary_verifier=muaalem@fixed' in result_source(trace, 'full')


def test_matching_shared_opening_without_distinguishing_phoneme_is_rejected(monkeypatch):
    data = fixture(monkeypatch)
    data[2][44, 5] = -12
    data[2][44, 0] = -.1
    run(data)
    assert data[3]['onset_ms'] == 800
    assert data[4][0]['reason'] == 'verifier-greedy-mismatch'


def test_identical_canonical_openings_cannot_prove_the_ayah_owner(monkeypatch):
    data = fixture(monkeypatch)
    data[1].emitter.build_targets = lambda words, intro=False: ([1, 2, 3, 4, 5, 6], [(0, 2), (2, 4), (4, 6)])
    run(data)
    assert data[3]['onset_ms'] == 800
    assert data[4][0]['reason'] == 'indistinguishable-canonical-opening'


def test_uninstalled_verifier_never_downloads_or_changes_selected_model(monkeypatch, tmp_path):
    monkeypatch.setattr(model_runtime, 'environment', lambda _: tmp_path)
    monkeypatch.setattr(model_store, 'is_installed', lambda _: False)

    def forbidden(*args, **kwargs):
        pytest.fail('Missing verifier must not provision or load a runtime')

    monkeypatch.setattr(model_runtime, 'load_runtime', forbidden)
    selected = model_store.ACTIVE_ID
    with pytest.raises(ValueError, match='установите Muaalem'):
        full_boundary._muaalem_verifier()
    assert model_store.ACTIVE_ID == selected


def test_cancelled_auxiliary_read_is_recorded_without_changing_boundary(monkeypatch):
    data = fixture(monkeypatch)
    event = threading.Event()

    def cancel(_):
        event.set()
        return data[2]

    data[1].emitter.emissions = cancel
    with cancellation_scope(event), pytest.raises(GenerationCancelled):
        run(data)
    assert data[3]['onset_ms'] == 800
    assert data[4][0]['outcome'] == 'cancelled'


def test_wrong_previous_tail_prevents_boundary_move(monkeypatch):
    data = fixture(monkeypatch)
    data[1].emitter.emissions = lambda _: data[2]
    run(data)
    assert data[3]['onset_ms'] == 800
    assert data[4][0]['reason'] == 'previous-tail-greedy-mismatch'


def test_failed_auxiliary_inference_is_recorded(monkeypatch):
    data = fixture(monkeypatch)

    def fail(_):
        raise RuntimeError('inference failed')

    data[1].emitter.emissions = fail
    with pytest.raises(RuntimeError, match='inference failed'):
        run(data)
    assert data[3]['onset_ms'] == 800
    assert data[4][0]['outcome'] == 'failed'
