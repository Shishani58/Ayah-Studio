import threading
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import full_boundary, full_transition_verifier, model_store
from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope


def fixture(monkeypatch):
    texts = ["zz", "bc", "df", "gh", "jk", "lm", "nn"]
    words = [SimpleNamespace(surah=1, ayah=1 if i < 3 else 2, index=i, text=t)
             for i, t in enumerate(texts)]
    vocab = {c: i + 1 for i, c in enumerate(dict.fromkeys("".join(texts)))}
    targets = [vocab[c] for text in texts for c in text]
    selected = targets[2:10]  # last two words of 1 + first two words of 2
    matrix = np.full((250, len(vocab) + 1), -12., np.float32)
    matrix[:, 0] = 0
    for t, f in zip(selected, [55, 60, 100, 105, 150, 155, 200, 205]):
        matrix[f, 0], matrix[f, t] = -12, 0
    calls = []

    def build_targets(actual_words, intro=False):
        # Partial phonetization would introduce a false wasl/waqf. Assert the
        # full canonical context, though only four words enter the proof.
        assert [w.index for w in actual_words] == list(range(7))
        return targets, [(i * 2, i * 2 + 2) for i in range(7)]

    def emit(pcm):
        calls.append(len(pcm))
        return matrix

    bundle = SimpleNamespace(blank_id=0, lag_ms=0, emitter=SimpleNamespace(
        build_targets=build_targets, emissions=emit, runtime_identity={"test": "pinned"}))
    monkeypatch.setattr(full_transition_verifier, "_installed_verifier", lambda: (bundle, "muaalem@test"))
    return words, np.zeros(112000, np.float32), [], {}, calls, matrix, selected, bundle


def run(data):
    words, pcm, reads, trace, *_ = data
    return full_transition_verifier.verify_transition(pcm, 16000, words, 2, 1000, 6000, {}, (3300, 3800),
                                                      reads=reads, trace=trace)


def test_exact_phrase_uses_full_canonical_context_before_slicing(monkeypatch):
    data = fixture(monkeypatch)
    proof = run(data)
    assert proof["tail_ms"] == 3120
    assert proof["first_ms"] == 4000
    assert data[2][0]["proof_supported"]
    assert data[2][0]["target_context"] == "complete-canonical-ayahs"
    assert data[2][0]["targets"] == data[6]
    assert not data[2][0]["accepted"]  # caller must also prove a safe DSP cut
    assert data[3]["full_boundary_verifier_source"] == "muaalem@test"
    assert data[4] == [80000]


def test_missing_verifier_does_not_download_or_record_a_model_read(monkeypatch):
    data = fixture(monkeypatch)
    monkeypatch.setattr(full_transition_verifier, "_installed_verifier", lambda: None)
    assert run(data) is None
    assert data[3]["full_transition_verifier"] == "unavailable"
    assert not data[2] and not data[4]


def test_installed_files_without_runtime_decline_without_provisioning(monkeypatch):
    monkeypatch.setattr(model_store, "is_installed", lambda _: True)

    def unavailable():
        raise ValueError("Для проверки спорной границы FULL установите Muaalem во вкладке «Модели».")

    monkeypatch.setattr(full_boundary, "_muaalem_verifier", unavailable)
    assert full_transition_verifier._installed_verifier() is None


def test_unexpected_verifier_loading_error_is_not_hidden(monkeypatch):
    monkeypatch.setattr(model_store, "is_installed", lambda _: True)

    def fail():
        raise ValueError("corrupted model")

    monkeypatch.setattr(full_boundary, "_muaalem_verifier", fail)
    with pytest.raises(ValueError, match="corrupted model"):
        full_transition_verifier._installed_verifier()


@pytest.mark.parametrize("case", ["mismatch", "repeat", "weak-token"])
def test_verifier_rejects_wrong_ambiguous_or_weak_phrase(case, monkeypatch):
    data = fixture(monkeypatch)
    matrix = data[5]
    if case == "mismatch":
        matrix[55] = -12
        matrix[55, 0] = 0
    elif case == "repeat":
        matrix[:] = -12
        matrix[:, 0] = 0
        for base in (10, 130):
            for i, t in enumerate(data[6]):
                matrix[base + i * 5, 0], matrix[base + i * 5, t] = -12, 0
    else:
        matrix[55, data[6][0]], matrix[55, 0] = np.log(.4), np.log(.3)
    assert run(data) is None
    assert not data[2][0]["accepted"]
    assert data[2][0]["reason"] == ("weak-verifier-phrase" if case == "weak-token" else "verifier-phrase-mismatch")


@pytest.mark.parametrize("cancel", [False, True])
def test_error_and_cancellation_are_recorded(cancel, monkeypatch):
    data = fixture(monkeypatch)
    event = threading.Event()

    def emit(_):
        if cancel:
            event.set()
            return data[5]
        raise RuntimeError("test failure")

    data[7].emitter.emissions = emit
    with cancellation_scope(event), pytest.raises(GenerationCancelled if cancel else RuntimeError):
        run(data)
    assert data[2][0]["outcome"] == ("cancelled" if cancel else "failed")
    assert not data[2][0]["accepted"]


def test_partial_target_match_does_not_count_as_complete_neighbour_proof(monkeypatch):
    data = fixture(monkeypatch)
    # The final target phoneme is missing despite a matching long prefix.
    data[5][205] = -12
    data[5][205, 0] = 0
    assert run(data) is None
    assert data[2][0]["reason"] == "verifier-phrase-mismatch"


def test_oversized_crop_is_rejected_before_loading_a_model(monkeypatch):
    data = fixture(monkeypatch)

    def forbidden():
        pytest.fail("oversized crop loaded a model")

    monkeypatch.setattr(full_transition_verifier, "_installed_verifier", forbidden)
    assert full_transition_verifier.verify_transition(
        np.zeros(224000, np.float32), 16000, data[0], 2, 0, 14000, {}, (3300, 3800),
        reads=data[2], trace=data[3]) is None
    assert not data[2] and not data[4]


def test_repeated_adjacent_phrases_cannot_prove_an_owner(monkeypatch):
    data = fixture(monkeypatch)
    duplicate = [SimpleNamespace(surah=1, ayah=3 if i < 2 else 4, index=7 + i, text=t)
                 for i, t in enumerate(["bc", "df", "gh", "jk"])]
    data[0].extend(duplicate)
    assert run(data) is None
    assert not data[2] and not data[4]
