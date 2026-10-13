from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, align_tiled
from surah_cutter.core._ctc import forced_align
from surah_cutter.core.candidates import Pause
from surah_cutter.data import transliteration


def test_ctc_backtrace_estimate_routes_al_baqarah_to_tiles():
    samples = 7_000 * align.EXPECTED_SR
    targets = 47_266

    estimated = align._estimated_ctc_backtrace_bytes(
        samples,
        align.EXPECTED_SR,
        targets,
    )

    assert estimated > 240 * 1024**3
    assert align._full_ctc_requires_tiling(
        samples,
        align.EXPECTED_SR,
        targets,
        max_backtrace_bytes=1024**3,
    )
    assert not align._full_ctc_requires_tiling(
        10 * align.EXPECTED_SR,
        align.EXPECTED_SR,
        100,
        max_backtrace_bytes=1024**3,
    )


def test_align_surah_routes_oversized_exact_ctc_before_emissions(monkeypatch):
    emitter = SimpleNamespace(vocab={"a": 1, "|": 2}, blank_id=0)
    bundle = align._OnnxBundle(emitter)
    words = [SimpleNamespace(surah=2, ayah=1)]
    pcm = np.zeros(align.EXPECTED_SR, dtype=np.float32)

    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _surah: ["a"])
    monkeypatch.setattr(align, "_basmala_trs", lambda _transliteration: [])
    monkeypatch.setattr(align, "_full_ctc_requires_tiling", lambda *_args: True)
    monkeypatch.setattr(
        align,
        "_emissions",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unbounded emissions")),
    )

    received: dict[str, object] = {}

    def fake_tiled(_pcm, _sr, _words, *, surah, bundle, check_onsets):
        assert check_onsets
        received.update(surah=surah, bundle=bundle)
        return [(1, 123, 0.9)]

    monkeypatch.setattr(align_tiled, "align_surah_tiled", fake_tiled)

    rows = align.align_surah(
        pcm,
        align.EXPECTED_SR,
        words,
        surah=2,
        model=bundle,
    )

    assert rows == [(1, 123, 0.9)]
    assert received == {"surah": 2, "bundle": bundle}


def onset_check(scores, spans, pauses):
    return align._check_ayah_onsets(
        [SimpleNamespace(ayah=1)], [(0, len(spans))], spans,
        scores, 20.0, pauses=pauses,
    )[1]


def test_weak_prefix_across_real_pause_is_rechecked_without_blank_confidence():
    scores = np.full(100, -0.0001, np.float32)
    scores[10] = -12
    scores[55] = -0.1
    scores[57:60] = -0.1
    check = onset_check(scores, [(10, 10), (55, 55), (57, 59)], [Pause(500, 1000, 500)])

    assert check["onset_ms"] == 750
    assert check["repaired"]
    assert not check["reliable"]
    assert check["conf"] < 0.1  # 44 confident blank frames cannot mask the prefix.
    assert check["first_token_conf"] < 0.001


@pytest.mark.parametrize("pauses", [[], [Pause(400, 600, 200), Pause(700, 1000, 300)]])
def test_weak_onset_without_unique_pause_is_flagged_and_not_moved(pauses):
    scores = np.full(100, -0.1, np.float32)
    scores[10] = -12
    check = onset_check(scores, [(10, 10), (55, 55)], pauses)

    assert check["onset_ms"] == check["raw_onset_ms"]
    assert not check["repaired"]
    assert not check["reliable"]


def test_supported_early_token_is_not_discarded_for_a_later_pause():
    scores = np.full(100, -0.1, np.float32)
    check = onset_check(scores, [(10, 10), (55, 55)], [Pause(500, 1000, 500)])

    assert not check["repaired"]
    assert check["reliable"]


def test_prefix_with_another_strong_token_before_pause_is_not_moved():
    scores = np.full(100, -0.1, np.float32)
    scores[10] = -12
    check = onset_check(
        scores, [(10, 10), (15, 15), (55, 55)], [Pause(500, 1000, 500)]
    )

    assert not check["repaired"]


def test_pause_without_strong_word_suffix_cannot_move_the_onset():
    scores = np.full(100, -12, np.float32)
    check = onset_check(scores, [(10, 10), (55, 55)], [Pause(500, 1000, 500)])

    assert not check["repaired"]
    assert not check["reliable"]


def test_sliced_emissions_use_absolute_pause_times():
    scores = np.full(100, -0.1, np.float32)
    scores[10] = -12
    check = align._check_ayah_onsets(
        [SimpleNamespace(ayah=2)], [(0, 2)], [(10, 10), (55, 55)], scores, 20,
        pauses=[Pause(5500, 6000, 500)], base_ms=5000,
    )[2]

    assert check["raw_onset_ms"] == 5000
    assert check["onset_ms"] == 5750


@pytest.mark.parametrize("frames, targets", [(0, [1]), (1, [1, 1]), (2, [1, 2, 1])])
def test_impossible_full_ctc_path_is_rejected(frames, targets):
    with pytest.raises(ValueError):
        forced_align(np.zeros((1, frames, 3)), np.array([targets]), blank=0)


def test_full_alignment_rejects_incomplete_path_instead_of_padding(monkeypatch):
    from surah_cutter.core import _ctc

    bundle = align._OnnxBundle(SimpleNamespace(vocab={"a": 1, "|": 2}, blank_id=0))
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["aaa"])
    monkeypatch.setattr(align, "_emissions", lambda *_: np.zeros((20, 3), np.float32))
    monkeypatch.setattr(_ctc, "forced_align", lambda *_args, **_kwargs:
                        (np.ones((1, 20), np.int64), np.zeros((1, 20), np.float32)))

    with pytest.raises(ValueError, match="полный текст"):
        align.align_surah(
            np.zeros(16_000, np.float32), 16_000,
            [SimpleNamespace(surah=1, ayah=1)], model=bundle,
        )
