from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, candidates


def test_local_dsp_recovers_gap_without_promoting_weak_token_to_reliable():
    # Two audible noise bursts with a quiet interval. Global pauses supplied
    # to the checker miss this gap, as on the measured Shatri42 recording.
    rng = np.random.default_rng(11)
    pcm = rng.normal(0, .12, 32_000).astype(np.float32)
    pcm[12_800:19_200] *= .005
    scores = np.full(100, -.1, np.float32)
    scores[30] = -12
    check = align._check_ayah_onsets(
        [SimpleNamespace(ayah=1)], [(0, 2)], [(30, 30), (65, 65)], scores, 20,
        pauses=[], pcm=pcm,
    )[1]
    assert check["repaired"]
    assert 800 < check["onset_ms"] < 1200
    assert check["reason"] == "weak-prefix-across-local-pause"
    assert not check["reliable"]
    assert check["first_token_conf"] < .001


@pytest.mark.parametrize("offset", [0, 5000])
def test_local_pause_uses_absolute_pcm_and_preserves_ambiguity(monkeypatch, offset):
    seen = []

    def detector(pcm, sr):
        seen.append((len(pcm), sr))
        return [SimpleNamespace(accepted=True, pause=p) for p in
                [candidates.Pause(1200, 1500, 300), candidates.Pause(1600, 1900, 300)]]

    monkeypatch.setattr(candidates, "pause_pipeline_decisions", detector)
    scores = np.full(200, -.1, np.float32)
    scores[50] = -12
    check = align._check_ayah_onsets(
        [SimpleNamespace(ayah=1)], [(0, 2)], [(50, 50), (100, 100)], scores, 20,
        pauses=[], pcm=np.zeros(160_000), base_ms=offset,
    )[1]
    assert seen == [(48_000, 16000)]
    assert check["local_pause_crop_ms"] == [offset, offset + 3000]
    assert not check["repaired"]


@pytest.mark.parametrize("pcm,first_score,last_frame,last_score", [
    (None, -12, 65, -.1),  # FAST/local callers do not opt into the FULL check.
    (np.zeros(160_000), -.1, 65, -.1),  # Supported prefix cannot be discarded.
    (np.zeros(160_000), -12, 65, -12),  # No strong suffix of the same word.
    (np.zeros(160_000), -12, 260, -.1),  # Long spans cannot authorize a scan.
])
def test_local_scan_requires_full_pcm_and_bounded_weak_prefix(
    monkeypatch, pcm, first_score, last_frame, last_score,
):
    def forbidden(*args, **kwargs):
        pytest.fail("Local detector must not run without bounded prefix evidence")

    monkeypatch.setattr(candidates, "pause_pipeline_decisions", forbidden)
    scores = np.full(300, -12, np.float32)
    scores[30], scores[last_frame] = first_score, last_score
    check = align._check_ayah_onsets(
        [SimpleNamespace(ayah=1)], [(0, 2)], [(30, 30), (last_frame, last_frame)], scores, 20,
        pauses=[], pcm=pcm,
    )[1]
    assert not check["repaired"]
