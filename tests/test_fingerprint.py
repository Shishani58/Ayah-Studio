from __future__ import annotations

import numpy as np

from surah_cutter.core import decode, fingerprint


def test_partial_fingerprint_refine_requires_local_same_recording_match(tmp_path, monkeypatch):
    sr = 16_000
    hop = sr // 20
    amplitudes = np.linspace(0.02, 1.0, 160, dtype=np.float32)
    amplitudes[50:70] = np.asarray(
        [0.11, 0.92, 0.34, 0.78, 0.25, 0.66, 0.43, 0.88, 0.19, 0.71,
         0.29, 0.97, 0.52, 0.16, 0.83, 0.38, 0.61, 0.24, 0.95, 0.47],
        dtype=np.float32,
    )
    full = np.repeat(amplitudes, hop)
    clip = full[50 * hop:70 * hop]
    clip_path = tmp_path / "026002.mp3"
    clip_path.touch()
    monkeypatch.setattr(decode, "decode", lambda path: clip if str(path) == str(clip_path) else full)

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 1_000, 0.9), (2, 4_000, 0.2), (3, 6_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=2_000,
    )

    assert rows[1][1] == 2_500
    assert rows[1][2] > 0.99
    assert repairs[0]["ayah"] == 2
    assert repairs[0]["fast_ms"] == 4_000
    assert repairs[0]["matched_ms"] == 2_500
    assert repairs[0]["correlation"] > 0.99
    assert repairs[0]["search_start_ms"] == 2_000
    assert repairs[0]["search_end_ms"] == 7_000
    assert repairs[0]["clip_frames"] == 20


def test_partial_fingerprint_refine_rejects_out_of_order_match(tmp_path, monkeypatch):
    sr = 16_000
    hop = sr // 20
    rng = np.random.default_rng(7)
    full = np.repeat((0.01 + rng.random(120)).astype(np.float32), hop)
    clip = full[10 * hop:30 * hop]
    clip_path = tmp_path / "026002.mp3"
    clip_path.touch()
    monkeypatch.setattr(decode, "decode", lambda _path: clip)

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 1_500, 0.9), (2, 4_000, 0.2), (3, 6_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=4_000,
    )

    assert rows[1] == (2, 4_000, 0.2)
    assert repairs == []


def test_partial_fingerprint_refine_rejects_distant_local_match(tmp_path, monkeypatch):
    sr = 16_000
    hop = sr // 20
    rng = np.random.default_rng(11)
    full = np.repeat((0.01 + rng.random(180)).astype(np.float32), hop)
    clip = full[70 * hop:90 * hop]
    clip_path = tmp_path / "026002.mp3"
    clip_path.touch()
    monkeypatch.setattr(decode, "decode", lambda _path: clip)

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 500, 0.9), (2, 1_000, 0.2), (3, 6_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=4_000,
    )

    assert rows[1] == (2, 1_000, 0.2)
    assert repairs == []


def test_partial_fingerprint_refine_accepts_agreed_energy_signals(tmp_path, monkeypatch):
    sr = 16_000
    full = np.linspace(0.01, 1.0, sr * 12, dtype=np.float32)
    clip = full[sr * 2:sr * 3]
    clip_path = tmp_path / "026002.mp3"
    clip_path.touch()
    monkeypatch.setattr(decode, "decode", lambda _path: clip)
    matches = iter(((50, 0.75), (50, 0.76), (50, 0.90)))
    monkeypatch.setattr(
        fingerprint,
        "_best_normalized_match",
        lambda *_args: next(matches),
    )

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 1_000, 0.9), (2, 2_400, 0.2), (3, 5_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
    )

    assert rows[1][1] == 2_500
    assert repairs[0]["match_method"] == "energy_consensus"
    assert repairs[0]["log_correlation"] == 0.76
    assert repairs[0]["delta_correlation"] == 0.90


def test_fingerprint_rejects_isolated_match_from_complete_mismatched_folder(
    tmp_path,
    monkeypatch,
):
    sr = 16_000
    hop = sr // 20
    rng = np.random.default_rng(23)
    full = np.repeat((0.01 + rng.random(160)).astype(np.float32), hop)
    for ayah in range(1, 7):
        (tmp_path / f"026{ayah:03d}.mp3").touch()

    def fake_decode(path):
        if str(path).endswith("026001.mp3"):
            return full[50 * hop:70 * hop]
        return np.zeros(3 * hop, dtype=np.float32)

    monkeypatch.setattr(decode, "decode", fake_decode)
    original = [(1, 2_000, 0.9), *[
        (ayah, 3_500 + (ayah - 2) * 1_000, 0.2)
        for ayah in range(2, 7)
    ]]

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        original,
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=2_000,
    )

    assert rows == original
    assert repairs == []


def test_partial_fingerprint_can_make_strong_backward_local_rollback(tmp_path, monkeypatch):
    sr = 16_000
    hop = sr // 20
    rng = np.random.default_rng(29)
    full = np.repeat((0.01 + rng.random(180)).astype(np.float32), hop)
    clip = full[70 * hop:90 * hop]
    clip_path = tmp_path / "026002.mp3"
    clip_path.touch()
    monkeypatch.setattr(decode, "decode", lambda _path: clip)

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 1_000, 0.9), (2, 6_000, 0.2), (3, 8_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=4_000,
    )

    assert rows[1][1] == 3_500
    assert repairs[0]["match_method"] == "distant_rollback"


def test_partial_fingerprint_can_make_short_strong_forward_correction(tmp_path, monkeypatch):
    sr = 16_000
    hop = sr // 20
    rng = np.random.default_rng(31)
    full = np.repeat((0.01 + rng.random(180)).astype(np.float32), hop)
    clip = full[66 * hop:86 * hop]
    clip_path = tmp_path / "026002.mp3"
    clip_path.touch()
    monkeypatch.setattr(decode, "decode", lambda _path: clip)

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 500, 0.9), (2, 1_000, 0.2), (3, 5_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=4_000,
    )

    assert rows[1][1] == 3_300
    assert repairs[0]["match_method"] == "strong_forward"


def test_two_agreeing_distant_forward_clips_move_only_their_local_block(
    tmp_path,
    monkeypatch,
):
    sr = 16_000
    hop = sr // 20
    rng = np.random.default_rng(37)
    full = np.repeat((0.01 + rng.random(500)).astype(np.float32), hop)
    clips = {
        "026002.mp3": full[150 * hop:170 * hop],
        "026003.mp3": full[190 * hop:210 * hop],
    }
    for name in clips:
        (tmp_path / name).touch()
    monkeypatch.setattr(decode, "decode", lambda path: clips[str(path).split("/")[-1]])

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 500, 0.9), (2, 2_000, 0.2), (3, 2_500, 0.2), (4, 11_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=10_000,
    )

    assert [row[1] for row in rows] == [500, 7_500, 9_500, 11_000]
    assert [repair["match_method"] for repair in repairs] == [
        "paired_strong_forward",
        "paired_strong_forward",
    ]


def test_strong_clip_can_move_forward_only_inside_its_current_pause_owner(
    tmp_path,
    monkeypatch,
):
    sr = 16_000
    hop = sr // 20
    rng = np.random.default_rng(41)
    full = np.repeat((0.01 + rng.random(400)).astype(np.float32), hop)
    clip = full[150 * hop:170 * hop]
    clip_path = tmp_path / "026002.mp3"
    clip_path.touch()
    monkeypatch.setattr(decode, "decode", lambda _path: clip)

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 500, 0.9), (2, 2_000, 0.2), (3, 11_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=10_000,
        owner_intervals={2: (7_000, 8_000)},
    )

    assert rows[1][1] == 7_500
    assert repairs[0]["match_method"] == "owner_bound_forward"


def test_energy_consensus_can_move_forward_only_inside_its_current_owner(
    tmp_path,
    monkeypatch,
):
    sr = 16_000
    hop = sr // 20
    rng = np.random.default_rng(43)
    full = np.repeat((0.01 + rng.random(400)).astype(np.float32), hop)
    clip_path = tmp_path / "026002.mp3"
    clip_path.touch()
    monkeypatch.setattr(decode, "decode", lambda _path: full[150 * hop:170 * hop])
    matches = iter(((150, 0.65), (150, 0.79), (150, 0.91)))
    monkeypatch.setattr(
        fingerprint,
        "_best_normalized_match",
        lambda *_args: next(matches),
    )

    rows, repairs = fingerprint.refine_fast_rows_by_fingerprint(
        full,
        sr,
        [(1, 500, 0.9), (2, 2_000, 0.2), (3, 11_000, 0.9)],
        26,
        ayah_audio_dir=tmp_path,
        search_radius_ms=10_000,
        owner_intervals={2: (7_000, 8_000)},
    )

    assert rows[1][1] == 7_500
    assert repairs[0]["match_method"] == "owner_bound_energy_consensus"
