from __future__ import annotations

import numpy as np

from surah_cutter.core import candidates, draft


def test_draft_pause_backed_start_uses_pause_center(monkeypatch):
    sr = 16_000
    pcm = np.zeros(sr * 5, dtype=np.float32)
    pause = candidates.Pause(1000.0, 2000.0, 1000.0)

    monkeypatch.setattr(draft, "_pause_marks", lambda _pcm, _sr, _surah: [pause])
    monkeypatch.setattr(
        draft,
        "_expected_ms",
        lambda _surah, _total_ms, _pauses, *, db_path: {1: 0.0, 2: 1500.0, 3: 3200.0},
    )

    result = draft.draft_from_pauses(pcm, sr, 1)

    assert result.rows[1] == (1, 2, 1500)
    assert result.rows[1][2] != 2000
    assert result.pause_backed == 1


def test_intro_basmala_mark_uses_pause_center():
    pause = candidates.Pause(5000.0, 7000.0, 2000.0)

    mark = draft._intro_basmala_end([pause], total_ms=60_000.0, surah=2)

    assert mark == 6000.0
