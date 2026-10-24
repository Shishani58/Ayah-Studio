"""Record fresh text-only FULL checks before/after signal refinement."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from surah_cutter.core import full_signal

TASK = Path(__file__).resolve().parent


def main():
    original = full_signal.refine_late_boundaries
    captured = []

    def refine(checks, pcm, sr, *, trace=None):
        before = deepcopy(checks)
        original(checks, pcm, sr, trace=trace)
        surah = {7: 1, 4: 112, 6: 114, 42: 80}[len(checks)]
        # Direct align_surah fills final trace metadata after this callback.
        # Verify the actual acoustic-frame/raw-onset relation instead.
        assert all(c['raw_onset_ms'] == max(0, round(c['full_ctc_first_ms'])) for c in before.values())
        captured.append(dict(audio=str(Path.home() / f'Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3'),
                             checks=list(before.values()), predicted_checks=deepcopy(list(checks.values())),
                             align_lag_ms=0, fresh=True))
        # Save current predictions before the test's external reference comparison.
        (TASK / 'fresh-controls.json').write_text(json.dumps(captured, indent=2) + '\n')

    full_signal.refine_late_boundaries = refine
    try:
        status = pytest.main(['-q', 'tests/test_alignment_audio.py', '-k', 'full', '--tb=short'])
    finally:
        full_signal.refine_late_boundaries = original
    assert status == 0 and len(captured) == 5


if __name__ == '__main__':
    main()
