"""Compare completed fresh predictions; this script never generates timings."""

import hashlib
import json
from pathlib import Path

TASK = Path(__file__).resolve().parent


def main():
    before = json.loads((TASK / 'native-full.json').read_text())
    after = json.loads((TASK / 'native-full-final.json').read_text())
    replay = json.loads((TASK / 'baqara-replay.json').read_text())
    protected = json.loads((TASK / 'protected.json').read_text())
    assert after['source_unchanged'] and 'error' not in after
    assert after['rows'] == before['rows']
    assert after['trace']['ayahs'] == before['trace']['ayahs']
    assert after['trace']['full_windows']['pause_decisions'] == before['trace']['full_windows']['pause_decisions']
    assert after['trace']['full_windows']['windows'] == before['trace']['full_windows']['windows']
    assert after['comparison']['confidences_identical']
    assert after['comparison']['windows_identical']
    assert all(c['onset_ms'] == r['onset_ms'] for c, r in zip(after['trace']['ayahs'], replay['checks'], strict=True))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected['before'].items())
    verification = dict(wall_s=after['wall_s'], dsp_s=after['trace']['full_windows']['dsp_s'],
                        inference_s=after['trace']['full_windows']['inference_s'],
                        inference_cpu_s=after['trace']['full_windows']['inference_cpu_s'],
                        n_ayahs=len(after['rows']), changed=len(replay['changes']),
                        all_rows_and_checks_equal_before_rms_optimization=True,
                        pause_decisions_and_windows_identical=True,
                        confidence_identical_to_user_run=True, source_unchanged=True,
                        user_audio_and_db_unchanged=True)
    (TASK / 'native-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
