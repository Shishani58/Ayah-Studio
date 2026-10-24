"""Replay signal refinement on saved predictions; never read timing references."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from surah_cutter.core import decode, full_signal

TASK = Path(__file__).resolve().parent
AUDIO = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri')
RUN = AUDIO / 'Abu_Bakr_Ash-Shatri.logs/runs/sura_002_full_f68c47d26f7744499f6d74b2948ff51e.json'


def replay(audio, source):
    pcm = decode.decode(str(audio))
    checks = {c['ayah']: deepcopy(c) for c in source}
    before = deepcopy(checks)
    trace = {}
    started = time.perf_counter()
    full_signal.refine_late_boundaries(checks, pcm, 16000, trace=trace)
    elapsed = time.perf_counter() - started
    assert all(c['conf'] == before[a]['conf'] for a, c in checks.items())
    assert all(c['raw_onset_ms'] == before[a]['raw_onset_ms'] for a, c in checks.items())
    times = [c['onset_ms'] for c in checks.values()]
    assert all(a < b for a, b in zip(times, times[1:]))
    changes = []
    for a, c in checks.items():
        if c['onset_ms'] == before[a]['onset_ms']:
            continue
        record = next(r for r in trace['full_signal_checks'] if r['ayah'] == a and r['accepted'])
        assert 0 < before[a]['onset_ms'] - c['onset_ms'] <= 500
        assert record['pause_ms'][0] < record['quiet_window_ms'][0]
        assert record['quiet_window_ms'][1] < record['pause_ms'][1]
        assert checks[a - 1]['full_ctc_end_ms'] <= record['quiet_window_ms'][0]
        assert not c['reliable']
        changes.append(dict(ayah=a, before_ms=before[a]['onset_ms'], after_ms=c['onset_ms'],
                            weak_head=record['weak_head'], pause_ms=record['pause_ms']))
    return dict(audio=str(audio), n=len(checks), elapsed_s=elapsed, changes=changes,
                outcomes=dict(Counter(r.get('reason') for r in trace['full_signal_checks'])),
                checks=list(checks.values()), trace=trace)


def main():
    protected = [AUDIO / '002.mp3', AUDIO / 'Abu_Bakr_Ash-Shatri.db']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    saved = json.loads(RUN.read_text())
    result = replay(AUDIO / '002.mp3', saved['trace']['ayahs'])
    (TASK / 'baqara-replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'trace')}), flush=True)
    controls = json.loads((TASK.parent / '2026-10-16-full-transition-refinement/verification.json').read_text())
    v9 = json.loads((TASK.parent / '2026-10-16-waveform-boundaries/comparison.json').read_text())
    results = []
    for item, previous in zip(controls, v9, strict=True):
        audio = next(p for p in item['protected_sha256'] if p.endswith('.mp3'))
        old = deepcopy(item['before_checks'])
        results.append(replay(audio, old))
        baseline = {c['ayah']: c['onset_ms'] for c in old}
        baseline.update({c['ayah']: c['after_ms'] for c in previous['changes']})
        delta = [c for c in results[-1]['checks'] if c['onset_ms'] != baseline[c['ayah']]]
        assert all(c['first_token_conf'] < .25 for c in delta)
        results[-1]['changes_since_v9'] = [dict(ayah=c['ayah'], before_ms=baseline[c['ayah']],
                                               after_ms=c['onset_ms']) for c in delta]
        print(json.dumps(dict(audio=audio, changes_since_v9=results[-1]['changes_since_v9'],
                              elapsed_s=results[-1]['elapsed_s'])), flush=True)
    (TASK / 'controls.json').write_text(json.dumps(results, indent=2) + '\n')
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before == after
    (TASK / 'protected.json').write_text(json.dumps(dict(before=before, after=after), indent=2) + '\n')


if __name__ == '__main__':
    main()
