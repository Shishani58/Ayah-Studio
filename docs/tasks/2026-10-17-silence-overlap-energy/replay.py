"""Validate revised cuts against real PCM and saved independent predictions."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from surah_cutter.core import decode, full_signal

TASK = Path(__file__).resolve().parent
AUDIO = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri')


def replay(item):
    checks = {c['ayah']: deepcopy(c) for c in item['checks']}
    original = deepcopy(checks)
    for check in checks.values():
        if check.get('reason') == 'waveform-pause-recheck':
            check.update(onset_ms=check['pre_signal_onset_ms'], repaired=False,
                         reason='weak-onset' if check['first_token_conf'] < .25 else 'weak-word')
    pcm = decode.decode(item['audio'])
    trace = {}
    started = time.perf_counter()
    full_signal.refine_late_boundaries(checks, pcm, 16000, trace=trace)
    elapsed = time.perf_counter() - started
    times = [c['onset_ms'] for c in checks.values()]
    assert all(a < b for a, b in zip(times, times[1:]))
    changes = []
    for a, c in checks.items():
        assert c['conf'] == original[a]['conf']
        assert c['raw_onset_ms'] == original[a]['raw_onset_ms']
        if c['onset_ms'] == original[a]['onset_ms']:
            continue
        assert 0 < original[a]['onset_ms'] - c['onset_ms'] <= 500, (
            item['audio'], a, original[a]['onset_ms'], c['onset_ms'])
        assert not c['reliable']
        record = next(r for r in trace['full_signal_checks'] if r['ayah'] == a and r['accepted'])
        lo, hi = record['quiet_window_ms']
        assert checks[a - 1]['full_ctc_end_ms'] <= lo < hi <= c['full_ctc_first_ms']
        shelves = [p for k in ('pause_local_decisions', 'pause_stability_decisions')
                   for p in c.get(k, []) if p['accepted'] and p['owned']
                   and p['start_ms'] < min(c['raw_onset_ms'], c['full_ctc_first_ms'])
                   and p['end_ms'] > checks[a - 1]['full_ctc_end_ms']]
        if not record['pause_edges_cross_anchor']:
            margin = trace['full_signal_policy']['margin_ms']
            shelves = [p for p in shelves if checks[a - 1]['full_ctc_end_ms'] <= p['start_ms']
                       and p['end_ms'] <= c['raw_onset_ms'] + margin
                       and p['end_ms'] - margin < record['before_ms']]
        assert shelves and all(p['start_ms'] <= lo < hi <= p['end_ms'] for p in shelves)
        changes.append(dict(ayah=a, before_ms=original[a]['onset_ms'], after_ms=c['onset_ms'],
                            pause_ms=record['pause_ms'], quiet_window_ms=[lo, hi]))
    return dict(audio=item['audio'], n=len(checks), elapsed_s=elapsed, changes=changes,
                checks=list(checks.values()), trace=trace)


def main():
    protected = [AUDIO / '002.mp3', AUDIO / 'Abu_Bakr_Ash-Shatri.db']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    sources = [Path(full_signal.__file__), Path(full_signal.__file__).with_name('full_pause_center.py')]
    source_sha = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    run = json.loads((AUDIO / 'Abu_Bakr_Ash-Shatri.logs/runs/sura_002_full_5f7b66839a21414391b9d85f1eb7b96b.json').read_text())
    result = replay(dict(audio=str(AUDIO / '002.mp3'), checks=run['trace']['ayahs']))
    (TASK / 'baqara-replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(surah=2, changed=len(result['changes']), elapsed_s=result['elapsed_s'])), flush=True)
    controls = json.loads((TASK / 'fresh-controls.json').read_text())
    results = [replay(item) for item in controls]
    for item, result in zip(controls, results, strict=True):
        assert [c['onset_ms'] for c in result['checks']] == [c['onset_ms'] for c in item['predicted_checks']]
    (TASK / 'controls.json').write_text(json.dumps(results, indent=2) + '\n')
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before == after
    source_after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert source_sha == source_after
    (TASK / 'protected.json').write_text(json.dumps(dict(before=before, after=after,
                                                       source_before=source_sha, source_after=source_after), indent=2) + '\n')
    print(json.dumps(dict(recordings=len(results), ayahs=sum(r['n'] for r in results),
                          changed=sum(len(r['changes']) for r in results))), flush=True)


if __name__ == '__main__':
    main()
