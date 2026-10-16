"""Compare minimal SF with production F13, including unchanged anchors and costs."""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def load(name):
    return json.loads((P / name).read_text())


def largest_review_block(rows):
    largest = current = 0
    for _, _, quality in rows:
        current = 0 if quality > 0 else current + 1
        largest = max(largest, current)
    return largest


reused = load('PF-wide-replay.json')
fresh_file = P / 'SF-wide-fresh-missing.json'
fresh = json.loads(fresh_file.read_text()) if fresh_file.exists() else []
combined = {c['surah']: c for c in reused}
assert sorted(combined) == list(range(1, 115))
wide = [combined[s] for s in range(1, 115)]
(P / 'SF-wide-combined.json').write_text(json.dumps(wide, ensure_ascii=False) + '\n')
pairs = [('shatri', load('A-wide.json'), wide)]
for reciter in ('saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi'):
    pairs.append((reciter, load('final-A-' + reciter + '.json'), load('PF-final-SF-' + reciter + '-replay.json')))
records, failures = [], []
for reciter, before, after in pairs:
    assert [c['surah'] for c in before] == [c['surah'] for c in after]
    for a, b in zip(before, after, strict=True):
        for key in ('audio_sha256', 'initial_pauses_sha256', 'reference_sha256'):
            assert a[key] == b[key], (reciter, a['surah'], key)
        for key in ('pcm_sha256', 'text_runtime', 'text_sha256', 'dsp_pauses_sha256', 'policy', 'limits'):
            assert a['trace']['run_snapshot'][key] == b['trace']['run_snapshot'][key], (reciter, a['surah'], key)
        gains, losses, changes, bad = [], [], [], []
        for old, new, error in zip(a['rows'], b['rows'], b['errors_ms'], strict=True):
            assert old[0] == new[0]
            if old[2] > 0 and new[2] == 0:
                losses.append(old[0])
            if old[2] > 0 and new[1] != old[1]:
                changes.append([old, new])
            if old[2] == 0 and new[2] > 0:
                gains.append(new[0])
                if error > 500:
                    bad.append([new[0], error])
        trace = b['trace']
        intervals = [r['window_ms'] for r in trace['acoustic_reads'] if r['outcome'] == 'completed']
        duration = b['duration_ms']
        hard = (all(0 <= left < right <= duration and right - left <= 6000 for left, right in intervals)
                and trace['audio_ms'] <= duration * 2 + 1
                and trace['covered_audio_ms'] <= duration * .65 + 1
                and trace['peak_cache_bytes'] <= 32 * 1024 * 1024
                and trace['emission_passes'] == 0 and trace['bounded_ctc_fallback'] is None)
        assert hard, (reciter, a['surah'], 'hard_locality_limits')
        reused_case = b.get('measurement_kind', '').startswith('counterfactual')
        record = {'reciter': reciter, 'surah': a['surah'], 'final_holdout': reciter != 'shatri' and a['surah'] in (67, 77),
                  'before_proofs': sum(r[2] > 0 for r in a['rows']),
                  'after_proofs': sum(r[2] > 0 for r in b['rows']), 'gained': gains, 'lost': losses,
                  'changed_previous_times': changes, 'new_over500': bad,
                  'before_bad': sum(r[2] > 0 and e > 500 for r, e in zip(a['rows'], a['errors_ms'], strict=True)),
                  'after_bad': sum(r[2] > 0 and e > 500 for r, e in zip(b['rows'], b['errors_ms'], strict=True)),
                  'before_all_over500': sum(e > 500 for e in a['errors_ms']),
                  'after_all_over500': sum(e > 500 for e in b['errors_ms']),
                  'before_largest_review_block': largest_review_block(a['rows']),
                  'after_largest_review_block': largest_review_block(b['rows']),
                  'before_blocked_ayahs': sum(x['review_reason'] == 'blocked_by_unresolved_ayah'
                                              for x in a['trace']['boundaries']),
                  'after_blocked_ayahs': sum(x['review_reason'] == 'blocked_by_unresolved_ayah'
                                             for x in b['trace']['boundaries']),
                  'before_calls': len(a['actual_dispatches']), 'after_calls': len(b['actual_dispatches']),
                  'before_input_ms': a['trace']['audio_ms'], 'after_input_ms': trace['audio_ms'],
                  'after_union_ms': trace['covered_audio_ms'], 'duration_ms': duration,
                  'before_search_seconds': a['search_seconds'],
                  'after_search_seconds': None if reused_case else b['search_seconds'],
                  'counterfactual_observation_reuse': reused_case, 'hard_limits_passed': hard}
        records.append(record)
        if losses or changes or bad:
            failures.append({k: record[k] for k in ('reciter', 'surah', 'lost', 'changed_previous_times', 'new_over500')})
summary = {}
for reciter in sorted({r['reciter'] for r in records}):
    rows = [r for r in records if r['reciter'] == reciter]
    summary[reciter] = {'records': len(rows), **{k: sum(r[k] for r in rows) for k in (
        'before_proofs', 'after_proofs', 'before_bad', 'after_bad', 'before_all_over500',
        'after_all_over500', 'before_blocked_ayahs', 'after_blocked_ayahs', 'before_calls', 'after_calls',
        'before_input_ms', 'after_input_ms', 'after_union_ms', 'duration_ms')},
        'gained': sum(len(r['gained']) for r in rows)}
positive = [name for name, s in summary.items() if s['gained'] > 0]
gate = {'passed': not failures and len(positive) >= 3, 'failures': failures, 'positive_reciters': positive,
        'quality': summary, 'paired_recordings': len(records), 'canonical_wide_rows': sum(len(c['rows']) for c in wide),
        'wide_final_exact_observation_reuse': len(reused), 'wide_missing_inputs_resolved_with_fresh_records': len(fresh),
        'physical_onset_accuracy_claimed': False, 'all_hard_limits_passed': True}
assert gate['canonical_wide_rows'] == 6236
(P / 'SF-corpus-gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2) + '\n')
(P / 'SF-per-record.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
with (P / 'SF-per-record.csv').open('w') as target:
    writer = csv.DictWriter(target, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print(json.dumps(gate, ensure_ascii=False, indent=2))
