"""Paired row-level quality, coverage, cascade and cost gates after predictions."""
import collections
import csv
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
records, causes, failures = [], {}, []


def blocks(rows):
    lengths, current = [], 0
    for _, _, quality in rows:
        if quality == 0:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


for reciter, suffix in [('shatri', 'wide'), *[(r, r) for r in (
        'saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi')]]:
    before = 'A-wide' if reciter == 'shatri' else 'selected-A-' + reciter
    after = 'BS-wide' if reciter == 'shatri' else 'selected-BS-' + reciter
    old = json.loads((P / (before + '.json')).read_text())
    new = json.loads((P / ('BS2-' + after + '-replay.json')).read_text())
    for a, b in zip(old, new, strict=True):
        assert a['surah'] == b['surah']
        for key in ('audio_sha256', 'initial_pauses_sha256', 'reference_sha256'):
            assert a[key] == b[key], (reciter, a['surah'], key)
        for key in ('pcm_sha256', 'text_runtime', 'text_sha256', 'dsp_pauses_sha256'):
            assert a['trace']['run_snapshot'][key] == b['trace']['run_snapshot'][key], (reciter, a['surah'], key)
        gained, lost, changed, new_bad = [], [], [], []
        for ar, br, ae, be in zip(a['rows'], b['rows'], a['errors_ms'], b['errors_ms'], strict=True):
            assert ar[0] == br[0]
            if ar[2] > 0 and br[2] == 0:
                lost.append(ar[0])
            if ar[2] > 0 and ar[1] != br[1]:
                changed.append([ar, br])
            if ar[2] == 0 and br[2] > 0:
                gained.append(br[0])
                if be > 500:
                    new_bad.append([br[0], be])
        record = {'reciter': reciter, 'surah': a['surah'], 'control': reciter != 'shatri' and a['surah'] in (67, 77),
            'ayahs': len(a['rows']), 'before_proofs': sum(r[2] > 0 for r in a['rows']),
            'after_proofs': sum(r[2] > 0 for r in b['rows']), 'gained': gained, 'lost': lost,
            'changed_previous_times': changed, 'new_bad_proofs': new_bad,
            'before_bad_proofs': sum(r[2] > 0 and e > 500 for r, e in zip(a['rows'], a['errors_ms'])),
            'after_bad_proofs': sum(r[2] > 0 and e > 500 for r, e in zip(b['rows'], b['errors_ms'])),
            'before_all_over500': sum(e > 500 for e in a['errors_ms']),
            'after_all_over500': sum(e > 500 for e in b['errors_ms']),
            'before_estimate_over500': sum(r[2] == 0 and e > 500 for r, e in zip(a['rows'], a['errors_ms'])),
            'after_estimate_over500': sum(r[2] == 0 and e > 500 for r, e in zip(b['rows'], b['errors_ms'])),
            'before_max_block': max(blocks(a['rows']), default=0), 'after_max_block': max(blocks(b['rows']), default=0),
            'before_calls': len(a['actual_dispatches']), 'after_calls': len(b['actual_dispatches']),
            'before_audio_ms': a['trace']['audio_ms'], 'after_audio_ms': b['trace']['audio_ms'],
            'before_covered_ms': a['trace']['covered_audio_ms'], 'after_covered_ms': b['trace']['covered_audio_ms'],
            'duration_ms': a['duration_ms'], 'before_seconds': a['seconds'], 'after_seconds': b['seconds'],
            'time_ratio': b['seconds'] / a['seconds'], 'publication_reviews': b.get('publication_changes', [])}
        if lost or changed or new_bad:
            failures.append({'reciter': reciter, 'surah': a['surah'], 'lost': lost, 'changed': changed, 'new_bad': new_bad})
        records.append(record)
        for variant, case in [('A', a), ('BS2', b)]:
            key = reciter + ':' + str(case['surah']) + ':' + variant
            causes[key] = {'estimate_reasons': dict(collections.Counter(x['review_reason'] for x in case['trace']['boundaries'] if x['evidence_kind'] == 'estimate')),
                'root_events': [e for e in case['trace']['transitions'] if e['state'] == 'REVIEW_REQUIRED'],
                'geometry_rejections': [e for e in case['trace']['transitions'] if e.get('declined_geometry')],
                'denied_reads': case['trace']['denied_acoustic_reads'], 'block_lengths': blocks(case['rows'])}

summary = {}
for reciter in sorted({r['reciter'] for r in records}):
    rows = [r for r in records if r['reciter'] == reciter]
    summary[reciter] = {'records': len(rows), 'gained': sum(len(r['gained']) for r in rows),
        **{key: sum(r[key] for r in rows) for key in ('before_proofs', 'after_proofs', 'before_all_over500',
             'after_all_over500', 'before_bad_proofs', 'after_bad_proofs', 'before_calls', 'after_calls',
             'before_audio_ms', 'after_audio_ms', 'before_covered_ms', 'after_covered_ms', 'duration_ms')},
        'median_time_ratio': float(np.median([r['time_ratio'] for r in rows])),
        'p95_time_ratio': float(np.percentile([r['time_ratio'] for r in rows], 95))}
gate = {'passed': not failures and sum(s['gained'] > 0 for s in summary.values()) >= 3,
        'failures': failures, 'positive_reciters': [r for r, s in summary.items() if s['gained'] > 0],
        'paired_records': len(records), 'quality': summary,
        'cost_scope': 'fresh A/BS reads; BS2 exact-PCM publication-only replay. Timing subject to host contention; exclusive warm gate separate.'}
for name, data in [('per-record.json', records), ('causes.json', causes), ('corpus-gate.json', gate)]:
    (P / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
with (P / 'per-record-table.csv').open('w') as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print(json.dumps(gate, ensure_ascii=False, indent=2))
