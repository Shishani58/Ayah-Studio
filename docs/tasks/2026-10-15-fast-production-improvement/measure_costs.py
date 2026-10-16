"""Summarize recorded fresh dispatch/timer/RSS data; never runs inference."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

P = Path(__file__).resolve().parent

def load(name):
    return json.loads((P / name).read_text())

def aggregate(cases):
    confirmed = [(row, error) for case in cases
                 for row, error in zip(case['rows'], case['errors_ms']) if row[2] > 0]
    return {
        'cases': len(cases), 'ayahs': sum(len(c['rows']) for c in cases),
        'confirmed': len(confirmed),
        'confirmed_reference_within500': sum(e <= 500 for _, e in confirmed),
        'confirmed_reference_over500': sum(e > 500 for _, e in confirmed),
        'executor_s': sum(c['seconds'] for c in cases),
        'model_s': sum(c['trace']['inference_seconds'] for c in cases),
        'executor_other_s': sum(c['seconds'] - c['trace']['inference_seconds'] for c in cases),
        'actual_dispatches': sum(len(c['actual_dispatches']) for c in cases),
        'model_input_ms': sum(c['trace']['audio_ms'] for c in cases),
        'union_ms_sum_per_recording': sum(c['trace']['covered_audio_ms'] for c in cases),
        'max_union_ratio': max(c['trace']['coverage'] for c in cases),
        'max_input_ratio': max(c['trace']['audio_ms'] / c['duration_ms'] for c in cases),
        'max_actual_read_ms': max((b - a for c in cases for r in c['trace']['acoustic_reads']
                                   for a, b in [r['window_ms']]), default=0),
        'max_cache_bytes': max(c['trace']['peak_cache_bytes'] for c in cases),
        'cache_hits': sum(c['trace']['cache_hits'] for c in cases),
        'cache_evictions': sum(c['trace']['cache_evictions'] for c in cases),
    }

out = {}
for label in ['wide-baseline', 'wide-prototype-c', 'warm-baseline', 'warm-prototype-c']:
    summary = load(label + '.summary.json')
    cases = load(label + '.json')
    process = load(label + '-process.json')
    times = [s['elapsed_s'] for s in process['samples']]
    by_model = {}
    for model in ['whisper-tarteel-base']:
        subset = [c for c in cases if c['model'] == model]
        record = aggregate(subset)
        if label == 'wide-prototype-c':
            publication = [c for c in load('f-replay-predictions.json') if c['model'] == model]
            accepted = [(r, e) for c in publication for r, e in zip(c['rows'], c['errors_ms']) if r[2] > 0]
            record.update(confirmed=len(accepted),
                confirmed_reference_within500=sum(e <= 500 for _, e in accepted),
                confirmed_reference_over500=sum(e > 500 for _, e in accepted),
                publication_source='F same-input replay; timers and model inputs measured in D')
        record['load_probe_s'] = summary['summary'][model]['load_probe_seconds']
        record['synthetic_startup_probe'] = {'calls': 1, 'input_ms': 1000,
            'signal': 'synthetic zeros; encoder+decoder; not recording input',
            'cost_bucket': 'load_probe_s'}
        if label.startswith('wide'):
            record['decode_s'] = summary['summary'][model]['decode_seconds']
            record['initial_dsp_s'] = summary['summary'][model]['dsp_seconds']
        else:
            groups = defaultdict(list)
            for c in subset:
                groups[c['surah']].append(c)
            record['warm_repeats'] = {
                str(s): {'executor_s': [c['seconds'] for c in group],
                         'median_executor_s': median(c['seconds'] for c in group),
                         'model_s': [c['trace']['inference_seconds'] for c in group],
                         'rows_identical': all(c['rows'] == group[0]['rows'] for c in group),
                         'dispatches_identical': all(c['actual_dispatches'] == group[0]['actual_dispatches'] for c in group)}
                for s, group in groups.items()}
        by_model[model] = record
    out[label] = {
        'models': by_model, 'process_wall_s': process['wall_s'],
        'tree_peak_kib': process['tree_peak_kib'],
        'rss_nominal_interval_s': process['sampling_s'],
        'rss_observed_max_gap_s': max((b - a for a, b in zip(times, times[1:])), default=0),
        'rss_samples': len(times),
        'rss_scope': 'same-time sum of benchmark process and observed descendants; per-PID series retained',
        'process_scope': 'wide historical process includes prior Tiny stage; per-model fields filtered to Base' if label.startswith('wide') else 'Base only',
        'host_before': process.get('host_before'), 'host_after': process.get('host_after'),
        'preprocessing_note': 'wide decode/DSP shared by models, do not sum twice; warm repeats overwrite per-surah preprocessing metadata, so only warm executor/model times used',
    }
(P / 'costs.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(out, ensure_ascii=False, indent=2))
