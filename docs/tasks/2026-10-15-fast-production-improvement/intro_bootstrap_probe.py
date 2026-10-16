"""Fresh bounded CTC diagnosis: heard basmala word cues a local intro-end read."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_executor import Policy
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text, transliteration
from tools.bench_fast_models import dispatch_spy, verify_dispatch

P = Path(sys.argv[1]).resolve()
assert not (P / 'ctc-intro-probe.json').exists()
prior = P.parent / '2026-10-15-fast-systematic-execution'
names = ['saad_al_ghamidi', 'mohammed_siddiq_al_minshawi', 'mahmoud_khalil_al_hussary']
results = []
for model in ['quran-phonetics', 'quran-phonetics-fp32']:
    model_store.ACTIVE_ID = model
    bundle = align.default_bundle()
    adapter = CTCAdapter(bundle)
    target = adapter.represent(quran_text.surah_words(112))[0]
    intro = tuple(c for c in ''.join(align._basmala_trs(transliteration)) if c not in ('-', '|', ' '))
    first_word = tuple(c for c in align._basmala_trs(transliteration)[0] if c not in ('-', '|', ' '))
    for name in names:
        old = next(c for c in json.loads((prior / f'baseline-{name}.json').read_text())
                   if c['model'] == model and c['surah'] == 112)
        start, end = old['trace']['acoustic_reads'][0]['window_ms']
        audio = Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / name / '112.mp3'
        pcm = decode.decode(str(audio))
        reads = LocalEvidenceReads(pcm, 16000, adapter)
        item = {'model': model, 'reciter': name, 'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
                'hypothesis': 'intro word timing forecasts one local intro-end crop; no reference window selection',
                'runtime': adapter.runtime_identity, 'lag_ms': adapter.lag_ms,
                'quality_threshold': Policy().ctc_quality, 'intro_target': ''.join(intro),
                'initial_cue_word': ''.join(first_word),
                'target': ''.join(target), 'prefix_occurrences': []}
        with dispatch_spy(bundle, pcm) as actual:
            obs = reads.read(start, end, reason='intro-diagnostic:independent-basmala-word')
            item['initial_text'] = obs.text
            if (tuple(obs.units[:len(first_word)]) == first_word
                    and min(obs.unit_quality[:len(first_word)], default=0.) >= Policy().ctc_quality):
                positions = obs.positions_ms[:len(first_word)]
                rate = (positions[-1] - positions[0]) / (len(first_word) - 1)
                forecast = reads.calls[-1]['window_ms'][0] + positions[0] + rate * len(intro)
                left, right = max(end, forecast - 1000), min(len(pcm) / 16, forecast + 1500)
                item.update(rate_ms_per_phone=rate, forecast_ms=forecast, requested_window_ms=[left, right])
                local = reads.read(left, right, reason='intro-diagnostic:heard-basmala-tempo-cue')
                if local:
                    item['local_text'] = local.text
                    for i in range(len(local.units) - len(target) + 1):
                        if tuple(local.units[i:i + len(target)]) == target:
                            item['prefix_occurrences'].append({'offset': i,
                                'quality': local.unit_quality[i:i + len(target)],
                                'onset_ms': reads.calls[-1]['window_ms'][0] + local.positions_ms[i]})
            else:
                item['decline'] = 'intro-word-not-independently-supported'
        trace = {'total_ms': len(pcm) / 16, **reads.diagnostics()}
        verify_dispatch(actual, pcm, trace)
        item.update(trace=trace, actual_dispatches=actual)
        results.append(item)
        (P / 'ctc-intro-probe.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(model, name, item['initial_text'], item.get('local_text'), item['prefix_occurrences'], flush=True)
    align._ONNX_CACHE.clear()
    release_runtimes()
# Predictions above are fully written before opening any reference DB.
evaluated = []
for item in results:
    occurrences = item['prefix_occurrences']
    selected = (round(occurrences[0]['onset_ms']) if len(occurrences) == 1
                and min(occurrences[0]['quality']) >= Policy().ctc_quality else None)
    reference = (Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / item['reciter']
                 / f"{item['reciter']}.db")
    with sqlite3.connect(f'file:{reference}?mode=ro', uri=True) as conn:
        truth = conn.execute('SELECT time FROM timings WHERE sura=112 AND ayah=1').fetchone()[0]
    evaluated.append({'model': item['model'], 'reciter': item['reciter'],
                      'selected_onset_ms': selected, 'reference_ms': truth,
                      'error_ms': abs(selected - truth) if selected is not None else None})
(P / 'ctc-intro-probe-evaluation.json').write_text(json.dumps(evaluated, ensure_ascii=False, indent=2) + '\n')
