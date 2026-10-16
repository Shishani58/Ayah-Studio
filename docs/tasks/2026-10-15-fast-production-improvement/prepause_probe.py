"""One Base: test independent opening evidence BEFORE a suspect DSP interval."""
import hashlib
import json
from pathlib import Path
import sys

from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_adapters import TarteelAdapter
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from tools.bench_fast_models import dispatch_spy, verify_dispatch

P = Path(sys.argv[1]).resolve()
assert not (P / 'prepause-probe.json').exists()
model_store.ACTIVE_ID = 'whisper-tarteel-base'
bundle = align.default_bundle()
adapter = TarteelAdapter(bundle)
cases = json.loads((P / 'wide-prototype-c.json').read_text())
results = []
for surah, ayah in [(84, 10), (28, 4), (41, 3), (72, 6)]:
    case = next(c for c in cases if (c['model'], c['surah']) == ('whisper-tarteel-base', surah))
    boundary = next(b for b in case['trace']['boundaries'] if b['ayah'] == ayah)
    left, right = boundary['proof']['pause_ms']
    start = max(0, left - 1300)
    phrase = boundary['proof']['phrase']
    pcm = decode.decode(f'/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03}.mp3')
    reads = LocalEvidenceReads(pcm, 16000, adapter)
    observations = []
    with dispatch_spy(bundle, pcm) as actual:
        for end in [left, (left + right) / 2]:
            obs = reads.read(start, end, reason='diagnostic:opening-before-selected-pause')
            matches = [i for i in range(len(obs.units) - len(phrase) + 1)
                       if list(obs.units[i:i + len(phrase)]) == phrase
                       and min(obs.unit_quality[i:i + len(phrase)], default=0) >= .6]
            observations.append({'end_ms': end, 'text': obs.text, 'units': obs.units,
                'quality': obs.unit_quality, 'matches': matches})
    trace = {'total_ms': len(pcm) / 16, **reads.diagnostics()}
    verify_dispatch(actual, pcm, trace)
    result = {'model': 'whisper-tarteel-base', 'surah': surah, 'ayah': ayah,
        'pause_ms': [left, right], 'phrase': phrase, 'requested_start_ms': start,
        'audio_sha256': hashlib.sha256(Path(f'/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03}.mp3').read_bytes()).hexdigest(),
        'observations': observations, 'trace': trace, 'actual_dispatches': actual}
    results.append(result)
    (P / 'prepause-probe.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(surah, ayah, observations, flush=True)
release_runtimes()
