"""Real model null/wrong-owner controls for the short-crop identity hypothesis."""
import hashlib
import json
from pathlib import Path
import sys


from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from tools.bench_fast_models import dispatch_spy, verify_dispatch

P = Path(sys.argv[1]).resolve()
assert not (P / 'negative-audio.json').exists()
audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/112.mp3')
original = decode.decode(str(audio))
results = []
for model in ['quran-phonetics', 'quran-phonetics-fp32']:
    model_store.ACTIVE_ID = model
    bundle = align.default_bundle()
    adapter = CTCAdapter(bundle)
    for kind, start, end in [('zero', 6120., 6820.), ('wrong-owner', 8680., 9380.),
                             ('late-non-target', 13480., 14180.)]:
        pcm = original.copy()
        if kind == 'zero':
            pcm[round(start * 16):round(end * 16)] = 0
        reads = LocalEvidenceReads(pcm, 16000, adapter)
        with dispatch_spy(bundle, pcm) as actual:
            obs = reads.read(start, end, reason=f'negative:{kind}:qul')
        trace = {'total_ms': len(pcm) / 16, **reads.diagnostics()}
        verify_dispatch(actual, pcm, trace)
        accepted = (tuple(obs.units[:3]) == tuple('qul') and not obs.truncated
                    and min(obs.unit_quality[:3], default=0.) >= .25)
        results.append({'model': model, 'kind': kind, 'target': 'qul', 'text': obs.text,
            'accepted': accepted, 'runtime': adapter.runtime_identity, 'trace': trace,
            'actual_dispatches': actual,
            'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest()})
        (P / 'negative-audio.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(model, kind, repr(obs.text), 'accepted', accepted, flush=True)
    align._ONNX_CACHE.clear()
    release_runtimes()
assert not any(r['accepted'] for r in results)
