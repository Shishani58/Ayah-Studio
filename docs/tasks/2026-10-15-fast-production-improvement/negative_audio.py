"""Independent real decoding of target, silence, and competing vocative spelling."""
import json
from pathlib import Path
import sys

from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_adapters import TarteelAdapter
from surah_cutter.core.fast_executor import Policy, _expansion_evidence
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data.quran_text import surah_words
from tools.bench_fast_models import dispatch_spy, verify_dispatch

P = Path(sys.argv[1]).resolve()
assert not (P / 'negative-audio.json').exists()
pcm = decode.decode('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/065.mp3')
results = []
for model in ['whisper-tarteel-tiny', 'whisper-tarteel-base']:
    model_store.ACTIVE_ID = model
    bundle = align.default_bundle()
    adapter = TarteelAdapter(bundle)
    words = surah_words(65)
    groups, ends = {}, {}
    for word, units in zip(words, adapter.represent(words)):
        groups.setdefault(word.ayah, []).extend(units)
        ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
    groups = {a: tuple(u) for a, u in groups.items()}
    reads = LocalEvidenceReads(pcm, 16000, adapter)
    with dispatch_spy(bundle, pcm) as actual:
        short = reads.read(6180, 7680, reason='negative-control:independent-short-vocative')
        full = reads.read(6180, 8880, reason='negative-control:independent-full-vocative')
    trace = {'total_ms': len(pcm) / 16, **reads.diagnostics()}
    verify_dispatch(actual, pcm, trace)
    wrong = ('يا', 'اولي')
    assert any(units[i:i + 2] == wrong for units in groups.values() for i in range(len(units)))
    result = {'model': model, 'original_trace': trace, 'actual_dispatches': actual,
        'independent_text': full.text, 'quality': full.unit_quality,
        'correct_target_prefix': groups[1][:2], 'wrong_internal_target': wrong,
        'wrong_target_would_confirm': tuple(full.units[:2]) == wrong,
        'short_cue': _expansion_evidence(short, groups[1][:2], ends[1], Policy().asr_quality, groups, 1, ends)}
    zero_pcm = pcm.copy()
    zero_pcm[round(6180 * 16):round(8880 * 16)] = 0
    zeros = LocalEvidenceReads(zero_pcm, 16000, adapter)
    with dispatch_spy(bundle, zero_pcm) as zero_actual:
        obs = zeros.read(6180, 8880, reason='negative-control:zero')
    zero_trace = {'total_ms': len(pcm) / 16, **zeros.diagnostics()}
    verify_dispatch(zero_actual, zero_pcm, zero_trace)
    result.update(zero_trace=zero_trace, zero_actual_dispatches=zero_actual,
                  zero_text=obs.text, zero_would_confirm=tuple(obs.units[:2]) == groups[1][:2])
    results.append(result)
    (P / 'negative-audio.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(model, 'full', full.text, 'wrong', result['wrong_target_would_confirm'],
          'zero', obs.text, result['zero_would_confirm'], flush=True)
    align._ONNX_CACHE.clear()
    release_runtimes()
assert not any(r['wrong_target_would_confirm'] or r['zero_would_confirm'] for r in results)
