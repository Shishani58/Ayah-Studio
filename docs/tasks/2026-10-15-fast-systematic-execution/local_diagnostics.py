"""Causal local CTC context and event diagnostics, never a production proof."""
import hashlib
import json
import math
from pathlib import Path
import sys


from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_ctc import prefix_scores
from surah_cutter.core.fast_executor import Policy, plan_phrase
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

P = Path(sys.argv[1]).resolve()
ROOT = Path.cwd().resolve()
assert Path(align.__file__).resolve().is_relative_to(ROOT)
output = P / 'local-diagnostics.json'
assert not output.exists(), 'Never repeat an experiment ID'
# Fixed from historical observed PCM, no reference timings involved.
CASES = [(1, 1, 200., 'positive-first'), (80, 1, 6180., 'missing-first-phone'),
         (98, 2, 16940., 'different-phone'), (112, 1, 6120., 'different-opening')]
policy = Policy()
results = []
for model in ['quran-phonetics', 'quran-phonetics-fp32']:
    model_store.ACTIVE_ID = model
    bundle = align.default_bundle()
    adapter = CTCAdapter(bundle)
    for surah, ayah, start, role in CASES:
        audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri') / f'{surah:03}.mp3'
        pcm = decode.decode(str(audio))
        words = quran_text.surah_words(surah)
        rendered = adapter.represent(words)
        groups, ends = {}, {}
        for w, units in zip(words, rendered):
            groups.setdefault(w.ayah, []).extend(units)
            ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
        groups = {a: tuple(u) for a, u in groups.items()}
        phrase = plan_phrase(groups, ayah, ends)['units']
        # Fixed shortest context versus ordinary local phrase context. Neither
        # crop is claimed to be a human-verified isolated word.
        reads = LocalEvidenceReads(pcm, 16000, adapter)
        with dispatch_spy(bundle, pcm) as actual:
            for duration in [700., 1500.]:
                obs = reads.read(start, start + duration, reason=f'H1:{role}:context={duration}')
                if obs is None:
                    continue
                alternatives = []
                owners = []
                for a, units in groups.items():
                    for word_start in [0, *ends[a][:-1]]:
                        rival = units[word_start:word_start + len(phrase)]
                        if len(rival) == len(phrase) and rival not in alternatives:
                            alternatives.append(rival)
                            owners.append([a, word_start])
                if phrase not in alternatives:
                    alternatives.insert(0, phrase)
                    owners.insert(0, [ayah, 0])
                ids = [[adapter.vocab[u] for u in alt] for alt in alternatives]
                offsets = sorted(set([0] + [min(len(obs.posterior) - 1,
                    max(0, round((ms + adapter.lag_ms) / obs.stride_ms)))
                    for ms in (obs.positions_ms or ())]))
                scores = []
                for frame in offsets:
                    values = prefix_scores(obs.posterior[frame:], ids, adapter.blank_id)
                    target = values[alternatives.index(phrase)]
                    rival = max((s for alt, s in zip(alternatives, values) if alt != phrase), default=-math.inf)
                    scores.append({'frame': frame, 'target_p': math.exp(target),
                        'strongest_rival_p': math.exp(rival),
                        'scores': [{'owner': own, 'target': ''.join(alt), 'log_p': s}
                                   for own, alt, s in zip(owners, alternatives, values)]})
                result = {'model': model, 'surah': surah, 'ayah': ayah, 'role': role,
                    'window_ms': reads.calls[-1]['window_ms'], 'samples': round(duration * 16),
                    'pcm_sha256': actual[-1]['sha256'], 'target': ''.join(phrase),
                    'text': obs.text, 'quality': obs.unit_quality, 'event_comparisons': scores,
                    'beam': obs.hypotheses, 'runtime': adapter.runtime_identity,
                    'posterior_sha256': obs.posterior_sha256,
                    'interpretation': 'offset maxima are diagnostic, not normalized internal-event probabilities'}
                results.append(result)
                print(model, surah, ayah, duration, repr(obs.text), 'target', ''.join(phrase),
                      'max-p', max(s['target_p'] for s in scores), flush=True)
                output.write_text(json.dumps({'kind': 'fresh-local-audio', 'policy': vars(policy),
                     'results': results}, ensure_ascii=False, indent=2) + '\n')
        trace = {'total_ms': len(pcm) / 16, **reads.diagnostics()}
        verify_dispatch(actual, pcm, trace)
        (P / f'local-ledger-{model}-{surah}.json').write_text(json.dumps({
            'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
            'trace': trace, 'actual_dispatches': actual}, ensure_ascii=False, indent=2) + '\n')
    align._ONNX_CACHE.clear()
    release_runtimes()
