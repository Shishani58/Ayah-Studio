"""Real Base negatives: executor zero/wrong-owner plus known internal local starts."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from surah_cutter.core import align, candidates, decode, fast_executor as executor, model_store
from surah_cutter.core.fast_adapters import make_adapter
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, validate, verify_dispatch

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
assert Path(executor.__file__).resolve().is_relative_to(Path(roots['B2']).resolve())
model_store.ACTIVE_ID = 'whisper-tarteel-base'
bundle = align._get_onnx_bundle()
bundle.emitter.probe()
adapter = make_adapter(bundle)
pcm1 = decode.decode('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/001.mp3')
decisions = candidates.pause_pipeline_decisions(pcm1, 16000, surah=1)
results = []
for kind, pcm in [('zero', np.zeros_like(pcm1)), ('wrong_surah', pcm1)]:
    trace = {}
    started = time.monotonic()
    with dispatch_spy(bundle, pcm) as calls:
        rows = executor.align_surah_sequential(pcm, 16000, quran_text.surah_words(112),
            surah=112, bundle=bundle, trace=trace, pause_decisions=decisions)
    verify_dispatch(calls, pcm, trace)
    validate(rows, trace, [1, 2, 3, 4])
    assert all(r[2] == 0 for r in rows), (kind, rows)
    results.append({'kind': kind, 'scope': 'full FAST executor', 'rows': rows,
                    'seconds': time.monotonic() - started, 'actual_dispatches': calls, 'trace': trace})

pcm2 = decode.decode('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3')
for surah, ayah, pcm, start in [(1, 4, pcm1, 21760.), (2, 5, pcm2, 56740.)]:
    # The 1:4 read interval is taken from the frozen geometry experiment below.
    if surah == 1:
        manifest = json.loads((P.parent / '2026-10-16-model-reading-audit/geometry-inputs.json').read_text())
        case = next(c for c in manifest['cases'] if c['surah'] == 1 and c['key'].endswith('-4000'))
        start = case['window_ms'][0]
    words = quran_text.surah_words(surah)
    rendered = adapter.represent(words)
    ends = adapter.proof_word_ends(words, rendered)
    groups, word_ends = {}, {}
    for w, units, boundaries in zip(words, rendered, ends):
        base = len(groups.get(w.ayah, ()))
        groups.setdefault(w.ayah, []).extend(units)
        word_ends.setdefault(w.ayah, []).extend(base + b for b in boundaries)
    groups = {a: tuple(u) for a, u in groups.items()}
    phrase = executor.plan_phrase(groups, ayah, word_ends)['units']
    reader = LocalEvidenceReads(pcm, 16000, adapter)
    with dispatch_spy(bundle, pcm) as calls:
        obs = reader.read(start, start + 4000, reason='known_wrong_or_internal_owner_control')
    trace = reader.diagnostics()
    verify_dispatch(calls, pcm, trace)
    necessary = (executor._prefix(obs.units, phrase) == len(phrase)
        and min(obs.unit_quality[:len(phrase)], default=0.) >= .6 and not obs.truncated)
    assert not necessary, (surah, ayah, obs)
    results.append({'kind': 'wrong_or_internal_opening', 'scope': 'necessary opening checks only',
        'surah': surah, 'ayah': ayah, 'phrase': phrase, 'observation': asdict(obs),
        'actual_dispatches': calls, 'trace': trace})
release_runtimes()
(P / 'negative-audio.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print('2 full-executor negatives and 2 local internal/wrong-owner controls rejected', flush=True)
