"""Run a smaller executor against independently acquired exact-PCM observations.

Only observations requested by the smaller executor are supplied. Missing crops
fail the case and require fresh inference; no nearest-window or text matching.
This is counterfactual evidence, not a fresh model run or a wall-time benchmark.
"""
import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from surah_cutter.core import align, candidates, decode, fast_executor as executor, model_store
from surah_cutter.core.fast_adapters import Observation, TarteelAdapter
from surah_cutter.data import quran_text
from surah_cutter.core.model_runtime import release_runtimes
from tools.bench_fast_models import dispatch_spy, verify_dispatch

P = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('variant')
parser.add_argument('before')
parser.add_argument('source')
parser.add_argument('output')
parser.add_argument('--fill-missing', action='store_true')
parser.add_argument('--surahs', nargs='+', type=int)
args = parser.parse_args()
roots = json.loads((P / 'snapshot-paths.json').read_text())
assert Path(executor.__file__).resolve().is_relative_to(Path(roots[args.variant]).resolve())
implementation = hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest()
summary = json.loads((P / (args.before + '.summary.json')).read_text())


def stream_records(path):
    text = path.read_text()
    decoder, offset = json.JSONDecoder(), 1
    while offset < len(text):
        while offset < len(text) and text[offset] in ' \r\n\t,':
            offset += 1
        if text[offset] == ']':
            break
        item, offset = decoder.raw_decode(text, offset)
        yield item


baseline = stream_records(P / (args.before + '.json'))
source = iter(stream_records(P / (args.source + '.json')))
next_source = next(source, None)
command = json.loads((P / (args.before + '-command.json')).read_text())['argv']
audio_dir = Path(command[command.index('--audio-dir') + 1])
reference = Path(command[command.index('--reference') + 1])
reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
real_bundle = real_adapter = None
if args.fill_missing:
    model_store.ACTIVE_ID = 'whisper-tarteel-base'
    real_bundle = align.default_bundle()
    real_bundle.emitter.probe()
    real_adapter = TarteelAdapter(real_bundle)
count, expected, missing = 0, 0, []
jsonl = P / (args.output + '.jsonl')
assert not jsonl.exists()
jsonl.write_text('')
for before in baseline:
    if args.surahs and before['surah'] not in args.surahs:
        continue
    expected += 1
    while next_source is not None and next_source['surah'] < before['surah']:
        next_source = next(source, None)
    case = next_source if next_source is not None and next_source['surah'] == before['surah'] else before
    assert before['audio_sha256'] == case['audio_sha256']
    assert before['initial_pauses_sha256'] == case['initial_pauses_sha256']
    assert before['trace']['run_snapshot']['text_runtime'] == case['trace']['run_snapshot']['text_runtime']
    observations = {}
    for source_label, original in ((args.before, before), (args.source, case)):
        for dispatch, read in zip(original['actual_dispatches'], original['trace']['acoustic_reads'], strict=True):
            observations.setdefault((dispatch['samples'], dispatch['sha256']),
                (read['observed'], read['seconds'], {'kind': 'saved_exact_PCM',
                    'recording': original['surah'], 'source': source_label, 'read_id': read['id']}))
    pcm = None
    for folder in Path(tempfile.gettempdir()).glob('local-fast-bench-*'):
        cached = folder / f'{case["surah"]}.npy'
        if cached.exists():
            data = np.load(cached, mmap_mode='r', allow_pickle=False)
            if hashlib.sha256(memoryview(np.ascontiguousarray(data))).hexdigest() == case['trace']['run_snapshot']['pcm_sha256']:
                pcm = data
                break
    if pcm is None:
        pcm = decode.decode(str(audio_dir / f'{case["surah"]:03}.mp3'))
    assert hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest() == case['trace']['run_snapshot']['pcm_sha256']
    dispatched, inference_costs, provenance = [], [], []

    class Replay(TarteelAdapter):
        def observe(self, local, sr, *, total_samples):
            digest = hashlib.sha256(memoryview(np.ascontiguousarray(local))).hexdigest()
            key = (len(local), digest)
            if key not in observations:
                if real_adapter is None:
                    raise LookupError('missing_exact_pcm_observation:' + digest)
                started = monotonic()
                fresh = real_adapter.observe(local, sr, total_samples=total_samples)
                observations[key] = (asdict(fresh), monotonic() - started,
                    {'kind': 'fresh_model_read', 'fresh_dispatch_index': len(fresh_dispatches) - 1})
            observed, seconds, origin = observations[key]
            dispatched.append({'samples': len(local), 'sha256': digest})
            inference_costs.append(seconds)
            provenance.append(origin)
            data = copy.deepcopy(observed)
            for name in ('units', 'unit_quality', 'attribution', 'hypotheses', 'positions_ms'):
                if data.get(name) is not None:
                    data[name] = tuple(data.get(name, ()))
            return Observation(**data)

    adapter = Replay(real_bundle or SimpleNamespace(emitter=SimpleNamespace(model_id=case['model'],
        runtime_identity=case['trace']['run_snapshot']['text_runtime'])))
    if real_adapter is not None:
        assert adapter.runtime_identity == case['trace']['run_snapshot']['text_runtime']
    decisions = [candidates.PauseDecision(**{**d, 'pause': candidates.Pause(**d['pause'])})
                 for d in summary['initial_pauses'][str(case['surah'])]['decisions']]
    supplemental = [candidates.Pause(p['start_ms'], p['end_ms'], p['end_ms'] - p['start_ms'])
                    for p in case['trace']['first_onset_hypotheses']['supplemental_pauses']]
    trace = {}
    try:
        with (dispatch_spy(real_bundle, pcm) if real_bundle is not None else nullcontext([])) as fresh_dispatches, \
                patch.object(executor, 'first_pause_hypotheses', return_value=supplemental):
            rows = executor.align_surah_sequential(pcm, 16000, quran_text.surah_words(case['surah']),
                surah=case['surah'], adapter=adapter, trace=trace, pause_decisions=decisions)
    except LookupError as error:
        missing.append({'surah': case['surah'], 'reason': str(error),
                        'last_event': trace['transitions'][-2]})
        print('missing', case['surah'], str(error), flush=True)
        del pcm
        continue
    # Predictions are complete; reference is used only for this evaluation.
    verify_dispatch(dispatched, pcm, trace)
    for actual, origin in zip(dispatched, provenance, strict=True):
        if origin['kind'] == 'fresh_model_read':
            assert actual == fresh_dispatches[origin['fresh_dispatch_index']]
    with sqlite3.connect(f'file:{reference.resolve()}?mode=ro', uri=True) as conn:
        truth = dict(conn.execute('SELECT ayah,time FROM timings WHERE sura=?', (case['surah'],)))
    errors = [abs(ms - truth[a]) for a, ms, _ in rows]
    result = {**case, 'rows': rows, 'errors_ms': errors, 'reference_sha256': reference_sha,
        'trace': trace, 'actual_dispatches': dispatched,
        'measurement_kind': ('hybrid exact-PCM reuse plus genuine missing model observations; not whole fresh inference or time benchmark'
                             if args.fill_missing else 'counterfactual exact-PCM observation reuse; not fresh inference or time benchmark'),
        'fresh_dispatches': fresh_dispatches, 'observation_provenance': provenance,
        'reused_observation_sources': [args.before, args.source],
        'sum_original_inference_seconds': sum(inference_costs)}
    with jsonl.open('a') as log:
        log.write(json.dumps(result, ensure_ascii=False) + '\n')
    count += 1
    print(case['surah'], 'proofs', sum(r[2] > 0 for r in rows), flush=True)
    del pcm, result
assert reference_sha == hashlib.sha256(reference.read_bytes()).hexdigest()
assert implementation == hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest()
release_runtimes()
with (P / (args.output + '.json')).open('w') as target:
    target.write('[\n')
    with jsonl.open() as log:
        for index, line in enumerate(log):
            target.write((',' if index else '') + line.rstrip() + '\n')
    target.write(']\n')
(P / (args.output + '-status.json')).write_text(json.dumps({'complete': count,
    'expected': expected, 'missing': missing, 'source_sha256': implementation}, indent=2) + '\n')
print('complete', count, 'missing', missing, flush=True)
