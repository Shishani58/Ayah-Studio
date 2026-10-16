"""Replay stored real ASR observations through E; never count as fresh calls."""
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from surah_cutter.core import candidates, decode, fast_executor
from surah_cutter.core.fast_adapters import Observation, TarteelAdapter
from surah_cutter.data import quran_text

P = Path(sys.argv[1]).resolve()
output_name = sys.argv[2] if len(sys.argv) > 2 else 'e-wide-replay'
root = Path(json.loads((P / 'snapshot-paths.json').read_text())['prototype-e']).resolve()
assert Path(fast_executor.__file__).resolve().is_relative_to(root)
labels = ['wide-prototype-c']
jobs = collections.defaultdict(list)
for label in labels:
    command = json.loads((P / f'{label}-command.json').read_text())['argv']
    audio_dir = Path(command[command.index('--audio-dir') + 1])
    summary = json.loads((P / f'{label}.summary.json').read_text())
    for case in json.loads((P / f'{label}.json').read_text()):
        if case['model'] != 'whisper-tarteel-base':
            continue
        jobs[audio_dir / f'{case["surah"]:03}.mp3'].append((label, case, summary))
results = []
for audio, subset in jobs.items():
    pcm = decode.decode(str(audio))
    for label, case, summary in subset:

        class ReplayAdapter(TarteelAdapter):
            def observe(self, local, sr, *, total_samples):
                digest = hashlib.sha256(memoryview(np.ascontiguousarray(local, dtype=np.float32))).hexdigest()
                index = self.replay_index
                expected = case['actual_dispatches'][index]
                assert digest == expected['sha256'] and len(local) == expected['samples'], (label, case['surah'], index)
                data = copy.deepcopy(case['trace']['acoustic_reads'][index]['observed'])
                self.replay_index += 1
                for field in ['units', 'unit_quality', 'attribution', 'hypotheses']:
                    data[field] = tuple(data.get(field, ()))
                if data['positions_ms'] is not None:
                    data['positions_ms'] = tuple(data['positions_ms'])
                return Observation(**data)

        adapter = ReplayAdapter(SimpleNamespace(emitter=SimpleNamespace(
            model_id=case['model'], runtime_identity=case['trace']['run_snapshot']['text_runtime'])))
        adapter.replay_index = 0
        info = summary['initial_pauses'][str(case['surah'])]
        decisions = [candidates.PauseDecision(**{**d, 'pause': candidates.Pause(**d['pause'])})
                     for d in info['decisions']]
        supplemental = [candidates.Pause(p['start_ms'], p['end_ms'], p['end_ms'] - p['start_ms'])
                        for p in case['trace']['first_onset_hypotheses']['supplemental_pauses']]
        trace = {}
        with patch.object(fast_executor, 'first_pause_hypotheses', return_value=supplemental):
            rows = fast_executor.align_surah_sequential(pcm, 16000, quran_text.surah_words(case['surah']),
                adapter=adapter, surah=case['surah'], pause_decisions=decisions, trace=trace)
        assert adapter.replay_index == len(case['actual_dispatches'])
        windows = [r['window_ms'] for r in trace['acoustic_reads']]
        old_windows = [r['window_ms'] for r in case['trace']['acoustic_reads']]
        result = {'source': label, 'model': case['model'], 'surah': case['surah'],
                  'rows_equal': [list(r) for r in rows] == case['rows'], 'windows_equal': windows == old_windows,
                  'kind': 'replay-stored-real-observations-no-fresh-inference'}
        results.append(result)
        print(result, flush=True)
        (P / f'{output_name}.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
assert len(results) == 114 and all(r['rows_equal'] and r['windows_equal'] for r in results)
