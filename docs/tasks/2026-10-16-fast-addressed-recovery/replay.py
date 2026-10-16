"""Exact PCM observation replay of B2 through B3's publication-only guard."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from surah_cutter.core import candidates, decode, fast_executor as executor
from surah_cutter.core.fast_adapters import Observation, TarteelAdapter
from surah_cutter.data import quran_text

P = Path(__file__).resolve().parent
source = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else 'B3'
roots = json.loads((P / 'snapshot-paths.json').read_text())
assert Path(executor.__file__).resolve().is_relative_to(Path(roots[variant]).resolve())
summary = json.loads((P / (source + '.summary.json')).read_text())
command = json.loads((P / (source + '-command.json')).read_text())['argv']
audio_dir = Path(command[command.index('--audio-dir') + 1])
output = []
for case in json.loads((P / (source + '.json')).read_text()):
    pcm = decode.decode(str(audio_dir / f'{case["surah"]:03}.mp3'))
    assert hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest() == case['trace']['run_snapshot']['pcm_sha256']

    class Replay(TarteelAdapter):
        index = 0

        def observe(self, local, sr, *, total_samples):
            expected = case['actual_dispatches'][self.index]
            assert len(local) == expected['samples']
            assert hashlib.sha256(memoryview(np.ascontiguousarray(local))).hexdigest() == expected['sha256']
            data = copy.deepcopy(case['trace']['acoustic_reads'][self.index]['observed'])
            self.index += 1
            for name in ('units', 'unit_quality', 'attribution', 'hypotheses', 'positions_ms'):
                if data.get(name) is not None:
                    data[name] = tuple(data.get(name, ()))
            return Observation(**data)

    adapter = Replay(SimpleNamespace(emitter=SimpleNamespace(model_id=case['model'],
        runtime_identity=case['trace']['run_snapshot']['text_runtime'])))
    decisions = [candidates.PauseDecision(**{**d, 'pause': candidates.Pause(**d['pause'])})
                 for d in summary['initial_pauses'][str(case['surah'])]['decisions']]
    supplemental = [candidates.Pause(p['start_ms'], p['end_ms'], p['end_ms']-p['start_ms'])
                    for p in case['trace']['first_onset_hypotheses']['supplemental_pauses']]
    trace = {}
    with patch.object(executor, 'first_pause_hypotheses', return_value=supplemental):
        rows = executor.align_surah_sequential(pcm, 16000, quran_text.surah_words(case['surah']),
            surah=case['surah'], adapter=adapter, trace=trace, pause_decisions=decisions)
    rows = [list(r) for r in rows]
    assert adapter.index == len(case['actual_dispatches'])
    assert [r['window_ms'] for r in trace['acoustic_reads']] == [r['window_ms'] for r in case['trace']['acoustic_reads']]
    indexed = {b['ayah']: b for b in trace['boundaries']}
    changes = [{'before': a, 'after': b, 'reason': indexed[b[0]]['review_reason']}
               for a, b in zip(case['rows'], rows) if a != b]
    assert all(c['before'][:2] == c['after'][:2] and c['before'][2] > 0 and c['after'][2] == 0
               and c['reason'] == 'recovered_wide_nonstationary_midpoint_unproved' for c in changes)
    output.append({**case, 'rows': rows, 'trace': trace, 'measurement_kind': 'exact-PCM B2-observation replay; no fresh inference',
                   'actual_dispatches_source': source, 'publication_changes': changes})
    (P / (source.replace('B2', variant, 1) + '-replay.json')).write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(case['surah'], 'exact inputs/windows', 'review', [c['after'][0] for c in changes], flush=True)
