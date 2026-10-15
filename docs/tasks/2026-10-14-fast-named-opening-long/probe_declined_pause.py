"""Independent local reads at a declined midpoint; do not accept a boundary."""
import ast
import hashlib
import json
from pathlib import Path
import sys

from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

task = Path(__file__).parent.resolve()
previous = task.parent / '2026-10-14-fast-pronunciation-variants'
parent = task.parent / '2026-10-14-fast-muaalem-local-analysis'
sys.path.insert(0, str(parent))
from adapter import MuaalemLocal  # noqa: E402

source = previous / 'collapse_raw_tempo_probe.py'
module = ast.parse(source.read_text())
definitions = [n for n in module.body if (isinstance(n, ast.FunctionDef) and n.name == 'collapse')
               or (isinstance(n, ast.ClassDef) and n.name == 'DurationNeutral')]
from dataclasses import replace  # noqa: E402
namespace = {'LONG': {'ا', 'ۦ', 'ۥ', 'ـ'}, 'Original': MuaalemLocal,
             'Path': Path, 'hashlib': hashlib, 'replace': replace, '__file__': str(source)}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)

record = json.loads((task / 'named-alif-2.json').read_text())[0]
pause = record['trace']['phase1_pauses'][5]
audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3')
assert hashlib.sha256(audio.read_bytes()).hexdigest() == record['audio_sha256']
pcm = decode.decode(str(audio))
model_store.ACTIVE_ID = 'muaalem'
bundle = align._get_onnx_bundle()
adapter = namespace['DurationNeutral'](bundle)
words = quran_text.surah_words(2)
rendered = adapter.represent(words)
target = tuple(u for word, units in zip(words, rendered) if word.ayah == 4 and word.index < 18
               for u in units)
assert len(target) == 20
reads = LocalEvidenceReads(pcm, 16000, adapter, cache_limit_bytes=32 * 1024 * 1024)
results = []
with dispatch_spy(bundle, pcm) as dispatches:
    for speech in [1300, 2500, 4000]:
        observation = reads.read(pause['end_ms'] - 200, pause['end_ms'] + speech,
                                 reason=f'ayah=4:declined_pause_probe:{speech}')
        matched = 0
        while matched < min(len(target), len(observation.units)) and target[matched] == observation.units[matched]:
            matched += 1
        results.append({'speech_ms': speech, 'matched': matched, 'needed': len(target),
                        'prefix_min_quality': min(observation.unit_quality[:matched], default=0),
                        'native_onset_ms': round(reads.calls[-1]['window_ms'][0] + observation.positions_ms[0])
                        if observation.positions_ms else None})
        (task / 'declined-pause-2-4.json').write_text(json.dumps({
            'audio_sha256': record['audio_sha256'], 'runtime_identity': adapter.runtime_identity,
            'pause': pause, 'target': target, 'results': results, 'reads': reads.calls,
            'actual_dispatches': dispatches, 'scope': 'diagnosis only; no accepted boundary',
        }, ensure_ascii=False, indent=2) + '\n')
        if matched == len(target) and min(observation.unit_quality[:matched]) >= .25:
            break
        if matched < 3 or min(observation.unit_quality[:matched], default=0) < .25:
            break
verify_dispatch(dispatches, pcm, {'acoustic_reads': reads.calls})
release_runtimes()
print(json.dumps(results, ensure_ascii=False, indent=2))
