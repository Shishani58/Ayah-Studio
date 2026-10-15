"""Read minimal separate previous context; replay the retained opening proof."""
import ast
from dataclasses import replace
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
tree = ast.parse(source.read_text())
definitions = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == 'collapse')
               or (isinstance(n, ast.ClassDef) and n.name == 'DurationNeutral')]
namespace = {'LONG': {'ا', 'ۦ', 'ۥ', 'ـ'}, 'Original': MuaalemLocal, 'Path': Path,
             'hashlib': hashlib, 'replace': replace, '__file__': str(source)}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
static = json.loads((task / 'static-seam-2-18.json').read_text())
case_path = task.parent / '2026-10-14-fast-named-opening-long/native-onset-2.json'
case = json.loads(case_path.read_text())[0]
pause = case['trace']['phase1_pauses'][31]
opening = case['trace']['acoustic_reads'][67]
variant = static['waqf_variants'][0]
head = tuple(variant['normalized_phones'])
assert tuple(opening['observed']['units'][:len(head)]) == head
head_quality = min(opening['observed']['unit_quality'][:len(head)])
assert head_quality >= .25
audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3')
assert hashlib.sha256(audio.read_bytes()).hexdigest() == case['audio_sha256']
pcm = decode.decode(str(audio))
model_store.ACTIVE_ID = 'muaalem'
bundle = align._get_onnx_bundle()
adapter = namespace['DurationNeutral'](bundle)
adapter.represent(quran_text.surah_words(2))
tail = tuple(static['phonetic_seam']['tail'])
reads = LocalEvidenceReads(pcm, 16000, adapter, cache_limit_bytes=32 * 1024 * 1024)
results = []
with dispatch_spy(bundle, pcm) as dispatches:
    for length in [1300, 2500, 4000]:
        observed = reads.read(pause['start_ms'] - length, pause['start_ms'],
                              reason=f'ayah=18:previous_tail_probe:{length}')
        complete = tuple(observed.units[-len(tail):]) == tail
        quality = min(observed.unit_quality[-len(tail):], default=0.) if complete else None
        results.append({'length_ms': length, 'whole_tail_match': complete, 'min_quality': quality})
        (task / 'previous-tail-2-18.json').write_text(json.dumps({
            'runtime_identity': adapter.runtime_identity, 'audio_sha256': case['audio_sha256'],
            'pause': pause, 'tail': tail, 'fresh_context_results': results,
            'reads': reads.calls, 'actual_dispatches': dispatches,
            'retained_opening': opening, 'retained_source_sha256': hashlib.sha256(case_path.read_bytes()).hexdigest(),
            'waqf_head': head, 'head_min_quality': head_quality,
            'scope': 'fresh previous context; opening is retained replay, not fresh inference; no boundary accepted',
        }, ensure_ascii=False, indent=2) + '\n')
        if complete and quality >= .25:
            break
verify_dispatch(dispatches, pcm, {'acoustic_reads': reads.calls})
release_runtimes()
print(json.dumps(results, indent=2))
