"""One context-authorized longer opening; no acceptance or lower quality gate."""
import ast
from pathlib import Path

base = Path(__file__).with_name('probe_previous_tail.py')
tree = ast.parse(base.read_text())
setup = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'static' for t in node.targets):
        break
    setup.append(node)
else:
    raise ValueError('Expected pure adapter setup boundary')
exec(compile(ast.Module(body=setup, type_ignores=[]), str(base), 'exec'), globals())

static = json.loads((task / 'static-seam-2-18.json').read_text())  # noqa: F821
context = json.loads((task / 'previous-tail-2-18.json').read_text())  # noqa: F821
assert context['fresh_context_results'][-1]['whole_tail_match']
assert context['fresh_context_results'][-1]['min_quality'] >= .25
case_path = task.parent / '2026-10-14-fast-named-opening-long/native-onset-2.json'  # noqa: F821
case = json.loads(case_path.read_text())[0]  # noqa: F821
audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3')
assert hashlib.sha256(audio.read_bytes()).hexdigest() == case['audio_sha256']  # noqa: F821
pcm = decode.decode(str(audio))  # noqa: F821
model_store.ACTIVE_ID = 'muaalem'  # noqa: F821
bundle = align._get_onnx_bundle()  # noqa: F821
adapter = namespace['DurationNeutral'](bundle)  # noqa: F821
adapter.represent(quran_text.surah_words(2))  # noqa: F821
target = tuple(static['connected_word_phones'][0])
reads = LocalEvidenceReads(pcm, 16000, adapter, cache_limit_bytes=32 * 1024 * 1024)  # noqa: F821
with dispatch_spy(bundle, pcm) as dispatches:  # noqa: F821
    observed = reads.read(253700, 257900, reason='ayah=18:heard_previous_context_extended_opening')
    matched = 0
    while matched < min(len(target), len(observed.units)) and target[matched] == observed.units[matched]:
        matched += 1
    record = {'audio_sha256': case['audio_sha256'], 'runtime_identity': adapter.runtime_identity,
              'connected_target': target, 'matched': matched, 'needed': len(target),
              'prefix_min_quality': min(observed.unit_quality[:matched], default=0),
              'reads': reads.calls, 'actual_dispatches': dispatches,
              'source_setup_sha256': hashlib.sha256(base.read_bytes()).hexdigest(),  # noqa: F821
              'scope': 'one fresh extended opening; no boundary accepted; prior context retained separately'}
verify_dispatch(dispatches, pcm, {'acoustic_reads': reads.calls})  # noqa: F821
(task / 'extended-opening-2-18.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')  # noqa: F821
release_runtimes()  # noqa: F821
print('connected target matched', matched, '/', len(target))
print('observed', observed.units)
