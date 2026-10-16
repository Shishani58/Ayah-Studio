"""Revalidate saved published boundaries with current code; never do inference.

This proves the acceptance/evidence contract only, not scheduler equivalence.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from surah_cutter.core import candidates, fast_executor
from surah_cutter.core.fast_adapters import CTCAdapter, TarteelAdapter, loaded_implementation_sha
from surah_cutter.data import quran_text

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
output = P / 'final-revalidation2.json'
if output.exists():
    raise FileExistsError(output)
source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(ROOT.glob('surah_cutter/**/*.py'))}
records = []
for label in ('final-replay2', 'final-regression80', 'final-smoke', 'current-shatri', 'current-ghamidi'):
    path = P / (label + '.jsonl')
    for line in path.read_text().splitlines():
        item = json.loads(line)
        trace = item['trace']
        snapshot = trace['run_snapshot']
        words = quran_text.surah_words(item['surah'])
        if item['model'].startswith('whisper-tarteel-'):
            adapter = TarteelAdapter(SimpleNamespace(emitter=SimpleNamespace(
                model_id=item['model'], runtime_identity=snapshot['text_runtime'])))
        else:
            # Only the immutable canonical-text representation is needed.
            adapter = CTCAdapter.__new__(CTCAdapter)
        rendered = adapter.represent(words)
        proof_ends = (adapter.proof_word_ends(words, rendered) if hasattr(adapter, 'proof_word_ends')
                      else tuple((len(units),) for units in rendered))
        groups, word_ends, canonical_ends = {}, {}, {}
        for word, units, ends in zip(words, rendered, proof_ends, strict=True):
            base = len(groups.get(word.ayah, ()))
            groups.setdefault(word.ayah, []).extend(units)
            word_ends.setdefault(word.ayah, []).extend(base + end for end in ends)
            canonical_ends.setdefault(word.ayah, []).append(base + len(units))
        groups = {a: tuple(units) for a, units in groups.items()}
        pauses = [candidates.Pause(p['start_ms'], p['end_ms'], 0.) for p in trace['phase1_pauses']]
        pauses += [candidates.Pause(p['start_ms'], p['end_ms'], 0.)
                   for p in trace['first_onset_hypotheses']['supplemental_pauses']]
        policy = fast_executor.Policy(**snapshot['policy'])
        threshold = policy.ctc_quality if item['model'] == 'quran-phonetics' else policy.asr_quality
        canonical = sorted(groups)
        try:
            published = {b['ayah']: b['start_ms'] for b in trace['boundaries'] if b['start_ms'] is not None}
            unknown = trace['unresolved_ayahs']
            assert sorted(published.keys() | set(unknown)) == canonical
            assert not (published.keys() & set(unknown))
            assert [[a, t] for a, t, confidence in item['rows'] if confidence > 0] == list(map(list, published.items()))
            fast_executor.validate_boundaries([fast_executor.Boundary(**b) for b in trace['boundaries']],
                canonical, trace['total_ms'], trace['acoustic_reads'], groups=groups,
                word_ends=word_ends, canonical_ends=canonical_ends, threshold=threshold,
                pauses=pauses, policy=policy, pause_geometry_data=trace['pause_geometry'])
            outcome = {'outcome': 'passed'}
        except Exception as error:
            outcome = {'outcome': 'failed', 'error': type(error).__name__ + ': ' + str(error)}
        records.append({'artifact': path.name, 'line_sha256': hashlib.sha256(line.encode()).hexdigest(),
            'model': item['model'], 'reciter': item['reciter'], 'surah': item['surah'],
            'published': len(item['rows']), 'measurement_loaded_code_sha256': snapshot['implementation_sha256'],
            **outcome})
        print(label, item['model'], item['surah'], outcome, flush=True)
output.write_text(json.dumps({'scope': 'current validator; no scheduler rerun or model calls',
    'current_loaded_code_sha256': loaded_implementation_sha(), 'current_source_sha256': source,
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': records}, indent=2) + '\n')
