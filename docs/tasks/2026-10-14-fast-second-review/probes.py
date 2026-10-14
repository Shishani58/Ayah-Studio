"""Read-only review probes. No inference, model selection, or database writes."""
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

from surah_cutter.core.fast_adapters import TarteelAdapter, arabic_units
from surah_cutter.core.fast_executor import plan_phrase
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.core.model_worker import whisper_alignment_text
from surah_cutter.data import quran_text

ROOT = Path(__file__).parent
helpers = runpy.run_path('tests/test_fast_executor.py')
Fake, run, observation, pause = (helpers[k] for k in ('Fake', 'run', 'observation', 'pause'))
result = {}

def respond(fake, start, duration):
    phrase = fake.phrase(1)
    return observation(('wrong-first-unit',) + phrase[1:] if duration < 2000 else phrase)

fake = Fake(respond)
rows, trace = run(fake, [pause(1000)])
result['first_word_error_no_expansion'] = {
    'calls': fake.calls, 'confirmed': sum(r[2] > 0 for r in rows),
    'expansions': [e for e in trace['transitions'] if e['state'] == 'EXPAND_LOCAL'],
}
assert not result['first_word_error_no_expansion']['expansions']
assert all(samples <= 24000 for _, samples in fake.calls)

fake = Fake(lambda f, start, duration: observation(f.phrase(1) if start in (1000, 8000) else ()))
rows, trace = run(fake, [pause(1000, 7000)])
result['long_pause'] = {'calls': fake.calls, 'confirmed': sum(r[2] > 0 for r in rows),
                        'reason': trace['boundaries'][0]['review_reason']}
assert not any(start == 1000 for start, samples in fake.calls)
assert result['long_pause']['confirmed'] == 0

class Emitter:
    model_id = 'whisper-tarteel-tiny'
    runtime_versions = {'torch': 'version-A'}
    def recognize_local(self, pcm, *, total_samples):
        return {'text': 'قل هو الله احد ضعيف', 'token_confidence': .98,
                'min_token_confidence': .1, 'truncated': False}

adapter = TarteelAdapter(SimpleNamespace(emitter=Emitter()))
obs = adapter.observe(None, 16000, total_samples=99999)
result['tail_quality_poison'] = {'units': obs.units, 'unit_quality': obs.unit_quality,
                               'mean': obs.quality}
assert obs.unit_quality[0] == .1
first_word = quran_text.surah_words(2)[0]
result['opening_letters'] = {
    'canonical': first_word.text,
    'fast': inference_words([first_word], 'whisper-tarteel-tiny'),
    'spoken': whisper_alignment_text(first_word.text),
}
assert arabic_units(result['opening_letters']['fast'][0]) != arabic_units(result['opening_letters']['spoken'])

result['canonical_repeated_phrases'] = {}
for surah in (55, 77, 109):
    words = quran_text.surah_words(surah)
    groups, ends = {}, {}
    for word in words:
        groups.setdefault(word.ayah, []).extend(arabic_units(word.text))
        ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
    groups = {a: tuple(u) for a, u in groups.items()}
    ambiguous = {}
    for ayah in groups:
        planned = plan_phrase(groups, ayah, ends)
        if planned['ambiguous']:
            ambiguous[ayah] = {'units': planned['units'], 'collisions': planned['collisions'][:4]}
    result['canonical_repeated_phrases'][surah] = ambiguous
assert '3' in json.loads(json.dumps(result['canonical_repeated_phrases'][109]))

new = json.loads(Path('docs/tasks/2026-10-14-fast-full-execution/corpus-final.json').read_text())
old = json.loads(Path('docs/tasks/2026-10-14-fast-full-execution/corpus-legacy.json').read_text())
assert len({(i['model'], i['surah']) for i in new}) == len(new) == 120
assert len({(i['model'], i['surah']) for i in old}) == len(old) == 111
previous = {(i['model'], i['surah']): i for i in old}
result['paired'] = {}
for model in sorted({i['model'] for i in new}):
    metrics = []
    for data in (old, new):
        items = [i for i in data if i['model'] == model and i['surah'] >= 78]
        confirmed = [(i, a, error) for i in items
                     for (a, ms, confidence), error in zip(i['rows'], i['errors_ms'])
                     if confidence > 0 and a not in i['trace']['review_required_ayahs']]
        metrics.append({'records': len(items), 'ayahs': sum(len(i['rows']) for i in items),
                        'confirmed': len(confirmed), 'over500': sum(e > 500 for i, a, e in confirmed)})
    result['paired'][model] = {'legacy': metrics[0], 'new': metrics[1]}
    for item in [i for i in new if i['model'] == model and i['surah'] >= 78]:
        prior = previous[model, item['surah']]
        assert all(item[k] == prior[k] for k in ('audio_sha256', 'initial_pauses_sha256', 'reference_sha256'))
result['real_examples'] = []
for item in new:
    if item['surah'] in (2, 9, 78, 109):
        trace = item['trace']
        result['real_examples'].append({'model': item['model'], 'surah': item['surah'],
            'confirmed': sum(r[2] > 0 for r in item['rows']),
            'stop': next((e for e in trace['transitions'] if e['state'] == 'REVIEW_REQUIRED'), None),
            'coverage': trace['coverage'], 'denied': len(trace['denied_acoustic_reads'])})
replay = json.loads(Path('docs/tasks/2026-10-14-fast-full-execution/replay.json').read_text())
executor = Path('surah_cutter/core/fast_executor.py')
result['replay_current_source_matches'] = replay['current_source_sha256'] == hashlib.sha256(executor.read_bytes()).hexdigest()
assert result['replay_current_source_matches']
(ROOT / 'probes.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('canonical_repeated_phrases', 'real_examples')}, ensure_ascii=False, indent=2))
