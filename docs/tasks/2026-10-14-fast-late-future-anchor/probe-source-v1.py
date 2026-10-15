"""Bounded future-only diagnostic; no current boundary or timestamp accepted."""
import ast
from pathlib import Path

base = Path(__file__).parents[1] / '2026-10-14-fast-intra-ayah-waqf/probe_previous_tail.py'
tree = ast.parse(base.read_text())
setup = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'static' for t in node.targets):
        break
    setup.append(node)
else:
    raise ValueError('Expected pure setup boundary')
exec(compile(ast.Module(body=setup, type_ignores=[]), str(base), 'exec'), globals())

from surah_cutter.core.fast_executor import plan_phrase, _beam_view, _expansion_evidence

prior = task.parent / '2026-10-14-fast-previous-body-rejection'
case = json.loads((prior / 'rejected-2.json').read_text())[0]
audit = json.loads((prior / 'rejected-2-audit/identity-mapping-002.json').read_text())
audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3')
assert hashlib.sha256(audio.read_bytes()).hexdigest() == case['audio_sha256']
pcm = decode.decode(str(audio))
model_store.ACTIVE_ID = 'muaalem'
bundle = align._get_onnx_bundle()
adapter = namespace['DurationNeutral'](bundle)
words = quran_text.surah_words(2)
rendered = adapter.represent(words)
groups, ends, weights = {}, {}, {}
for word, units, span in zip(words, rendered, adapter.mapping['spans']):
    groups.setdefault(word.ayah, []).extend(units)
    ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
    weights[word.ayah] = weights.get(word.ayah, 0) + span[1] - span[0]
groups = {a: tuple(units) for a, units in groups.items()}
identity = {int(a): tuple(texts) for a, texts in audit['canonical_words'].items()}
identity_ends = {a: list(range(1, len(texts) + 1)) for a, texts in identity.items()}
plans = {}
for ayah in [49, 50]:
    native = plan_phrase(groups, ayah, ends)
    canonical = plan_phrase(identity, ayah, identity_ends)
    count = len(canonical['units'])
    minimum = ends[ayah][count - 1] if count > 1 else 0
    size = max(len(native['units']), minimum)
    plans[ayah] = {'phrase': groups[ayah][:size], 'ambiguous': native['ambiguous'] or canonical['ambiguous']}

first_size = ends[48][0]
cue = next(r for r in case['trace']['acoustic_reads'] if r['reason'].startswith('ayah=48:')
           and tuple(r['observed']['units'][:first_size]) == groups[48][:first_size]
           and min(r['observed']['unit_quality'][:first_size]) >= .25)
pause_id = next(i for i, p in enumerate(case['trace']['phase1_pauses'])
                if abs(max(p['start_ms'], p['end_ms'] - 200) - cue['window_ms'][0]) < .1)
cue_pause = case['trace']['phase1_pauses'][pause_id]
cue_ms = cue_pause['center_ms']
anchors = [b for b in case['trace']['boundaries'] if b['confidence'] > 0 and b['ayah'] < 48]
assert anchors[-1]['ayah'] == 47
tempo = (anchors[-1]['start_ms'] - anchors[0]['start_ms']) / sum(
    weights[a] for a in weights if anchors[0]['ayah'] <= a < 47)
span = tempo * weights[48]
forecast = cue_ms + span
left, right = cue_ms + span * .2, min(len(pcm) / 16, cue_ms + span * 2.5)
eligible = [(i, p) for i, p in enumerate(case['trace']['phase1_pauses'])
            if cue_pause['end_ms'] < p['center_ms'] and left <= p['center_ms'] <= right]
selected = sorted(sorted(eligible, key=lambda ip: (abs(ip[1]['center_ms'] - forecast), ip[0]))[:8],
                  key=lambda ip: ip[1]['center_ms'])
reads = LocalEvidenceReads(pcm, 16000, adapter, cache_limit_bytes=32 * 1024 * 1024)
out = {'audio_sha256': case['audio_sha256'], 'runtime_identity': adapter.runtime_identity,
       'cue_48': {'read_id': cue['id'], 'pause_id': pause_id, 'scope': 'search only; owner48 unresolved'},
       'tempo': tempo, 'span': span, 'forecast': forecast, 'region': [left, right],
       'selected': selected, 'plans': plans, 'evaluations': [], 'found': None,
       'scope': 'future diagnostic only; no accepted current48 or boundary timestamp'}
with dispatch_spy(bundle, pcm) as dispatches:
    for index, pause in selected:
        for speech in [1300, 2500, 4000]:
            observed = reads.read(max(pause['start_ms'], pause['end_ms'] - 200), pause['end_ms'] + speech,
                                  reason=f'ayah=48:late_future_only:pause={index}:speech={speech}')
            if observed is None:
                break
            expand = False
            for ayah, plan in plans.items():
                if plan['ambiguous']:
                    continue
                phrase = tuple(plan['phrase'])
                view, beam = _beam_view(observed, phrase, .25, 2.1972245773362196)
                matched = 0
                while matched < min(len(view.units), len(phrase)) and view.units[matched] == phrase[matched]:
                    matched += 1
                quality = min(view.unit_quality[:len(phrase)], default=0)
                out['evaluations'].append({'pause_id': index, 'speech_ms': speech, 'future_ayah': ayah,
                                            'matched': matched, 'needed': len(phrase), 'min_quality': quality})
                if matched == len(phrase) and quality >= .25 and not view.truncated:
                    out['found'] = {'ayah': ayah, 'read_id': reads.last_read_id, 'phrase': phrase,
                                    'quality': quality, 'beam': beam,
                                    'geometry': case['trace']['pause_geometry'][index]}
                    break
                if _expansion_evidence(view, phrase, ends[ayah], .25, groups, ayah, ends):
                    expand = True
            out.update(reads=reads.calls, actual_dispatches=dispatches)
            (task / 'probe-48.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
            if out['found'] or not expand:
                break
        if out['found']:
            break
verify_dispatch(dispatches, pcm, {'acoustic_reads': reads.calls})
release_runtimes()
print('region', out['region'], 'selected', len(selected), 'calls', len(dispatches))
print('found', out['found'])
