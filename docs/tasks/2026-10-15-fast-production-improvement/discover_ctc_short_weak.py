"""Read-only search for a distinct CTC case: exact short word, weak first phone."""
import json
from pathlib import Path
import re
from types import SimpleNamespace

from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_executor import Policy, plan_phrase
from surah_cutter.data.quran_text import surah_words

P = Path(__file__).resolve().parent
prior = P.parent / '2026-10-15-fast-systematic-execution'
files = [prior / 'baseline-a2.json', *prior.glob('baseline-*_al_*.json'),
         prior / 'baseline-abdul_basit_mujawwad.json']
vocab = json.loads(Path('models/wav2vec2-quran-phonetics.onnx.vocab.json').read_text())
adapter = CTCAdapter(SimpleNamespace(blank_id=37, vocab=vocab,
    emitter=SimpleNamespace(runtime_identity={'weights_sha256': 'static-no-inference'})))
found = []
for file in sorted(set(files)):
    if not file.exists() or not file.with_suffix('.summary.json').exists():
        continue
    for case in json.loads(file.read_text()):
        if not case['model'].startswith('quran-phonetics'):
            continue
        groups, ends = {}, {}
        words = surah_words(case['surah'])
        for word, units in zip(words, adapter.represent(words)):
            groups.setdefault(word.ayah, []).extend(units)
            ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
        groups = {a: tuple(u) for a, u in groups.items()}
        proved = {r[0] for r in case['rows'] if r[2] > 0}
        for read in case['trace']['acoustic_reads']:
            match = re.match(r'ayah=(\d+):pause:minimal_phrase$', read['reason'])
            if not match:
                continue
            ayah = int(match[1])
            phrase = plan_phrase(groups, ayah, ends)['units']
            obs = read['observed']
            if (ayah not in proved and len(phrase) == 3 and ends[ayah][0] == 3
                    and tuple(obs['units'][:3]) == phrase and len(obs['unit_quality']) >= 3
                    and obs['unit_quality'][0] < Policy().ctc_quality
                    and min(obs['unit_quality'][1:3]) >= Policy().ctc_quality):
                item = {'source': str(file), 'model': case['model'], 'surah': case['surah'],
                    'ayah': ayah, 'target': ''.join(phrase), 'read_id': read['id'],
                    'window_ms': read['window_ms'], 'observed': obs['text'], 'quality': obs['unit_quality']}
                found.append(item)
                print(item, flush=True)
(P / 'ctc-short-weak-discovery.json').write_text(json.dumps(found, ensure_ascii=False, indent=2) + '\n')
