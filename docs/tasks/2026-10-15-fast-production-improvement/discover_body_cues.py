"""Read-only discovery of existing strong owner cues, not fresh inference."""
import collections
import json
from pathlib import Path
import re
from types import SimpleNamespace

from surah_cutter.core.fast_adapters import TarteelAdapter
from surah_cutter.data.quran_text import surah_words

P = Path(__file__).resolve().parent
PRIOR = P.parent / '2026-10-15-fast-systematic-execution'
cases = json.loads((PRIOR / 'baseline-a2.json').read_text())
results = []
for case in cases:
    if not case['model'].startswith('whisper'):
        continue
    words = surah_words(case['surah'])
    adapter = TarteelAdapter(SimpleNamespace(emitter=SimpleNamespace(
        model_id=case['model'], runtime_identity={})))
    rendered = adapter.represent(words)
    owner_index = collections.defaultdict(list)
    for length in [1, 2]:
        for i in range(len(words) - length + 1):
            if words[i].ayah != words[i + length - 1].ayah:
                continue
            units = sum(rendered[i:i + length], ())
            owner_index[units].append(i)
    confirmed = {row[0] for row in case['rows'] if row[2] > 0}
    for read in case['trace']['acoustic_reads']:
        match = re.match(r'ayah=(\d+):', read['reason'])
        if not match or 'previous_tail' in read['reason']:
            continue
        ayah = int(match[1])
        if ayah in confirmed:
            continue
        obs = read['observed']
        if obs['truncated']:
            continue
        units, quality = tuple(obs['units']), obs['unit_quality']
        first = next(i for i, w in enumerate(words) if w.ayah == ayah)
        hits = []
        for pair, indices in owner_index.items():
            if len(indices) != 1 or indices[0] <= first or words[indices[0]].ayah != ayah:
                continue
            if len(pair) == 1 and len(pair[0]) < 4:
                continue
            for offset in range(len(units) - len(pair) + 1):
                if (units[offset:offset + len(pair)] == pair
                        and min(quality[offset:offset + len(pair)], default=0) >= .6
                        and not any('spoken_alias' in a or 'unsupported' in a
                            for a in obs.get('attribution', ())[offset:offset + len(pair)])):
                    hits.append({'word_index': indices[0], 'pair': pair, 'offset': offset,
                                 'quality': quality[offset:offset + len(pair)]})
        if hits:
            item = {'model': case['model'], 'surah': case['surah'], 'ayah': ayah,
                    'read_id': read['id'], 'reason': read['reason'], 'window_ms': read['window_ms'],
                    'expected_head': rendered[first], 'observed': units, 'quality': quality,
                    'hits': hits}
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
(P / 'body-cue-discovery.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
