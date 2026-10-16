"""Replay/static identification of strong but unconsumed opening observations."""
import json
from pathlib import Path

from surah_cutter.core.fast_adapters import arabic_units
from surah_cutter.core.fast_executor import Policy, plan_phrase
from surah_cutter.data.quran_text import surah_words

P = Path(__file__).resolve().parent
source = P / 'baseline-a2.json'
assert (P / 'baseline-a2.summary.json').exists()
output = []
for case in json.loads(source.read_text()):
    if not case['model'].startswith('whisper'):
        continue
    words = surah_words(case['surah'])
    groups, ends = {}, {}
    for word in words:
        groups.setdefault(word.ayah, []).extend(arabic_units(word.text))
        ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
    groups = {a: tuple(u) for a, u in groups.items()}
    plans = {a: plan_phrase(groups, a, ends) for a in groups}
    proven = {row[0] for row in case['rows'] if row[2] > 0}
    found = []
    for read in case['trace']['acoustic_reads']:
        obs = read['observed']
        units, qualities = tuple(obs['units']), obs['unit_quality']
        for ayah, plan in plans.items():
            phrase = plan['units']
            if (not plan['ambiguous'] and ayah not in proven and units[:len(phrase)] == phrase
                    and min(qualities[:len(phrase)], default=0) >= Policy().asr_quality):
                found.append({'ayah': ayah, 'read_id': read['id'], 'reason': read['reason'],
                              'text': obs['text'], 'phrase': phrase, 'truncated': obs['truncated']})
    output.append({'model': case['model'], 'surah': case['surah'], 'unconsumed': found})
    print(case['model'], case['surah'], 'unconsumed', len(found), flush=True)
(P / 'unconsumed-cues.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
