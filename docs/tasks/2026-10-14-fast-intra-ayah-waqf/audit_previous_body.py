"""Offline identify a fully decoded previous-ayah body behind a shared first word."""
import ast
import json
from pathlib import Path

task = Path(__file__).parent.resolve()
previous = task.parent / '2026-10-14-fast-pronunciation-variants'
mapping = json.loads((task / 'context-2-audit/identity-mapping-002.json').read_text())['phonetic_mapping']
vocab_path = Path('/Users/ayub/.surah-cutter/models/muaalem/a57907844f7c72969b40bc526a57e9c9dbc8bf3f/muaalem/tokens.json')
vocab = {int(i): text for i, text in json.loads(vocab_path.read_text())['outputs']['phonemes']['id_to_text'].items()}
tree = ast.parse((previous / 'collapse_raw_tempo_probe.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collapse')
namespace = {'LONG': {'ا', 'ۦ', 'ۥ', 'ـ'}}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<retained-collapse>', 'exec'), namespace)
collapse = namespace['collapse']
groups, ends, canonical = {}, {}, {}
for (ayah, index, text), (left, right) in zip(mapping['words'], mapping['spans']):
    groups.setdefault(ayah, []).extend(collapse(tuple(vocab[t] for t in mapping['targets'][left:right]))[0])
    ends.setdefault(ayah, []).append(len(groups[ayah]))
    canonical.setdefault(ayah, []).append((index, text))
records = json.loads((task / '2-42-stop-evidence.json').read_text())['reads']
first = tuple(groups[42][:ends[42][0]])
findings = []
for read in records:
    units = tuple(read['observed']['units'])
    candidates = []
    for word_index in range(1, len(ends[41]) - 1):
        start = ends[41][word_index - 1]
        first_end, second_end = ends[41][word_index:word_index + 2]
        alternative = tuple(groups[41][start:second_end])
        if tuple(groups[41][start:first_end]) != first or units[:len(alternative)] != alternative:
            continue
        quality = min(read['observed']['unit_quality'][:len(alternative)])
        if quality < .25:
            continue
        candidates.append({'owner_ayah': 41, 'start_unit': start,
                           'canonical_words': canonical[41][word_index:word_index + 2],
                           'whole_two_word_units': alternative, 'min_quality': quality})
    findings.append({'read_id': read['id'], 'window': read['window_ms'], 'rivals': candidates})
result = {'requested_ayah': 42, 'previous_confirmed_ayah': 41,
          'requested_words': canonical[42][:2], 'shared_first_word_units': first,
          'findings': findings,
          'scope': 'offline evidence only; a shared first word never established the requested owner'}
(task / '2-42-previous-body-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
for finding in findings:
    if finding['rivals']:
        print(finding['read_id'], finding['window'], finding['rivals'][0]['canonical_words'],
              finding['rivals'][0]['min_quality'])
