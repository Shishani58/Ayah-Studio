"""Replay saved selected tokens to check exact word-score preservation, no new inference."""
import ast
import json
import math
from pathlib import Path
import unicodedata

from surah_cutter.core.fast_adapters import attributed_asr_units

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
source = (Path(roots['A']) / 'surah_cutter/core/fast_adapters.py').read_text()
tree = ast.parse(source)
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('arabic_units', 'attributed_asr_units')]
namespace = {'math': math, 'unicodedata': unicodedata}
exec(compile(ast.Module(nodes, type_ignores=[]), '<frozen-A-word-attribution>', 'exec'), namespace)
old = namespace['attributed_asr_units']
checked = []
for label in ('deep', 'geometry', 'recovery'):
    items = json.loads((P.parent / '2026-10-16-model-reading-audit' / (label + '-results.json')).read_text())
    for case in items:
        for width, readings in case['readings'].items():
            for rank, reading in enumerate(readings):
                result = {'text': reading['text'], 'selected_tokens': reading['selected_tokens']}
                before = old(result)
                after = attributed_asr_units(result)
                assert before[:2] == after[:2]
                assert before[2] == tuple({k: v for k, v in item.items() if k != 'letter_quality'} for item in after[2])
                checked.append({'key': case['key'], 'width': width, 'rank': rank,
                    'same_units_word_quality_and_existing_attribution': True,
                    'new_letter_quality': [a.get('letter_quality') for a in after[2]]})
(P / 'letter-attribution-replay.json').write_text(json.dumps({'kind': 'saved token replay, no inference',
    'checked': len(checked), 'results': checked}, ensure_ascii=False, indent=2) + '\n')
print(len(checked), 'selected-token outputs: word units/quality/old attribution exactly unchanged')
