"""Attribute selected BPE bytes to spelling prefixes; never lower whole-word proof quality."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(roots['B5']).with_name('B6')
assert not root.exists()
shutil.copytree(roots['B5'], root, ignore=shutil.ignore_patterns('__pycache__'))
adapter_file = root / 'surah_cutter/core/fast_adapters.py'
code = adapter_file.read_text()
code = code.replace('word_owners, raw_words, current, raw_word, byte_offset = [], [], set(), [], 0',
    'word_owners, raw_words, current, raw_word, byte_offset = [], [], set(), [], 0\n        word_letters, letters = [], []', 1)
code = code.replace('                        current, raw_word = set(), []',
    '                        word_letters.append(letters)\n                        current, raw_word, letters = set(), [], []', 1)
needle = '                    raw_word.append(c)\n'
assert needle in code
code = code.replace(needle, '''                    raw_word.append(c)
                    # A selected token can span letters/words or split UTF-8.
                    # Use all contributing bytes, and attach combining marks
                    # to their preceding letter. These are correlated minima,
                    # not independent character votes or a new proof score.
                    q = min(probabilities[i] for i in contributing)
                    if unicodedata.category(c).startswith("L"):
                        letters.extend(q for unit in arabic_units(c) for _ in unit)
                    elif letters:
                        letters[-1] = min(letters[-1], q)
''', 1)
code = code.replace('            raw_words.append("".join(raw_word))\n        if len(word_owners)',
    '            raw_words.append("".join(raw_word))\n            word_letters.append(letters)\n        if len(word_owners)', 1)
code = code.replace('                             "probabilities": [probabilities[i] for i in ids]}\n                            for word, raw, ids in zip(units, raw_words, word_owners))',
    '                             "probabilities": [probabilities[i] for i in ids],\n                             "letter_quality": scores if len(scores) == len(word) else []}\n                            for word, raw, ids, scores in zip(units, raw_words, word_owners, word_letters))', 1)
code = code.replace('tarteel-independent-words-v4-raw-alias-ownership',
                    'tarteel-independent-words-v5-selected-letter-cues')
compile(code, str(adapter_file), 'exec')
adapter_file.write_text(code)
file = root / 'surah_cutter/core/fast_executor.py'
code = file.read_text()
needle = 'def _partial_asr_head('
helper = '''def _asr_head_cue_quality(observation):
    """Min selected-token probability for three heard letters, cue only."""
    if len(observation.units) != 1 or len(observation.unit_quality) != 1:
        return {"metric": "whole_word", "quality": 0.}
    q = observation.unit_quality[0]
    if not math.isfinite(q) or not 0 <= q <= 1:
        return {"metric": "whole_word", "quality": 0.}
    if len(observation.attribution) == 1:
        item = observation.attribution[0]
        scores = item.get("letter_quality", ())
        if (item.get("word") == observation.units[0] and len(scores) == len(observation.units[0])
                and len(scores) >= 3 and all(math.isfinite(x) and 0 <= x <= 1 for x in scores)):
            return {"metric": "first_three_letters_selected_token_min", "quality": min(scores[:3]),
                    "whole_word_quality": q}
    return {"metric": "whole_word", "quality": q}


'''
assert needle in code
code = code.replace(needle, helper + needle, 1)
code = code.replace('or observation.unit_quality[0] < threshold\n            or any("unsupported"',
    'or _asr_head_cue_quality(observation)["quality"] < threshold\n            or any("unsupported"', 1)
code = code.replace('or min(observation.unit_quality, default=0.) < threshold):\n        return None\n    rivals =',
    'or (not asr_cue and min(observation.unit_quality, default=0.) < threshold)\n            or (asr_cue and _asr_head_cue_quality(observation)["quality"] < threshold)):\n        return None\n    rivals =', 1)
code = code.replace('"complete_competitor_owners": rivals}',
    '"complete_competitor_owners": rivals,\n            **({"cue_quality": _asr_head_cue_quality(observation)} if asr_cue else {})}', 1)
compile(code, str(file), 'exec')
file.write_text(code)
roots['B6'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(roots, indent=2) + '\n')
(P / 'B6-sha.json').write_text(json.dumps({'sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in (file, adapter_file)}, 'proof_quality_unchanged': 'all selected tokens per whole essential word >=0.6',
    'cue_quality': 'first3 independently decoded letters incl following diacritics, selected BPE minima >=0.6',
    'control_results_seen': False}, indent=2) + '\n')
