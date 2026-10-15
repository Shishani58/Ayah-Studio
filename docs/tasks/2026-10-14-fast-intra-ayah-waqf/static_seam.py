"""Native connected and waqf prefix representations, with canonical seam audit."""
import ast
import hashlib
import json
from pathlib import Path

from quran_transcript import MoshafAttributes, quran_phonetizer

from surah_cutter.core.fast_executor import plan_phrase, plan_seam
from surah_cutter.data import quran_text

task = Path(__file__).parent.resolve()
root = task.parents[2]
previous = task.parent / '2026-10-14-fast-pronunciation-variants'
audit_path = task.parent / '2026-10-14-fast-named-opening-long/native-onset-2-audit/identity-mapping-002.json'
audit = json.loads(audit_path.read_text())
mapping = audit['phonetic_mapping']
vocab_path = Path('/Users/ayub/.surah-cutter/models/muaalem/a57907844f7c72969b40bc526a57e9c9dbc8bf3f/muaalem/tokens.json')
vocab = {int(i): text for i, text in json.loads(vocab_path.read_text())['outputs']['phonemes']['id_to_text'].items()}
tree = ast.parse((previous / 'collapse_raw_tempo_probe.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collapse')
namespace = {'LONG': {'ا', 'ۦ', 'ۥ', 'ـ'}}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<retained-collapse>', 'exec'), namespace)
collapse = namespace['collapse']
words = quran_text.surah_words(2)
assert [[w.ayah, w.index, w.text] for w in words] == mapping['words']
groups, ends, normalized_words = {}, {}, []
for word, (start, end) in zip(words, mapping['spans']):
    units = collapse(tuple(vocab[token] for token in mapping['targets'][start:end]))[0]
    normalized_words.append(units)
    groups.setdefault(word.ayah, []).extend(units)
    ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
groups = {a: tuple(u) for a, u in groups.items()}
canonical = {int(a): tuple(texts) for a, texts in audit['canonical_words'].items()}
canonical_ends = {a: list(range(1, len(texts) + 1)) for a, texts in canonical.items()}
seam = plan_seam(groups, 18, ends)
canonical_seam = plan_seam(canonical, 18, canonical_ends)
canonical_opening = plan_phrase(canonical, 18, canonical_ends)
target_path = root / 'data/model_text_targets.json'
tables = json.loads(target_path.read_text())['surahs']['2']
texts = [tables[w.index][1] for w in words if w.ayah == 18]
attributes = MoshafAttributes(rewaya='hafs', madd_monfasel_len=4,
                              madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=4)
full = quran_phonetizer(' '.join(texts), attributes, sura_idx=2, remove_spaces=True)
native_full = ''.join(vocab[token] for word, (start, end) in zip(words, mapping['spans'])
                      if word.ayah == 18 for token in mapping['targets'][start:end])
assert full.phonemes == native_full
variants = []
for count in [1, 2, 3, 6]:
    text = ' '.join(texts[:count])
    phon = quran_phonetizer(text, attributes, sura_idx=2, remove_spaces=True)
    variants.append({'word_count': count, 'model_text': text, 'raw_phones': phon.phonemes,
                     'normalized_phones': collapse(tuple(phon.phonemes))[0],
                     'source_mapping': [{'char': char, 'pos': m.pos, 'deleted': m.deleted}
                                        for char, m in zip(text, phon.mappings)]})
result = {'canonical_opening': canonical_opening, 'canonical_seam': canonical_seam,
          'phonetic_seam': seam, 'connected_word_phones': normalized_words[180:186],
          'waqf_variants': variants, 'native_full_parity': True,
          'canonical_sha256': hashlib.sha256(quran_text.DEFAULT_DB_PATH.read_bytes()).hexdigest(),
          'model_text_sha256': hashlib.sha256(target_path.read_bytes()).hexdigest(),
          'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
          'scope': 'static only; no neural forwards; canonical text unchanged'}
(task / 'static-seam-2-18.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('canonical opening:', canonical_opening['units'])
print('canonical seam:', canonical_seam)
print('phonetic seam:', seam)
for variant in variants:
    print('waqf', variant['word_count'], variant['model_text'], ''.join(variant['normalized_phones']))
