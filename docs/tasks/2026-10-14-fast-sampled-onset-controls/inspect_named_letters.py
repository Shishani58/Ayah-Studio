"""Static original-character ownership for named opening letters; no audio."""
import hashlib
import json
from pathlib import Path

from quran_transcript import MoshafAttributes, quran_phonetizer

from surah_cutter.data import quran_text

task = Path(__file__).parent.resolve()
root = task.parents[2]
path = root / 'data/model_text_targets.json'
tables = json.loads(path.read_text())['surahs']
attrs = MoshafAttributes(rewaya='hafs', madd_monfasel_len=4, madd_mottasel_len=4,
                        madd_mottasel_waqf=4, madd_aared_len=4)
cases = []
for surah in [2, 32]:
    words = quran_text.surah_words(surah)
    texts = [tables[str(surah)][w.index][1] for w in words if w.ayah == 1]
    text = ' '.join(texts)
    phon = quran_phonetizer(text, attrs, sura_idx=surah, remove_spaces=True)
    cases.append({'surah': surah, 'canonical_words': [w.text for w in words if w.ayah == 1],
                  'model_text': text, 'phones': phon.phonemes,
                  'mapping': [{'char': ch, 'pos': m.pos, 'deleted': m.deleted,
                               'phones': phon.phonemes[m.pos[0]:m.pos[1]]}
                              for ch, m in zip(text, phon.mappings)]})
result = {'cases': cases, 'text_targets_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'canonical_sha256': hashlib.sha256(quran_text.DEFAULT_DB_PATH.read_bytes()).hexdigest(),
          'scope': 'static package character mappings only; no neural forwards'}
(task / 'named-letter-mappings.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
for case in cases:
    print(case['surah'], case['model_text'], case['phones'])
    print(case['mapping'])
