"""Discover declared first alif-name mappings from installed package, statically."""
import hashlib
import json
from pathlib import Path

from quran_transcript import MoshafAttributes, quran_phonetizer
from quran_transcript.alphabet import uthmani

from surah_cutter.data import quran_text

task = Path(__file__).parent.resolve()
root = task.parents[2]
targets_path = root / 'data/model_text_targets.json'
tables = json.loads(targets_path.read_text())['surahs']
attributes = MoshafAttributes(rewaya='hafs', madd_monfasel_len=4,
                              madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=4)
cases = []
for surah in range(1, 115):
    words = [w for w in quran_text.surah_words(surah) if w.ayah == 1]
    texts = [tables[str(surah)][w.index][1] for w in words]
    # Membership in the native disconnected-letter expansion vocabulary avoids
    # mistaking an ordinary word beginning with alif for a named letter.
    if texts[0] not in uthmani.hrof_moqtaa_disassemble or texts[0][0] != 'ا':
        continue
    result = quran_phonetizer(' '.join(texts), attributes, sura_idx=surah, remove_spaces=True)
    mappings = result.mappings[:len(texts[0])]
    positions = [m.pos for m in mappings if not m.deleted and m.pos[1] > m.pos[0]]
    left, right = min(p[0] for p in positions), max(p[1] for p in positions)
    first = mappings[0]
    name = result.phonemes[first.pos[0]:first.pos[1]]
    assert left == first.pos[0] == 0 and not first.deleted and name == 'ءَلِف'
    cases.append({'surah': surah, 'canonical_first_word': words[0].text,
                  'model_first_word': texts[0], 'word_phone_span': [left, right],
                  'word_phones': result.phonemes[left:right],
                  'first_character_name_span': list(first.pos), 'first_character_name': name,
                  'native_expansion': uthmani.hrof_moqtaa_disassemble[texts[0]],
                  'scope': 'whole declared first alif-name; no arbitrary character boundaries'})
report = {'cases': cases, 'scope': 'static only; no neural forwards',
          'canonical_sha256': hashlib.sha256(quran_text.DEFAULT_DB_PATH.read_bytes()).hexdigest(),
          'model_text_sha256': hashlib.sha256(targets_path.read_bytes()).hexdigest()}
(task / 'all-alif-declarations.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('declared alif opening surahs:', [c['surah'] for c in cases])
