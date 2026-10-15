"""Static native phonetizer variants; no acoustic runtime is loaded."""
import hashlib
import itertools
import json
from pathlib import Path

from quran_transcript import MoshafAttributes, quran_phonetizer

from surah_cutter.data import quran_text

task = Path(__file__).parent
root = task.parents[2]
reference = root / "docs/tasks/2026-10-14-fast-muaalem-local-analysis"
vocab_path = Path("/Users/ayub/.surah-cutter/models/muaalem/a57907844f7c72969b40bc526a57e9c9dbc8bf3f/muaalem/tokens.json")
vocab = {text: int(i) for i, text in json.loads(vocab_path.read_text())["outputs"]["phonemes"]["id_to_text"].items()}
text_path = root / "data/model_text_targets.json"
tables = json.loads(text_path.read_text())["surahs"]
variants, parity = {}, []
for length in (2, 4, 6):
    attrs = MoshafAttributes(rewaya="hafs", madd_monfasel_len=4,
                            madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=length)
    cases = {}
    for surah in (1, 32, 55, 59, 65, 73, 77, 80, 98, 112, 2, 79):
        words = quran_text.surah_words(surah)
        prepared = []
        for w in words:
            row = tables[str(surah)][w.index]
            if row[0] != w.text:
                raise ValueError("Canonical model text mismatch")
            prepared.append((w.ayah, row[1]))
        targets, spans = [], []
        for _, group in itertools.groupby(prepared, key=lambda item: item[0]):
            texts = [text for ayah, text in group]
            output = quran_phonetizer(" ".join(texts), attrs, sura_idx=surah, remove_spaces=True)
            base = len(targets)
            targets.extend(vocab[c] for c in output.phonemes)
            offset = 0
            for text in texts:
                positions = [m.pos for m in output.mappings[offset:offset + len(text)]
                             if not m.deleted and m.pos[1] > m.pos[0]]
                if not positions:
                    raise ValueError("Unrepresented canonical word")
                spans.append([base + min(p[0] for p in positions), base + max(p[1] for p in positions)])
                offset += len(text) + 1
        mapping = {"targets": targets, "spans": spans,
                   "words": [[w.ayah, w.index, w.text] for w in words]}
        if length == 4:
            expected = json.loads((reference / f"mapping-{surah:03d}.json").read_text())
            parity.append({"surah": surah, "equal": mapping == expected})
            if mapping != expected:
                raise ValueError(f"Native madd4 parity failed: {surah}")
        cases[surah] = mapping
    variants[length] = cases
report = {"variants": variants, "madd4_native_parity": parity,
          "canonical_sha256": hashlib.sha256(quran_text.DEFAULT_DB_PATH.read_bytes()).hexdigest(),
          "text_targets_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
          "vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
          "scope": "static model representations from existing package; no neural forwards"}
(task / "variants.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(parity, indent=2))
