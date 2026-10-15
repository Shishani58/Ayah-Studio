"""Swap only static verified phones in an isolated process; acoustic model unchanged."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

task = Path(__file__).parent.resolve()
parent = task.parent / "2026-10-14-fast-muaalem-local-analysis"
sys.path.insert(0, str(parent))
import adapter  # noqa: E402

Original = adapter.MuaalemLocal
table_path = task / "variants.json"
table = json.loads(table_path.read_text())
assert len(table["madd4_native_parity"]) == 12 and all(p["equal"] for p in table["madd4_native_parity"])
length = int(os.environ["MUA_MADD"])
table_sha = hashlib.sha256(table_path.read_bytes()).hexdigest()


class VariantLocal(Original):
    def __init__(self, selected):
        super().__init__(selected)
        self.version += f"-madd{length}-{table_sha}"
        self.runtime_identity["experimental_static_phones"] = {
            "madd_aared_len": length, "table_sha256": table_sha, "native4_parity": True,
        }

    def represent(self, words):
        mapping = table["variants"][str(length)][str(words[0].surah)]
        if mapping["words"] != [[w.ayah, w.index, w.text] for w in words]:
            raise ValueError("Static variant canonical ownership mismatch")
        targets, spans = mapping["targets"], mapping["spans"]
        ends = [0] + [end for start, end in spans]
        if (ends[-1] != len(targets) or any(start != previous or end <= start
                                          for previous, (start, end) in zip(ends, spans))):
            raise ValueError("Variant phones are not a full nonoverlapping partition")
        self.mapping = mapping
        return tuple(tuple(self.inv[token] for token in targets[start:end]) for start, end in spans)


adapter.MuaalemLocal = VariantLocal
runpy.run_path(str(parent / "identity_floor_probe.py"), run_name="__main__")
