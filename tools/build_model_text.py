"""Build checked inference-only Arabic renderings; never edit Quran assets."""

import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import unicodedata

from quran_transcript import Aya


def skeleton(text):
    text = text.translate(str.maketrans({"ٱ": "ا", "ى": "ي", "ء": ""}))
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) == "Lo")


def main():
    root = Path(__file__).resolve().parents[1]
    database = root / "data/quran_mushaf.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT surah,ayah,qpc_hafs FROM words WHERE TRIM(qpc_hafs)!='' ORDER BY id").fetchall()
    surahs = {}
    for (surah, ayah), group in itertools.groupby(rows, key=lambda r: tuple(r[:2])):
        aya = Aya(surah, ayah)
        source = aya.get()
        encoding = aya._encode_imlaey_to_uthmani()
        cursor = 0
        records = surahs.setdefault(str(surah), [])
        for _, _, canonical in group:
            expected = skeleton(canonical)
            start = cursor
            while cursor < len(source.uthmani_words):
                cursor += 1
                actual = skeleton(" ".join(source.uthmani_words[start:cursor]))
                if actual == expected:
                    break
                assert expected.startswith(actual), (surah, ayah, canonical)
            assert actual == expected, (surah, ayah, canonical)
            imlaey = [word for i, word in enumerate(source.imlaey_words)
                      if start <= encoding.imlaey2uthmani[i] < cursor]
            assert imlaey, (surah, ayah, canonical)
            records.append([canonical, " ".join(source.uthmani_words[start:cursor]), " ".join(imlaey)])
        assert cursor == len(source.uthmani_words), (surah, ayah)
    import importlib.metadata

    output = {"canonical_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
              "generator": f"quran-transcript:{importlib.metadata.version('quran-transcript')}",
              "surahs": surahs}
    (root / "data/model_text_targets.json").write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"114 surahs, {len(rows)} canonical words checked")


if __name__ == "__main__":
    main()
