"""Classify exact selected BPE bytes from saved observations; no inference."""
import hashlib
import json
from pathlib import Path
import unicodedata

task = Path(__file__).parent.resolve()
vocab_path = Path("/Users/ayub/.surah-cutter/models/whisper-tarteel-base/5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c/vocab.json")
inverse = {i: token for token, i in json.loads(vocab_path.read_text()).items()}
byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
characters = byte_values[:]
offset = 0
for value in range(256):
    if value not in byte_values:
        byte_values.append(value)
        characters.append(256 + offset)
        offset += 1
decoder = {chr(c): b for b, c in zip(byte_values, characters)}
records = []
unsupported = []
for item in json.loads((task / "control.json").read_text()):
    for read in item["trace"]["acoustic_reads"]:
        observed = read["observed"]
        for word in observed["attribution"]:
            if not word.get("token_ids") or "spoken_alias" in word:
                continue
            payload = bytearray()
            owners = []
            for index, token_id in enumerate(word["token_ids"]):
                raw = bytes(decoder[c] for c in inverse[token_id])
                payload.extend(raw)
                owners.extend([index] * len(raw))
            try:
                text = payload.decode("utf8", errors="strict")
            except UnicodeDecodeError:
                unsupported.append([item["surah"], read["id"], word["word"]])
                continue
            letter_owners, mark_owners = set(), set()
            position = 0
            for char in text:
                length = len(char.encode("utf8"))
                current = owners[position:position + length]
                position += length
                for normalized in unicodedata.normalize("NFKC", char):
                    if unicodedata.category(normalized).startswith("L") and normalized != "ـ":
                        letter_owners.update(current)
                    elif unicodedata.category(normalized).startswith("M"):
                        mark_owners.update(current)
            if not letter_owners:
                continue
            probabilities = word["probabilities"]
            full_quality = min(probabilities)
            letter_quality = min(probabilities[i] for i in letter_owners)
            if full_quality < .6 <= letter_quality:
                mark_only = mark_owners - letter_owners
                records.append({"surah": item["surah"], "read_id": read["id"], "reason": read["reason"],
                                "window_ms": read["window_ms"], "word": word["word"],
                                "decoded_token_bytes": text, "full_quality": full_quality,
                                "letter_quality": letter_quality,
                                "mark_only_tokens": [{"id": word["token_ids"][i],
                                                      "probability": probabilities[i],
                                                      "bytes": list(bytes(decoder[c] for c in inverse[word["token_ids"][i]]))}
                                                     for i in sorted(mark_only)]})
report = {"vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
          "rescued_word_quality_only": records, "unsupported_word_crops": unsupported,
          "scope": "token-role diagnosis only; not new boundaries or a calibrated quality policy"}
(task / "token-roles.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print("words below .6 only because of discarded marks:", len(records))
print(json.dumps(records[:8], ensure_ascii=False, indent=1))
