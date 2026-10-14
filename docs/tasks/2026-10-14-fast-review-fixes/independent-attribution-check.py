"""Independent word-quality adversarial probes; no model/database calls."""
import json
import math
from pathlib import Path

from surah_cutter.core.fast_adapters import attributed_asr_units


def token(identifier, raw, probability):
    return {"id": identifier, "bytes": list(raw), "log_probability": math.log(probability)}


words = ("قل", "هو", "الله", "احد", "ضعيف")
text = " ".join(words)
tokens = [token(i, ((" " if i else "") + word).encode(), .1 if i == 4 else .98)
          for i, word in enumerate(words)]
units, quality, attribution = attributed_asr_units({"text": text, "selected_tokens": tokens})
assert units == words
assert all(abs(q - .98) < 1e-12 for q in quality[:4]) and abs(quality[4] - .1) < 1e-12

raw = "قل".encode()
units, split_quality, mapping = attributed_asr_units({
    "text": "قل", "selected_tokens": [token(11, raw[:1], .12), token(12, raw[1:], .99)]})
assert units == ("قل",) and abs(split_quality[0] - .12) < 1e-12
assert mapping[0]["token_ids"] == [11, 12]

units, shared_quality, mapping = attributed_asr_units({
    "text": "قل هو", "selected_tokens": [token(20, "قل هو".encode(), .31)]})
assert units == ("قل", "هو") and all(abs(q - .31) < 1e-12 for q in shared_quality)
assert all(m["token_ids"] == [20] for m in mapping)

_, quality, mapping = attributed_asr_units({"text": "قل", "selected_tokens": [token(30, "هو".encode(), .99)]})
assert quality == (0.,) and "unsupported" in mapping[0]

_, quality, mapping = attributed_asr_units({
    "text": "قل", "selected_tokens": [{"id": 40, "bytes": list(raw), "log_probability": float("nan")}]})
assert quality == (0.,)

Path(__file__).with_suffix(".json").write_text(json.dumps({
    "weak_unneeded_tail": True, "split_utf8_token": True, "shared_word_token": True,
    "nonreversible_text_rejected": True, "nonfinite_rejected": True,
}, indent=2) + "\n")
print("5 independent ASR attribution cases passed")
