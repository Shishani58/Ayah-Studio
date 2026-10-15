"""Inspect raw glyph alias differences in retained ASR reads, no inference."""
import json
from pathlib import Path
import unicodedata
from surah_cutter.core.fast_adapters import arabic_units
from surah_cutter.core.model_worker import whisper_alignment_text

task = Path(__file__).parent.resolve()
prior = task.parent / "2026-10-14-fast-previous-body-rejection"
result = []
for model in ["whisper-tarteel-tiny","whisper-tarteel-base"]:
    for case in json.loads((prior / f"{model}-after.json").read_text()):
        for read in case["trace"]["acoustic_reads"]:
            text = unicodedata.normalize("NFKC",read["observed"]["text"])
            raw_words,current = [],[]
            for char in text:
                if unicodedata.category(char).startswith(("L","M")):
                    if char != "ـ":
                        current.append(char)
                elif current:
                    raw_words.append("".join(current))
                    current=[]
            if current:
                raw_words.append("".join(current))
            normalized = arabic_units(text)
            if len(raw_words)!=len(normalized):
                result.append({"model":model,"surah":case["surah"],"read_id":read["id"],"unsupported_split":text})
                continue
            for raw,word in zip(raw_words,normalized):
                old = arabic_units(whisper_alignment_text(word))
                new = arabic_units(whisper_alignment_text(raw))
                if old != new:
                    result.append({"model":model,"surah":case["surah"],"read_id":read["id"],
                                   "reason":read["reason"],"raw_word":raw,"normalized_word":word,
                                   "old_units":old,"corrected_units":new})
(task / "retained-alias-cases.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
