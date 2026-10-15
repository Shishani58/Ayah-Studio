"""Post-decode text/timing checks; never supplies text to the acoustic model."""

import difflib
import json
import unicodedata
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent


def normalize(text):
    text = text.translate(str.maketrans("أإآٱى", "ااااي"))
    return "".join(c for c in text if unicodedata.category(c) not in ("Mn", "Me")
                   and (c.isalpha() or c.isspace())).split()


def distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        previous, row = row, [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
    return row[-1]


def main():
    results = []
    for model in ("fastconformer", "fastconformer-ctc", "whisper-tarteel-base"):
        data = json.loads((TASK / f"{model}-asr.json").read_text())
        summary = dict(model=model, decoder=data["decoder"], load_s=data["load_s"],
                       full=[], local=[], silence=[], shift=[])
        for case in data["cases"]:
            hyp = normalize(case.get("text", ""))
            if case["kind"] == "FULL-ASR":
                ref = normalize(case["reference_text"])
                # Non-Fatiha basmala can be heard in recordings but is not an ayah.
                # Score both raw and explicit prefix-adjusted versions.
                removed = 0
                adjusted = hyp
                basmala = normalize("بسم الله الرحمن الرحيم")
                if case["surah"] != 1 and hyp[:4] == basmala:
                    adjusted, removed = hyp[4:], 4
                record = dict(id=case["id"], ref_words=len(ref), raw_errors=distance(ref, hyp),
                              errors=distance(ref, adjusted), removed_basmala_words=removed,
                              elapsed_s=case["elapsed_s"], ayah_onsets=[])
                if "words" in case:
                    # Evaluation only: map exact independently recognized words to reference.
                    mapping = {}
                    for block in difflib.SequenceMatcher(None, ref, adjusted, autojunk=False).get_matching_blocks():
                        mapping.update((block.a + k, block.b + k + removed) for k in range(block.size))
                    start_index = 0
                    for ayah in case["reference_ayahs"]:
                        if start_index in mapping:
                            onset = case["words"][mapping[start_index]]["start_ms"]
                            record["ayah_onsets"].append(dict(ayah=ayah["ayah"], onset_ms=onset,
                                                              error_ms=onset - ayah["onset_ms"]))
                        start_index += len(normalize(ayah["text"]))
                summary["full"].append(record)
            elif case["kind"] == "SILENCE":
                summary["silence"].append(dict(text=case.get("text"), words=len(hyp)))
            else:
                summary["local"].append(dict(id=case["id"], text=case.get("text"),
                                             elapsed_s=case["elapsed_s"]))
        total = sum(c["ref_words"] for c in summary["full"])
        summary["full_wer_pct"] = 100 * sum(c["errors"] for c in summary["full"]) / total
        summary["full_reference_words"] = total
        fatiha = summary["full"][0]["ayah_onsets"]
        if fatiha:
            errors = [abs(a["error_ms"]) for a in fatiha]
            summary["fatiha_timing"] = dict(count=len(errors), median_ms=float(np.median(errors)),
                                             max_ms=max(errors), over500=sum(e > 500 for e in errors))
        normal = next(c for c in data["cases"] if c["id"] == "abu_bakr_ash_shatri-1-local")
        shifted = next(c for c in data["cases"] if c["id"] == "shatri-1-local-shift600")
        if normal.get("words") and shifted.get("words"):
            for left, right in zip(normal["words"], shifted["words"]):
                summary["shift"].append(dict(word=right["word"],
                                             same_word=normalize(left["word"]) == normalize(right["word"]),
                                             start_residual_ms=right["start_ms"] - left["start_ms"] - 600,
                                             end_residual_ms=right["end_ms"] - left["end_ms"] - 600))
        results.append(summary)
    (TASK / "evaluation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for result in results:
        print(result["model"], result["full_wer_pct"], result.get("fatiha_timing"), result["silence"])


if __name__ == "__main__":
    main()
