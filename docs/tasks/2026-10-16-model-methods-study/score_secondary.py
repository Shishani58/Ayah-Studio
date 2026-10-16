"""Offline scoring only. Reference timings never enter prediction processes."""
import json
import sys
import unicodedata
from collections import defaultdict

import numpy as np

from common import TASK, write_json

ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT))
from tools.research.references import load_reference  # noqa: E402


def norm(text):
    table = str.maketrans("أإآٱى", "ااااي")
    return "".join(c for c in text.translate(table)
                   if unicodedata.category(c).startswith("L"))


def words(text):
    return [v for w in text.split() if (v := norm(w))]


def align_words(expected, predicted):
    """Levenshtein with deterministic backtrace; exact matches supply timestamps."""
    n, m = len(expected), len(predicted)
    dp = np.empty((n + 1, m + 1), dtype=np.int32)
    dp[:, 0], dp[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1,
                           dp[i-1, j-1]+(expected[i-1] != predicted[j-1]))
    i, j, matches = n, m, {}
    while i or j:
        if i and j and dp[i, j] == dp[i-1, j-1]+(expected[i-1] != predicted[j-1]):
            if expected[i-1] == predicted[j-1]:
                matches[i-1] = j-1
            i, j = i-1, j-1
        elif i and dp[i, j] == dp[i-1, j]+1:
            i -= 1
        else:
            j -= 1
    return int(dp[n, m]), matches


def stats(values):
    if not values:
        return None
    a = np.abs(values)
    return dict(n=len(a), median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)),
                max_ms=float(np.max(a)), over500=int(np.sum(a > 500)))


def main():
    truth = {c["id"]: c for c in json.loads((TASK / "secondary-truth.json").read_text())}
    records, failures, negative = [], [], []
    tokenizer = None
    for folder in ("asr-nemo", "asr-whisper", "muaalem", "nemo-forced"):
        for path in sorted((TASK / folder).glob("*.json")):
            item = json.loads(path.read_text())
            if "case" not in item:
                continue
            case, method = item["case"], item["method"]
            if "error" in item:
                failures.append(dict(record=case["id"], method=method, error=item["error"]))
                continue
            if case["surah"] is None:
                negative.append(dict(method=method, text=item.get("text"),
                                     not_applicable=item.get("not_applicable")))
                continue
            target = truth[case["id"]]
            canonical = target["intro"] + target["words"]
            owners = [0]*len(target["intro"]) + target["owners"]
            reference = load_reference(case["reciter"])
            row = dict(record=case["id"], reciter=case["reciter"], method=method,
                       expected_ayahs=len(set(target["owners"])), path=str(path))
            starts = {}
            if "rows" in item:
                starts = {int(a): float(ms) for a, ms, *_ in item["rows"]}
                row["known_text_supplied"] = True
            elif "word_rows" in item:
                for owner, entry in zip(owners, item["word_rows"]):
                    if owner and owner not in starts:
                        starts[owner] = float(entry[0])
                row["known_text_supplied"] = True
            else:
                expected = [norm(w) for w in canonical]
                predicted = words(item["text"])
                edits, _ = align_words(expected, predicted)
                row.update(edits=edits, expected_words=len(expected), wer=edits/len(expected),
                           known_text_supplied=False, text=item["text"])
                timed = item.get("words", [])
                if method == "token_timestamps":
                    if tokenizer is None:
                        from transformers import WhisperTokenizer
                        identity = json.loads((TASK / "asr-whisper/identity.json").read_text())
                        from pathlib import Path
                        tokenizer = WhisperTokenizer.from_pretrained(str(Path.home() / ".surah-cutter/models/whisper-tarteel-base" / identity["revision"]), local_files_only=True)
                    for chunk in item["chunks"]:
                        output = {"tokens": np.asarray([chunk["tokens"][0]]),
                                  "token_timestamps": np.asarray([chunk["token_timestamps_s"][0]])}
                        _, data = tokenizer._decode_asr([output], return_timestamps="word",
                                                       return_language=False, time_precision=0.02)
                        for w in data["chunks"]:
                            if w["timestamp"][0] is not None:
                                timed.append(dict(word=w["text"], start_ms=chunk["start_ms"]+1000*w["timestamp"][0]))
                cleaned = [w for w in timed if norm(w["word"])]
                _, matches = align_words(expected, [norm(w["word"]) for w in cleaned])
                seen = set()
                for index, owner in enumerate(owners):
                    if owner and owner not in seen:
                        seen.add(owner)
                        if index in matches:
                            starts[owner] = cleaned[matches[index]]["start_ms"]
            errors_by_ayah = {a:ms-reference[case["surah"], a] for a, ms in sorted(starts.items())}
            errors = list(errors_by_ayah.values())
            ordered = [ms for _, ms in sorted(starts.items())]
            row.update(covered_ayahs=len(starts), errors=errors, timing=stats(errors),
                       starts_ms=starts, errors_by_ayah=errors_by_ayah,
                       nonincreasing_pairs=sum(b <= a for a, b in zip(ordered, ordered[1:])))
            records.append(row)
    groups = defaultdict(list)
    for row in records:
        groups[row["method"]].append(row)
    summary = []
    for method, rows in sorted(groups.items()):
        entry = dict(method=method, records=len(rows), expected_ayahs=sum(r["expected_ayahs"] for r in rows),
                     covered_ayahs=sum(r["covered_ayahs"] for r in rows),
                     timing=stats([e for r in rows for e in r["errors"]]))
        if "edits" in rows[0]:
            entry["wer"] = sum(r["edits"] for r in rows)/sum(r["expected_words"] for r in rows)
        summary.append(entry)
    write_json(TASK / "secondary-scores.json", dict(summary=summary, records=records,
               failures=failures, negative=negative,
               caveat="Timing is reference-cut agreement; partial ASR timing coverage is not full alignment accuracy."))
    print(json.dumps(summary, indent=2))
    print("failures", len(failures))


if __name__ == "__main__":
    main()
