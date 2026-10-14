"""Recompute frozen audio artifacts after inference; never call text models."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from surah_cutter.core import decode
from surah_cutter.data.alignment_identity import fast_run_key
from surah_cutter.data import quran_text

parser = argparse.ArgumentParser()
parser.add_argument("corpus", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--legacy", type=Path)
parser.add_argument("--expected-records", type=int, default=138)
parser.add_argument("--algorithm", default="sequential-fast-v3-experimental-sparse-4")
args = parser.parse_args()
items = json.loads(args.corpus.read_text())
summary = json.loads(args.corpus.with_suffix(".summary.json").read_text())
assert len(items) == len({(i["model"], i["surah"]) for i in items})
source_matches = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
                  for p, sha in summary["source_sha256"].items()}
assert all(source_matches.values()), source_matches
assert summary["canonical_sha256"] == hashlib.sha256(quran_text.DEFAULT_DB_PATH.read_bytes()).hexdigest()
assert len(items) == args.expected_records
keys = [i["trace"]["fast_run_key"] for i in items]
assert len(set(keys)) == len(keys)

audios = {}
for surah in sorted({i["surah"] for i in items}):
    path = Path(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3")
    pcm = decode.decode(str(path))
    audios[surah] = (hashlib.sha256(path.read_bytes()).hexdigest(), pcm)

reference = Path("data/references/abu_bakr_ash_shatri.db")
reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
with sqlite3.connect(f"{reference.resolve().as_uri()}?mode=ro", uri=True) as connection:
    truths = {s: dict(connection.execute("SELECT ayah,time FROM timings WHERE sura=?", (s,))) for s in audios}

metrics = {}
for item in items:
    model, surah, trace = item["model"], item["surah"], item["trace"]
    audio_sha, pcm = audios[surah]
    assert item["audio_sha256"] == audio_sha
    assert item["reference_sha256"] == reference_sha
    assert trace["run_snapshot"]["pcm_sha256"] == hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest()
    assert trace["fast_run_key"] == fast_run_key(trace["run_snapshot"])
    assert trace["run_snapshot"]["algorithm"] == args.algorithm
    initial = summary["initial_pauses"][str(surah)]
    assert initial["audio_sha256"] == audio_sha
    assert item["initial_pauses_sha256"] == hashlib.sha256(
        json.dumps(initial["decisions"], sort_keys=True).encode()).hexdigest()
    assert trace["peak_cache_bytes"] <= trace["run_snapshot"]["policy"]["cache_bytes"]
    assert abs(trace["total_ms"] - len(pcm) / 16) < 1e-9
    rows = item["rows"]
    assert [a for a, ms, c in rows] == list(range(1, quran_text.ayah_count(surah) + 1))
    assert all(type(ms) is int and 0 <= ms < trace["total_ms"] for a, ms, c in rows)
    assert all(a[1] < b[1] for a, b in zip(rows, rows[1:]))
    assert all(np.isfinite(c) and 0 <= c <= 1 for a, ms, c in rows)
    errors = [abs(ms - truths[surah][a]) for a, ms, c in rows]
    assert errors == item["errors_ms"]
    reads = trace["acoustic_reads"]
    assert len(reads) == len(item["actual_dispatches"])
    merged, sum_ms = [], 0.
    for actual, read in zip(item["actual_dispatches"], reads):
        left, right = (round(t * 16) for t in read["window_ms"])
        local = np.ascontiguousarray(pcm[left:right], dtype=np.float32)
        assert 0 < right - left <= 96000 and right - left < len(pcm)
        assert actual["samples"] == len(local)
        assert actual["sha256"] == hashlib.sha256(memoryview(local)).hexdigest()
        sum_ms += len(local) / 16
        assert "posterior" not in read.get("observed", {})
    for left, right in sorted(r["window_ms"] for r in reads):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    union_ms = sum(right - left for left, right in merged)
    assert abs(sum_ms - trace["audio_ms"]) < .1
    assert abs(union_ms - trace["covered_audio_ms"]) < .1
    assert union_ms <= trace["total_ms"] * .65 and sum_ms <= trace["total_ms"] * 2
    assert not item["sequence_violations"]
    if "boundaries" in trace:
        assert [(b["ayah"], b["start_ms"], b["confidence"]) for b in trace["boundaries"]] == [tuple(r) for r in rows]
        assert {b["ayah"] for b in trace["boundaries"] if b["evidence_kind"] == "estimate"} == set(trace["review_required_ayahs"])
    bucket = metrics.setdefault(model, {"records": 0, "ayahs": 0, "confirmed": 0, "over500": [], "all_errors": []})
    bucket["records"] += 1
    bucket["ayahs"] += len(rows)
    bucket["all_errors"].extend(errors)
    for (ayah, ms, confidence), error in zip(rows, errors):
        if confidence > 0 and ayah not in trace["review_required_ayahs"]:
            bucket["confirmed"] += 1
            if error > 500:
                bucket["over500"].append([surah, ayah, error])

for bucket in metrics.values():
    errors = bucket.pop("all_errors")
    bucket.update(median_ms=float(np.median(errors)), p95_ms=float(np.percentile(errors, 95)), max_ms=max(errors))

comparison = {}
if args.legacy:
    legacy = json.loads(args.legacy.read_text())
    current = {(i["model"], i["surah"]): i for i in items}
    for item in legacy:
        key = item["model"], item["surah"]
        if key not in current:
            continue
        paired = current[key]
        assert all(item[k] == paired[k] for k in ("audio_sha256", "reference_sha256", "initial_pauses_sha256"))
        bucket = comparison.setdefault(key[0], {"pairs": 0, "legacy_confirmed": 0, "new_confirmed": 0})
        bucket["pairs"] += 1
        for candidate, label in ((item, "legacy_confirmed"), (paired, "new_confirmed")):
            bucket[label] += sum(c > 0 and a not in candidate["trace"]["review_required_ayahs"] for a, ms, c in candidate["rows"])

args.output.write_text(json.dumps({"source_matches": source_matches, "unique_run_keys": len(keys),
                                   "metrics": metrics, "paired": comparison}, indent=2) + "\n")
print(json.dumps({"metrics": metrics, "paired": comparison}, indent=2))
