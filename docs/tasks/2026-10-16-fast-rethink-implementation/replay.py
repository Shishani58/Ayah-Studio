"""Exact-PCM ASR observation replay, with no model loading or fresh fallback.

No reference times are read. Replay wall time is not inference latency.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import zipfile

import numpy as np
from surah_cutter.core import candidates, decode, fast_executor
from surah_cutter.core.fast_adapters import Observation, TarteelAdapter
from surah_cutter.data import quran_text
from tools.bench_fast_models import verify_dispatch

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("label", help="new output basename; existing artifacts are never replaced")
args = parser.parse_args()
output = P / (args.label + ".jsonl")
archive = P / (args.label + "-source.zip")
manifest = P / (args.label + "-source-sha.json")
for path in (output, archive, manifest):
    if path.exists():
        raise FileExistsError(path)
paths = sorted(set(ROOT.glob("surah_cutter/**/*.py")) | {
    Path(__file__).resolve(), ROOT / "tools/bench_fast_models.py",
    ROOT / "AGENTS.md", ROOT / "CLAUDE.md",
})
source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as saved:
    for path in paths:
        saved.write(path, str(path.relative_to(ROOT)))
manifest.write_text(json.dumps({"files": source,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2) + "\n")

labels = ("baseline", "identity", "reentry", "current-shatri", "current-ghamidi",
          "current-hybrid2", "final-replay2", "final-smoke", "final-regression80")
cases = {}
for label in labels:
    for line in (P / (label + ".jsonl")).read_text().splitlines():
        item = json.loads(line)
        if item["model"] != "whisper-tarteel-base":
            continue
        key = (item["reciter"], item["surah"])
        case = cases.setdefault(key, {"base": item, "observations": {}})
        assert item["trace"]["run_snapshot"]["pcm_sha256"] == case["base"]["trace"]["run_snapshot"]["pcm_sha256"]
        assert item["trace"]["run_snapshot"]["text_runtime"] == case["base"]["trace"]["run_snapshot"]["text_runtime"]
        for actual, read in zip(item["dispatches"], item["trace"]["acoustic_reads"], strict=True):
            key = (actual["samples"], actual["sha256"])
            # A repeated actual PCM must have the same saved acoustic evidence.
            previous = case["observations"].get(key)
            if previous:
                assert previous[0] == read["observed"], (label, key)
            else:
                case["observations"][key] = (read["observed"],
                    {"artifact": label, "read_id": read["id"], "kind": "saved_exact_PCM"})

with output.open("x") as log:
    for key in (("shatri", 2), ("shatri", 78), ("shatri", 80), ("shatri", 112),
                ("saad_al_ghamidi", 80), ("saad_al_ghamidi", 112)):
        case = cases[key]
        base = case["base"]
        pcm = decode.decode(base["audio_path"])
        assert hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest() == base["trace"]["run_snapshot"]["pcm_sha256"]
        actual, provenance = [], []

        class Replay(TarteelAdapter):
            def observe(self, local, sr, *, total_samples):
                digest = hashlib.sha256(memoryview(np.ascontiguousarray(local))).hexdigest()
                crop = (len(local), digest)
                if crop not in case["observations"]:
                    raise LookupError("No saved observation for exact PCM crop: " + str(crop))
                data, origin = case["observations"][crop]
                actual.append({"samples": len(local), "sha256": digest})
                provenance.append(origin)
                data = copy.deepcopy(data)
                for name in ("units", "unit_quality", "attribution", "hypotheses", "positions_ms"):
                    if data.get(name) is not None:
                        data[name] = tuple(data[name])
                return Observation(**data)

        adapter = Replay(SimpleNamespace(emitter=SimpleNamespace(model_id=base["model"],
            runtime_identity=base["trace"]["run_snapshot"]["text_runtime"])))
        decisions = [candidates.PauseDecision(**{**d, "pause": candidates.Pause(**d["pause"])})
                     for d in base["decisions"]]
        trace = {}
        started = monotonic()
        try:
            rows = fast_executor.align_surah_sequential(pcm, 16000, quran_text.surah_words(base["surah"]),
                adapter=adapter, surah=base["surah"], trace=trace, pause_decisions=decisions)
            verify_dispatch(actual, pcm, trace)
            result = {"outcome": "completed", "rows": rows}
        except BaseException as error:
            result = {"outcome": "failed", "error": type(error).__name__ + ": " + str(error)}
        result.update({"model": base["model"], "reciter": key[0], "surah": key[1], "trace": trace,
            "dispatches": actual, "observation_provenance": provenance, "fresh_dispatches": [],
            "seconds": monotonic() - started, "measurement_kind": "current-source exact-PCM observation replay; zero fresh calls; not inference latency",
            "source_sha256": source, "source_archive": archive.name,
            "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "audio_path": base["audio_path"], "audio_sha256": base["audio_sha256"],
            "decisions": [asdict(d) for d in decisions]})
        log.write(json.dumps(result, ensure_ascii=False) + "\n")
        log.flush()
        print(key, result["outcome"], "proof", len(result.get("rows", [])), "calls", len(actual),
              result.get("error", ""), flush=True)
