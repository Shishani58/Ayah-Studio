"""Scalar proof-score correction only: no new inference, timing, windows, or keys."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from surah_cutter.core.fast_adapters import CTCAdapter, TarteelAdapter
from surah_cutter.core.fast_executor import Boundary, Policy, validate_boundaries
from surah_cutter.data.quran_text import surah_words


def main():
    root = Path(__file__).parent
    source = root / "corpus-final.json"
    data = json.loads(source.read_text())
    changes = []
    for item in data:
        trace = item["trace"]
        words = surah_words(item["surah"])
        adapter = object.__new__(CTCAdapter if item["model"] == "quran-phonetics" else TarteelAdapter)
        adapter.model_id = item["model"]
        rendered = adapter.represent(words)
        proof_ends = (adapter.proof_word_ends(words, rendered) if hasattr(adapter, "proof_word_ends")
                      else tuple((len(units),) for units in rendered))
        groups, ends, canonical = {}, {}, {}
        for word, units, word_ends in zip(words, rendered, proof_ends):
            base = len(groups.setdefault(word.ayah, ()))
            groups[word.ayah] += units
            ends.setdefault(word.ayah, []).extend(base + end for end in word_ends)
            canonical.setdefault(word.ayah, []).append(base + len(units))
        for boundary in trace["boundaries"]:
            proof = boundary.get("proof") or {}
            if boundary["evidence_kind"] != "pause" or not proof.get("contextual_owner"):
                continue
            context = trace["acoustic_reads"][proof["previous_context_read_id"]]["observed"]
            before = boundary["confidence"]
            after = min(before, min(context["unit_quality"][-len(proof["previous_tail"]):]))
            boundary["confidence"] = after
            for row in item["rows"]:
                if row[0] == boundary["ayah"]:
                    row[2] = after
            for entry in [*trace["ayahs"], *trace["pause_matches"]]:
                if entry["ayah"] == boundary["ayah"]:
                    entry["conf"] = after
            if after != before:
                changes.append({"model": item["model"], "surah": item["surah"],
                                "ayah": boundary["ayah"], "before": before, "after": after})
        pauses = [SimpleNamespace(start_ms=g["interval_ms"][0], end_ms=g["interval_ms"][1])
                  for g in trace["pause_geometry"]]
        boundaries = [Boundary(**b) for b in trace["boundaries"]]
        validate_boundaries(boundaries, sorted(groups), item["duration_ms"], trace["acoustic_reads"],
                            groups=groups, word_ends=ends, canonical_ends=canonical,
                            threshold=.25 if item["model"] == "quran-phonetics" else .6,
                            pauses=pauses, pause_geometry_data=trace["pause_geometry"], policy=Policy())
        trace["boundaries"] = [asdict(b) for b in boundaries]
    raw = json.loads(source.read_text())
    for old, corrected in zip(raw, data):
        assert old["actual_dispatches"] == corrected["actual_dispatches"]
        assert old["trace"]["acoustic_reads"] == corrected["trace"]["acoustic_reads"]
        assert old["trace"]["fast_run_key"] == corrected["trace"]["fast_run_key"]
        assert [r[:2] for r in old["rows"]] == [r[:2] for r in corrected["rows"]]
        assert [r[2] > 0 for r in old["rows"]] == [r[2] > 0 for r in corrected["rows"]]
    report = {"kind": "scalar_only_proof_score_replay_not_new_acoustic_inference",
              "raw_corpus_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "new_executor_sha256": hashlib.sha256(Path("surah_cutter/core/fast_executor.py").read_bytes()).hexdigest(),
              "rekeyed": False, "current_algorithm": "sequential-fast-v3-experimental-sparse-5",
              "retained_runtime_keys": "Historical sparse-4 provenance; not usable as sparse-5 runtime keys.",
              "validated_cases": len(data), "changes": changes, "results": data}
    (root / "corpus-score-replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Scalar-only replay:", len(data), "cases; changed scores", len(changes), "; no new inference/rekey")


if __name__ == "__main__":
    main()
