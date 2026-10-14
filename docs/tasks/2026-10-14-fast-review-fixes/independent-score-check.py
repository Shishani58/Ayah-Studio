"""Verify complete replay equality except independently derived tail scores."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

root = Path(__file__).parent
raw_path = root / "corpus-final.json"
raw = json.loads(raw_path.read_text())
replay = json.loads((root / "corpus-score-replay.json").read_text())
assert replay["raw_corpus_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
assert replay["rekeyed"] is False and replay["validated_cases"] == len(raw) == 138
executor = Path("surah_cutter/core/fast_executor.py")
assert replay["new_executor_sha256"] == hashlib.sha256(executor.read_bytes()).hexdigest()
diff_identity = json.loads((root / "source-4-to-5.json").read_text())
assert diff_identity["sparse5_executor_sha256"] == replay["new_executor_sha256"]
with TemporaryDirectory(prefix="independent-fast-source-diff-") as folder:
    restored = Path(folder) / "executor.py"
    restored.write_bytes(executor.read_bytes())
    subprocess.run(["patch", "-R", str(restored), "-i", str((root / "source-4-to-5.diff").resolve())],
                   check=True, capture_output=True)
    assert hashlib.sha256(restored.read_bytes()).hexdigest() == diff_identity["raw_sparse4_executor_sha256"]
assert json.loads((root / "corpus-final.summary.json").read_text())["source_sha256"][str(executor)] == diff_identity["raw_sparse4_executor_sha256"]
changes = []
for old, corrected in zip(raw, replay["results"]):
    expected = deepcopy(old)
    trace = expected["trace"]
    for boundary in trace["boundaries"]:
        proof = boundary.get("proof") or {}
        if boundary["evidence_kind"] == "pause" and proof.get("contextual_owner"):
            observed = trace["acoustic_reads"][proof["previous_context_read_id"]]["observed"]
            previous = boundary["confidence"]
            confidence = min(previous, *observed["unit_quality"][-len(proof["previous_tail"]):])
            boundary["confidence"] = confidence
            for row in expected["rows"]:
                if row[0] == boundary["ayah"]:
                    row[2] = confidence
            for entry in trace["ayahs"] + trace["pause_matches"]:
                if entry["ayah"] == boundary["ayah"]:
                    entry["conf"] = confidence
            if previous != confidence:
                changes.append({"model": old["model"], "surah": old["surah"], "ayah": boundary["ayah"],
                                "before": previous, "after": confidence})
    assert expected == corrected
assert changes == replay["changes"] and len(changes) == 19
(root / "independent-score-check.json").write_text(json.dumps({"cases": len(raw),
    "changed_scores": len(changes), "source_diff_restores_frozen_sha": True,
    "all_other_values_equal": True, "rekeyed": False}, indent=2) + "\n")
print("138 scalar replay cases and complete source delta independently verified")
