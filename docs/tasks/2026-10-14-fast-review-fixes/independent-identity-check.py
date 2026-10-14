"""Loaded CTC identity lifetime and influence; temporary files, no ONNX run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from surah_cutter.core.emissions import OnnxEmitter
from surah_cutter.core.fast_adapters import loaded_implementation_sha
from surah_cutter.core.fast_ctc import independent_beam
from surah_cutter.data.alignment_identity import fast_run_key


class Session:
    def __init__(self, path):
        self._model_path = str(path)

    def get_inputs(self):
        return [SimpleNamespace(name="audio")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def get_providers(self):
        return ["CPUExecutionProvider"]


with TemporaryDirectory(prefix="independent-fast-identity-") as folder:
    path = Path(folder) / "weights.onnx"
    path.write_bytes(b"original already loaded weights")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    emitter = OnnxEmitter(Session(path), {"blank": 0, "a": 1, "b": 2}, blank_id=0,
                          weights_sha256=digest)
    original = emitter.runtime_identity
    path.write_bytes(b"different file after loading")
    assert emitter.runtime_identity == original
    assert original["weights_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    mutable = emitter.runtime_identity
    mutable["vocab"]["a"] = 900
    assert emitter.runtime_identity == original

snapshot = json.loads(Path("docs/tasks/2026-10-14-fast-full-execution/corpus-final.json").read_text())[0]["trace"]["run_snapshot"]
snapshot["text_runtime"] = original
key = fast_run_key(snapshot)
changes = {}
for field in ("weights_sha256", "vocab", "blank_id", "do_normalize", "onnxruntime", "providers"):
    changed = deepcopy(snapshot)
    value = changed["text_runtime"][field]
    if field == "vocab":
        value["a"] += 1
    elif isinstance(value, bool):
        changed["text_runtime"][field] = not value
    elif isinstance(value, int):
        changed["text_runtime"][field] = value + 1
    elif isinstance(value, list):
        changed["text_runtime"][field] = ["DifferentProvider"]
    else:
        changed["text_runtime"][field] = value + "-changed"
    assert fast_run_key(changed) != key
    changes[field] = True
loaded = loaded_implementation_sha()
assert loaded == loaded_implementation_sha()
width = independent_beam.__kwdefaults__["width"]
try:
    independent_beam.__kwdefaults__["width"] = width + 1
    assert loaded_implementation_sha() != loaded
finally:
    independent_beam.__kwdefaults__["width"] = width
assert loaded_implementation_sha() == loaded

Path(__file__).with_suffix(".json").write_text(json.dumps({
    "file_change_does_not_relabel_loaded_weights": True, "returned_copy_cannot_mutate_snapshot": True,
    "influence_changes_key": changes,
    "loaded_code_kwdefault_changes_fingerprint": True,
}, indent=2) + "\n")
print("9 independent loaded-identity cases passed")
