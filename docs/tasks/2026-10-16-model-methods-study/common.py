"""Frozen source bootstrap and study-local helpers."""
import hashlib
import json
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

TASK = Path(__file__).resolve().parent
SNAPSHOT = json.loads((TASK / "snapshot.json").read_text())
SOURCE = Path(SNAPSHOT["snapshot"])
if not SOURCE.is_dir():
    SOURCE = Path(tempfile.mkdtemp(prefix="surah-model-methods-replay-"))
    with zipfile.ZipFile(TASK / "source.zip") as archive:
        archive.extractall(SOURCE)
    for name, expected in SNAPSHOT["files"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
    root = Path(SNAPSHOT["root"])
    for name in ("data", "models"):
        (SOURCE / name).symlink_to(root / name, target_is_directory=True)
sys.path.insert(0, str(SOURCE))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def prediction_guard():
    original = sqlite3.connect

    def guarded(database, *args, **kwargs):
        if "quran_mushaf.db" not in str(database):
            raise RuntimeError(f"Prediction cannot access a noncanonical database: {database}")
        return original(database, *args, **kwargs)

    sqlite3.connect = guarded
    from surah_cutter.data import reference_prior
    reference_prior.surah_start_fractions = lambda *_args, **_kwargs: {}
    from surah_cutter.core import align, model_store
    align.ALIGN_LAG_MS = 0
    model_store.lag_ms = lambda: 0


def corpus():
    return json.loads((TASK / "corpus.json").read_text())
