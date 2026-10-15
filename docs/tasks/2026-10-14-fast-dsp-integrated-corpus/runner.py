"""Explicit source import guard for the combined corpus run."""
import json
from pathlib import Path
import runpy
import sys

task = Path(__file__).parent.resolve()
source = Path((task / "snapshot-path.txt").read_text().strip()).resolve()
sys.path.insert(0,str(source))
from surah_cutter.core import candidates  # noqa: E402
assert Path(candidates.__file__).resolve() == source / "surah_cutter/core/candidates.py"
(task / "loaded-source.json").write_text(json.dumps({"candidates":str(Path(candidates.__file__).resolve())},indent=2)+"\n")
runpy.run_path(str(task.parent / "2026-10-14-fast-sampled-onset-controls/named_alif_probe.py"),run_name="__main__")
