"""Profile runner with an explicit source root; -m cProfile changes import precedence."""
import cProfile
import json
from pathlib import Path
import runpy
import sys

task = Path(__file__).parent.resolve()
source = Path(json.loads((task / "snapshot-paths.json").read_text())["after"]).resolve()
sys.path.insert(0,str(source))
from surah_cutter.core import candidates  # noqa: E402
assert Path(candidates.__file__).resolve() == source / "surah_cutter/core/candidates.py"
profile = cProfile.Profile()
profile.enable()
try:
    runpy.run_path(str(task.parent / "2026-10-14-fast-sampled-onset-controls/named_alif_probe.py"),run_name="__main__")
finally:
    profile.disable()
    profile.dump_stats(str(task / "profile-after-2.pstats"))
