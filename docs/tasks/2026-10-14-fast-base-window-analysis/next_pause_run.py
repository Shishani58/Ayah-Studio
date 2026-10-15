"""Paired real coverage for bounded DSP lookahead; no production edits."""
import os
from pathlib import Path
import subprocess
import sys

task = Path(__file__).parent.resolve()
root = task.parents[2]
source = Path((task / "snapshot-path.txt").read_text().strip())
main = (1, 32, 55, 59, 65, 73, 77, 80, 98, 112)
cases = [
    ("other_controls", source, ("quran-phonetics", "whisper-tarteel-tiny"), (*main, 2)),
    ("base_long_control", source, ("whisper-tarteel-base",), (2,)),
    ("next_pause", source.parent / "next-pause",
     ("quran-phonetics", "whisper-tarteel-tiny", "whisper-tarteel-base"), (*main, 2)),
]
for name, folder, models, surahs in cases:
    args = [sys.executable, "-u", "-m", "tools.bench_fast_models", "--models", *models,
            "--surahs", *map(str, surahs), "--audio-dir", "/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri",
            "--reference", str(root / "data/references/abu_bakr_ash_shatri.db"),
            "--output", str(task / (name + ".json"))]
    print("start", name, flush=True)
    with (task / (name + ".log")).open("w") as log:
        result = subprocess.run(args, cwd=folder, env={**os.environ, "PYTHONPATH": str(folder)},
                                stdout=log, stderr=subprocess.STDOUT)
    print("completed", name, "exit", result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
