"""Serial factorial, frozen source and late reference evaluation."""
import os
from pathlib import Path
import subprocess
import sys

task = Path(__file__).parent.resolve()
root = task.parents[2]
source = Path((task / "snapshot-path.txt").read_text().strip())
variants = {
    "control": (1300, 2500, 4000),
    "short_initial": (1000, 2500, 4000),
    "short_expansion": (1300, 2000, 4000),
    "short_both": (1000, 2000, 4000),
}
for name, widths in variants.items():
    args = [
        sys.executable, "-u", str(task / "variant.py"), ",".join(map(str, widths)),
        "--models", "whisper-tarteel-base", "--surahs",
        *map(str, (1, 32, 55, 59, 65, 73, 77, 80, 98, 112)),
        "--audio-dir", "/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri",
        "--reference", str(root / "data/references/abu_bakr_ash_shatri.db"),
        "--output", str(task / (name + ".json")),
    ]
    print("start", name, flush=True)
    with (task / (name + ".log")).open("w") as log:
        result = subprocess.run(args, cwd=source, env={**os.environ, "PYTHONPATH": str(source)},
                                stdout=log, stderr=subprocess.STDOUT)
    print("completed", name, "exit", result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
