import subprocess
import sys
from pathlib import Path

task = Path(__file__).parent.resolve()
for surah in [79,98]:
    with (task / f"probe-tiny-{surah}.log").open("w") as log:
        run = subprocess.run([sys.executable,str(task / "probe.py"),str(surah)],stdout=log,stderr=subprocess.STDOUT)
    print(surah,run.returncode,flush=True)
    if run.returncode:
        raise SystemExit(run.returncode)
