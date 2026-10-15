from pathlib import Path
import subprocess
import sys

task=Path(__file__).parent.resolve()
for surah in [79,98]:
    with (task / f"head-{surah}.log").open("w") as log:
        r=subprocess.run([sys.executable,str(task / "probe.py"),str(surah)],stdout=log,stderr=subprocess.STDOUT)
    print(surah,r.returncode,flush=True)
    if r.returncode:
        raise SystemExit(r.returncode)
