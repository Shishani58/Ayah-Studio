"""One Base: F same-input replay, safety gate, warm, transfer, production/tests."""
from pathlib import Path
import subprocess
import sys
P = Path(__file__).resolve().parent
jobs = [('run_aux.py', 'f-wide-replay', 'replay_f_wide.py', 'f-wide-replay'),
        ('compare_runs.py', 'f-wide'), ('run_warm.py',),
        ('run_final_f_transfer.py',), ('finish_implementation.py',)]
for job in jobs:
    print('start Base F stage', job, flush=True)
    subprocess.run([sys.executable, '-u', str(P / job[0]), *job[1:]], check=True)
