"""One selected Base model: warm, exact replay, fresh readers, adoption/tests."""
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
for name in ['run_warm.py', 'run_final_e_transfer.py', 'finish_implementation.py']:
    print('start Base final stage', name, flush=True)
    subprocess.run([sys.executable, '-u', str(P / name)], check=True)
