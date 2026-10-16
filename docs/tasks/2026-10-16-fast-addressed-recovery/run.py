"""Fresh isolated Base executor runs with exact dispatch/ledger and tree RSS."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
REPO = P.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('variant', choices=('A', 'AP', 'B', 'B2', 'B3', 'B3P', 'B4', 'B4P', 'B5', 'B6', 'S', 'SF', 'PF', 'PFP', 'BS', 'BS2', 'BS2P', 'C', 'D', 'legacy', 'current', 'production'))
parser.add_argument('label')
parser.add_argument('--surahs', nargs='+', type=int, default=[2, 1])
parser.add_argument('--reciter')
parser.add_argument('--frozen-inputs', action='store_true')
parser.add_argument('--fill-missing', action='store_true')
args = parser.parse_args()
root = Path(json.loads((P / 'snapshot-paths.json').read_text())[args.variant])
audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri')
reference = REPO / 'data/references/abu_bakr_ash_shatri.db'
if args.reciter:
    audio = Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / args.reciter
    reference = audio / (args.reciter + '.db')
if args.frozen_inputs:
    assert args.variant == 'PF' and args.reciter is None
script = 'frozen_bench_fast.py' if args.frozen_inputs else 'bench_fast_models.py'
command = [sys.executable, '-u', str(root / 'tools' / script),
           '--models', 'whisper-tarteel-base', '--surahs', *map(str, args.surahs),
           '--audio-dir', str(audio), '--reference', str(reference),
           '--output', str(P / (args.label + '.json'))]
if args.fill_missing:
    assert args.variant == 'PF' and args.reciter is None and not args.frozen_inputs
    command = [sys.executable, '-u', str(P / 'ablation_replay.py'), 'PF', 'A-wide', 'BS-wide',
               args.label, '--fill-missing', '--surahs', *map(str, args.surahs)]


def host_cpu():
    rows = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,pcpu=,comm='], text=True).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            rows.append({'pid': int(fields[0]), 'cpu_percent': float(fields[1]), 'command': fields[2]})
    return sorted(rows, key=lambda row: row['cpu_percent'], reverse=True)[:10]


metadata = P / (args.label + '-command.json')
assert not metadata.exists() and not (P / (args.label + '.json')).exists()
metadata.write_text(json.dumps({'argv': command, 'cwd': str(root), 'PYTHONPATH': str(root),
                               'profile_directory': os.environ.get('FAST_PROFILE_DIR'),
                               'started_unix': time.time()}, indent=2) + '\n')
with (P / (args.label + '.log')).open('w') as log:
    host_before = host_cpu()
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=root, env={**os.environ, 'PYTHONPATH': str(root)},
                               stdout=log, stderr=subprocess.STDOUT)
    samples = []
    while process.poll() is None:
        rows = [list(map(int, line.split())) for line in subprocess.check_output(
            ['ps', '-axo', 'pid=,ppid=,rss='], text=True).splitlines()]
        members = {process.pid}
        for _ in range(8):
            members.update(pid for pid, parent, _ in rows if parent in members)
        samples.append({'elapsed_s': time.monotonic() - started,
                        'tree_rss_kib': sum(kb for pid, _, kb in rows if pid in members)})
        time.sleep(.5)
    code = process.wait()
(P / (args.label + '-process.json')).write_text(json.dumps({'exit': code,
    'wall_s': time.monotonic() - started, 'samples': samples,
    'tree_peak_kib': max((s['tree_rss_kib'] for s in samples), default=0),
    'host_before': host_before, 'host_after': host_cpu()}, indent=2) + '\n')
print(args.label, 'exit', code, flush=True)
raise SystemExit(code)
