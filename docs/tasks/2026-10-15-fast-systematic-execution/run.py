"""Serial fresh audio runner, real import guard and simultaneous tree RSS."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
REPO = P.parents[2]
MODELS = ['quran-phonetics', 'quran-phonetics-fp32',
          'whisper-tarteel-tiny', 'whisper-tarteel-base']
A = [1, 32, 55, 59, 65, 73, 77, 80, 98, 112, 2, 79]


def dump(name, obj):
    (P / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--variant', default='baseline')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--surahs', nargs='+', type=int, default=A)
    parser.add_argument('--reciter')
    args = parser.parse_args()
    label = args.label
    assert not (P / f'{label}-command.json').exists(), 'Never repeat a launched ID'
    root = Path(json.loads((P / 'snapshot-paths.json').read_text())[args.variant]).resolve()
    audio = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri')
    ref = REPO / 'data/references/abu_bakr_ash_shatri.db'
    if args.reciter:
        audio = Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / args.reciter
        ref = audio / f'{args.reciter}.db'
    command = [sys.executable, '-u', '-c', '''
import hashlib,json,os,runpy,sys
from pathlib import Path
import surah_cutter.core.fast_executor as e
import surah_cutter.core.fast_adapters as a
import surah_cutter.core.fast_ctc as c
root=Path.cwd().resolve()
paths={m.__name__:str(Path(m.__file__).resolve()) for m in (e,a,c)}
assert all(Path(p).is_relative_to(root) for p in paths.values()), paths
out=Path(sys.argv[1]);sys.argv=sys.argv[2:]
out.write_text(json.dumps({'cwd':str(root),'modules':paths,'policy':vars(e.Policy()),
 'sha256':{n:hashlib.sha256(Path(p).read_bytes()).hexdigest() for n,p in paths.items()}},indent=2))
runpy.run_path(str(root/'tools/bench_fast_models.py'),run_name='__main__')
''', str(P / f'{label}-imports.json'), str(root / 'tools/bench_fast_models.py'),
        '--models', *args.models, '--surahs', *map(str, args.surahs),
        '--audio-dir', str(audio), '--reference', str(ref), '--output', str(P / f'{label}.json')]
    dump(f'{label}-command.json', {'argv': command, 'cwd': str(root), 'PYTHONPATH': str(root),
                                  'started_unix': time.time()})
    with (P / f'{label}.log').open('w') as log:
        started = time.monotonic()
        proc = subprocess.Popen(command, cwd=root, env={**os.environ, 'PYTHONPATH': str(root)},
                                stdout=log, stderr=subprocess.STDOUT)
        checkpoint = json.loads((P / 'checkpoint.json').read_text())
        checkpoint['processes'].append({'label': label, 'pid': proc.pid, 'root': str(root),
                                        'log': str(P / f'{label}.log'), 'status': 'running'})
        dump('checkpoint.json', checkpoint)
        rss = []
        while proc.poll() is None:
            raw = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
            rows = [list(map(int, line.split())) for line in raw.splitlines()]
            tree = {proc.pid}
            for _ in range(8):
                tree.update(pid for pid, parent, _ in rows if parent in tree)
            members = {str(pid): kb for pid, _, kb in rows if pid in tree}
            rss.append({'elapsed_s': time.monotonic() - started, 'rss_kib': members,
                        'tree_kib': sum(members.values())})
            time.sleep(.2)
        code = proc.wait()
    dump(f'{label}-process.json', {'pid': proc.pid, 'exit': code,
         'wall_s': time.monotonic() - started, 'sampling_s': .2, 'samples': rss,
         'tree_peak_kib': max((x['tree_kib'] for x in rss), default=0)})
    checkpoint = json.loads((P / 'checkpoint.json').read_text())
    for item in checkpoint['processes']:
        if item['label'] == label:
            item.update(status='completed', exit=code)
    dump('checkpoint.json', checkpoint)
    print(label, 'exit', code, flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
