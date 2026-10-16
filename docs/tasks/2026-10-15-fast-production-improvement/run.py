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


def host_usage():
    rows = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,pcpu=,rss=,comm='], text=True).splitlines():
        values = line.strip().split(None, 3)
        if len(values) == 4:
            rows.append({'pid': int(values[0]), 'cpu_percent': float(values[1]),
                         'rss_kib': int(values[2]), 'executable': values[3]})
    return {'unix': time.time(), 'top_cpu': sorted(rows, key=lambda r: r['cpu_percent'], reverse=True)[:12]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--variant', default='baseline')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--surahs', nargs='+', type=int, default=A)
    parser.add_argument('--reciter')
    args = parser.parse_args()
    label = args.label
    requested_models = list(args.models)
    single_model = P / 'single-model-selection.json'
    if single_model.exists():
        args.models = [json.loads(single_model.read_text())['model']]
    assert not (P / f'{label}-command.json').exists(), 'Never repeat a launched ID'
    effective_variant = args.variant
    selection_path = P / 'selected-wide-candidate.json'
    if args.variant == 'prototype-c' and label.startswith(('wide-', 'warm-')) and selection_path.exists():
        selection = json.loads(selection_path.read_text())
        effective_variant = selection.get('warm_variant', selection['variant']) if label.startswith('warm-') else selection['variant']
    root = Path(json.loads((P / 'snapshot-paths.json').read_text())[effective_variant]).resolve()
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
                                  'started_unix': time.time(), 'variant_requested': args.variant, 'variant_effective': effective_variant,
                                  'models_requested': requested_models, 'models_effective': args.models})
    if label == 'wide-prototype-c' and selection_path.exists():
        hook = json.loads(selection_path.read_text()).get('pre_wide_hook')
        if hook and not (P / f"{hook['label']}-command.json").exists():
            hook_code = subprocess.run([sys.executable, str(P / 'run_aux.py'), hook['label'], hook['script']]).returncode
            dump(f'{label}-pre-hook.json', {'hook': hook, 'exit': hook_code, 'scope': 'separate CTC diagnostic, no ASR comparison change'})
    host_before = host_usage() if label.startswith('warm-') else None
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
         'tree_peak_kib': max((x['tree_kib'] for x in rss), default=0),
         'host_before': host_before, 'host_after': host_usage() if host_before else None})
    checkpoint = json.loads((P / 'checkpoint.json').read_text())
    for item in checkpoint['processes']:
        if item['label'] == label:
            item.update(status='completed', exit=code)
    dump('checkpoint.json', checkpoint)
    print(label, 'exit', code, flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
