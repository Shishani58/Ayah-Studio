"""Short native scheduling check, including an idle cached acoustic session."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import webview

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from surah_cutter.core import decode, full_windows  # noqa: E402
from surah_cutter.core.emissions import _normalize  # noqa: E402

window = webview.create_window('Activity check', html='<p>Short activity check</p>', hidden=True)
parser = argparse.ArgumentParser()
parser.add_argument('--spin', action='store_true')
parser.add_argument('--power', action='store_true')
parser.add_argument('--compare-workers', action='store_true')
args = parser.parse_args()


@contextmanager
def basic_activity(critical=False):
    from Foundation import NSProcessInfo, NSActivityUserInitiated
    from objc import autorelease_pool

    with autorelease_pool():
        process = NSProcessInfo.processInfo()
        flags = NSActivityUserInitiated
        if critical:
            from Foundation import NSActivityLatencyCritical

            flags |= NSActivityLatencyCritical
        token = process.beginActivityWithOptions_reason_(flags, 'Short scheduling check')
        try:
            yield
        finally:
            process.endActivity_(token)


def usage():
    # Public SDK layout, own-process counters only. No elevated privileges.
    source = Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/resource.h').read_text()
    source = source.split('struct rusage_info_v6 {')[1].split('};')[0]
    fields = []
    for kind, name, count in re.findall(r'(uint8_t|uint64_t)\s+(\w+)(?:\[(\d+)\])?;', source):
        value = ctypes.c_uint8 if kind == 'uint8_t' else ctypes.c_uint64
        fields.append((name, value * int(count) if count else value))

    class Usage(ctypes.Structure):
        _fields_ = fields

    info = Usage()
    code = ctypes.CDLL('/usr/lib/libproc.dylib').proc_pid_rusage(os.getpid(), 6, ctypes.byref(info))
    assert code == 0
    return {name: getattr(info, name) for name in ('ri_energy_nj', 'ri_instructions', 'ri_cycles')}


def worker():
    import onnxruntime as ort

    result = []
    try:
        with basic_activity():
            raw = decode.decode('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3')
            spans = full_windows.plan_pause_windows(len(raw), full_windows.detect_strong_pauses(raw),
                                                   window_s=20, pause_only=True)[:6]
            pcm = _normalize(raw)
            clips = [np.pad(pcm[max(0, a - 32000):b + 32000], (max(0, 32000 - a), 0))[None]
                     for a, b, _ in spans]
            del raw, pcm
            model = str(ROOT / 'models/wav2vec2-quran-phonetics.onnx')
            if not args.spin:
                session = ort.InferenceSession(model, providers=['CPUExecutionProvider'])

        if args.power:
            with basic_activity(), ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda clip: session.run(['logits'], {'input_values': clip})[0], clips))
        for idle in ((0, 0, 0) if args.spin or args.power else (0, 35)):
            if idle:
                time.sleep(idle)
            critical = args.power and not args.compare_workers and len(result) != 1
            with basic_activity(critical=critical):
                if args.spin:
                    options = ort.SessionOptions()
                    spinning = len(result) != 1
                    if not spinning:
                        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
                    session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
                started, cpu = time.perf_counter(), time.process_time()
                if args.power:
                    before = usage()
                workers = 2 if args.compare_workers and len(result) == 1 else 3
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    arrays = list(pool.map(lambda clip: session.run(['logits'], {'input_values': clip})[0], clips))
                item = dict(idle_s=idle, wall_s=time.perf_counter() - started,
                            cpu_s=time.process_time() - cpu,
                            sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays])
                if args.spin:
                    item['spinning'] = spinning
                    del session
                if args.power:
                    after = usage()
                    item['critical'] = critical
                    item['usage_delta'] = {k: after[k] - before[k] for k in before}
                if args.compare_workers:
                    item['workers'] = workers
                result.append(item)
                print(json.dumps({k: v for k, v in item.items() if k != 'sha256'}), flush=True)
        # Comparison happens only after saving the newly calculated arrays' hashes.
        output = Path(__file__).with_name('native-workers.json' if args.compare_workers else
                                         'native-power.json' if args.power else
                                         'native-spin.json' if args.spin else 'native-basic-activity.json')
        output.write_text(json.dumps(result, indent=2) + '\n')
        old = json.loads((ROOT / 'docs/tasks/2026-10-16-full-runtime-regression/native-pool-critical-only.json').read_text())
        assert all(item['sha256'] == old[0]['sha256'] for item in result)
        print('All six logits arrays exactly equal to the prior critical-activity run', flush=True)
    finally:
        window.destroy()


webview.start(worker)
