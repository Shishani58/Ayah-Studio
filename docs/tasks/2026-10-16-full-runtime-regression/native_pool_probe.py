"""Isolate native ORT pool scheduling; every case reads identical PCM windows."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import webview

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from surah_cutter.core.decode import decode  # noqa: E402
from surah_cutter.core.emissions import _normalize, _run  # noqa: E402
from surah_cutter.core.cancellation import cancellation_scope  # noqa: E402
from surah_cutter.webgui.activity import generation_activity  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--six", action="store_true")
parser.add_argument("--delay", type=float, default=0)
parser.add_argument("--basic-activity", action="store_true")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
window = webview.create_window("Native ORT scheduling probe", html="<p>Probe</p>", hidden=True)


def worker():
    if args.basic_activity:
        import Foundation

        Foundation.NSActivityLatencyCritical = 0  # Only this disposable diagnostic process.
    results = []
    with generation_activity():
        import onnxruntime as ort

        raw = decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3")
        # DSP-only plan, with no ayah timings supplied to the model.
        from surah_cutter.core.full_windows import detect_strong_pauses, plan_pause_windows

        spans = plan_pause_windows(len(raw), detect_strong_pauses(raw), window_s=20, pause_only=True)
        spans = spans[:6 if args.six else 32]
        pcm = _normalize(raw)
        clips = []
        for left, right, _ in spans:
            left, right = left - 32000, right + 32000
            clips.append(np.pad(pcm[max(0, left):right], (max(0, -left), 0))[None])
        previous = None
        cases = ([(3, False, True)] if args.six else
                 [(3, False, True), (2, False, True), (3, True, True),
                  (3, False, False), (3, False, True)])
        for workers, dynamic, arena in cases:
            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = arena
            if dynamic:
                options.add_session_config_entry("session.dynamic_block_base", "4")
            session = ort.InferenceSession(str(ROOT / "models/wav2vec2-quran-phonetics.onnx"),
                                          sess_options=options, providers=["CPUExecutionProvider"])
            if args.delay:
                time.sleep(args.delay)
            lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

            def run(clip, loaded_session=session):
                with cancellation_scope(threading.Event()):
                    return _run(loaded_session, clip, "input_values", "logits"), lib.qos_class_self()

            started, cpu = time.perf_counter(), time.process_time()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                values = list(pool.map(run, clips))
            arrays = [v for v, _ in values]
            item = {"workers": workers, "dynamic": dynamic, "arena": arena, "wall_s": time.perf_counter() - started,
                    "cpu_s": time.process_time() - cpu, "caller_qos": [q for _, q in values],
                    "sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]}
            if previous is not None:
                item["max_abs_delta"] = max(float(np.max(np.abs(a - b))) for a, b in zip(arrays, previous))
            results.append(item)
            print(json.dumps({k: v for k, v in item.items() if k not in ("sha256", "caller_qos")}), flush=True)
            if previous is None:
                previous = arrays
            del session
    output = args.output or Path(__file__).with_name("native-pool-matrix.json")
    output.write_text(json.dumps(results, indent=2))
    window.destroy()


webview.start(worker)
