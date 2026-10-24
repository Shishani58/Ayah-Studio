"""Real hidden pywebview FULL; no project DB access and no timing inputs.

Run with repository .venv/bin/python. Outputs are separate diagnostic JSON files.
The saved comparison run is read only AFTER process_surah has returned.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import webview  # noqa: E402

from surah_cutter.core import pipeline  # noqa: E402
from surah_cutter.core.cancellation import cancellation_scope  # noqa: E402
from surah_cutter.core.progress import progress_scope  # noqa: E402
from surah_cutter.webgui.activity import generation_activity  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--unprotected", action="store_true")
parser.add_argument("--surah", type=int, default=2)
parser.add_argument("--serial-ort", action="store_true")
parser.add_argument("--workers", type=int)
args = parser.parse_args()
window = webview.create_window("Ayah Studio runtime verification", html="<p>FULL verification</p>", hidden=True)


def worker():
    result = {"protected": not args.unprotected, "surah": args.surah}
    result["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*(ROOT / "surah_cutter/core").glob("*.py"), ROOT / "surah_cutter/webgui/activity.py"]
    }
    started = time.perf_counter()
    cpu = time.process_time()
    trace = {}
    last = [0.0]

    def progress(event):
        now = time.perf_counter()
        if now - last[0] > 20:
            print(round(now - started, 2), event, flush=True)
            last[0] = now

    try:
        with (nullcontext() if args.unprotected else generation_activity(),
              cancellation_scope(threading.Event()), progress_scope(progress)):
            if args.workers:
                from surah_cutter.core import full_windows

                full_windows.FULL_WORKERS = args.workers
            if args.serial_ort:
                import onnxruntime as ort
                from surah_cutter.core import align, emissions, full_windows, model_store

                path = model_store.runtime_path("quran-phonetics")
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                session = ort.InferenceSession(str(path), sess_options=options,
                                              providers=["CPUExecutionProvider"])
                vocab, normalize = emissions._load_sidecar(path, None)
                emitter = emissions.OnnxEmitter(session, vocab,
                                               blank_id=emissions._resolve_blank_id(vocab),
                                               do_normalize=normalize,
                                               weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                align._ONNX_CACHE[str(path)] = align._OnnxBundle(emitter)
                full_windows.FULL_WORKERS = 6
            rows = pipeline.process_surah(
                f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{args.surah:03d}.mp3",
                args.surah, mode="full", trace=trace,
            )
        result.update(wall_s=time.perf_counter() - started, cpu_s=time.process_time() - cpu,
                      rows=rows, trace=trace)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        if args.surah == 2:
            previous = Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/Abu_Bakr_Ash-Shatri.logs/runs/"
                            "sura_002_full_f3d8581a7d214b65afee83af76ccfd39.json")
            saved = json.loads(previous.read_text())
            result["comparison"] = {
                "run": previous.name,
                "rows_identical": [list(row) for row in rows] == saved["rows"],
                "confidences_identical": [a["conf"] for a in trace["ayahs"]]
                == [a["conf"] for a in saved["trace"]["ayahs"]],
                "windows_identical": trace["full_windows"]["windows"]
                == saved["trace"]["full_windows"]["windows"],
            }
        result["source_sha256_after"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*(ROOT / "surah_cutter/core").glob("*.py"), ROOT / "surah_cutter/webgui/activity.py"]
        }
        result["source_unchanged"] = result["source_sha256"] == result["source_sha256_after"]
    except BaseException:
        result.update(error=traceback.format_exc(), wall_s=time.perf_counter() - started)
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("trace", "rows", "source_sha256", "source_sha256_after")}), flush=True)
        window.destroy()


webview.start(worker)
