"""Serial study phases; keep model jobs and their logs independently inspectable."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, nargs="*", default=[])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--label", default="controller")
    args = parser.parse_args()
    for pid in args.wait_pid:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(2)
    ids = ["shatri-001", "shatri-080", "shatri-098", "shatri-112"]
    for rec in ["saad_al_ghamidi", "mahmoud_khalil_al_hussary", "mohammed_siddiq_al_minshawi",
                "abdul_basit_mujawwad"]:
        ids.extend([f"{rec}-080", f"{rec}-112"])
    app = ROOT / ".venv/bin/python"
    runtime = Path.home() / ".surah-cutter/model-runtime/bin/python"
    nemo = Path.home() / ".surah-cutter/nemo-runtime/bin/python"
    jobs = [
        ("qpc-blank", app, "predict_qpc.py", ["--variants", "blank_minus1"]),
        ("qpc-fixed30", app, "predict_qpc.py", ["--context", "fixed30", "--ids", *ids]),
        ("qpc-strong6", app, "predict_qpc.py", ["--context", "strong6", "--ids", *ids]),
        ("qpc-fp32", app, "predict_qpc.py", ["--precision", "fp32", "--variants", "current", "--ids", *ids]),
        ("qpc-localnorm", app, "predict_qpc.py", ["--normalization", "call", "--variants", "current", "phones", "--ids", *ids]),
        ("qpc-latency", app, "latency.py", []),
        ("qpc-controls", app, "controls_qpc.py", []),
        ("qpc-greedy", app, "qpc_greedy.py", []),
        ("parity", runtime, "parity.py", []),
        ("muaalem", runtime, "muaalem.py", []),
        ("nemo", nemo, "secondary_asr.py", ["nemo"]),
        ("nemo-forced", nemo, "nemo_forced.py", []),
        ("whisper", runtime, "secondary_asr.py", ["whisper"]),
        ("whisper-fast-methods", runtime, "secondary_asr.py", ["whisper", "--methods", "greedy", "token_timestamps", "teacher_attention", "greedy_sdpa"]),
        ("whisper-beam", runtime, "secondary_asr.py", ["whisper", "--methods", "beam5"]),
        ("whisper-overlap", runtime, "secondary_asr.py", ["whisper", "--methods", "greedy_overlap"]),
        ("whisper-nbest", runtime, "secondary_asr.py", ["whisper", "--methods", "nbest5"]),
        ("negative-muaalem", runtime, "negative_models.py", ["muaalem"]),
        ("nemo-negative", nemo, "negative_models.py", ["fastconformer"]),
        ("negative-whisper", runtime, "negative_models.py", ["whisper-tarteel-base"]),
        ("score-qpc", app, "score_qpc.py", ["strong20", "fixed30", "strong6"]),
    ]
    state = []
    for name, python, script, flags in jobs:
        if args.only and name not in args.only:
            continue
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        if name.startswith("nemo"):
            shim = TASK / "runtime-shim"
            if not shim.is_dir():
                shim = Path("/tmp/surah-cutter-asr-ml-dtypes-20261014")
            if shim.is_dir():
                env["PYTHONPATH"] = str(shim)
        command = [str(python), str(TASK / script), *flags]
        record = {"name": name, "command": command, "started": time.time(), "state": "running"}
        state.append(record)
        (TASK / (args.label + ".json")).write_text(json.dumps(state, indent=2) + "\n")
        with (TASK / (name + ".log")).open("w") as log:
            done = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=done.returncode, finished=time.time(), state="complete" if done.returncode == 0 else "failed")
        (TASK / (args.label + ".json")).write_text(json.dumps(state, indent=2) + "\n")
        print(name, record["state"], flush=True)


if __name__ == "__main__":
    main()
