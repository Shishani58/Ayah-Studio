"""Release a suspended study process after the primary model queue completes."""
import argparse
import os
import signal
import time

from common import TASK, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int)
    parser.add_argument("--wait-pid", type=int, nargs="+", required=True)
    args = parser.parse_args()
    state = {"suspended_pid": args.pid, "wait_pids": args.wait_pid, "recorded_at": time.time()}
    write_json(TASK / "resource-suspension.json", state)
    try:
        for pid in args.wait_pid:
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(2)
    finally:
        try:
            os.kill(args.pid, signal.SIGCONT)
            state["resumed_at"] = time.time()
        except ProcessLookupError:
            state["already_finished_at"] = time.time()
        write_json(TASK / "resource-suspension.json", state)


if __name__ == "__main__":
    main()
