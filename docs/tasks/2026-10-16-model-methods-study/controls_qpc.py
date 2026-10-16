"""Negative FULL inputs and a known artificial shift of genuine recitation."""
import sys

import numpy as np
import soundfile as sf

from common import TASK, corpus, digest, write_json
import predict_qpc


def main():
    out = TASK / "qpc-controls"
    out.mkdir(exist_ok=True)
    cache = out / "qpc-logits"
    if not cache.exists():
        cache.symlink_to(TASK / "qpc-logits", target_is_directory=True)
    cases = []
    existing = {c["id"]: c for c in corpus()}
    for record, target in [("shatri-001", 112), ("shatri-080", 112), ("shatri-112", 114)]:
        case = dict(existing[record], id=f"wrong-{record}-target{target}", surah=target,
                    control="wrong-canonical-surah", split="negative")
        cases.append(case)
    zero = TASK / "secondary-inputs/zero4.wav"
    cases.append({"id": "zero-target112", "audio": str(zero), "sha256": digest(zero.read_bytes()),
                  "surah": 112, "reciter": None, "split": "negative", "control": "zero"})
    original = np.load(TASK / "secondary-inputs/shatri-112.npy", allow_pickle=False)
    matched_zero = out / "zero-matched112.wav"
    sf.write(matched_zero, np.zeros_like(original), 16000, subtype="FLOAT")
    cases.append({"id": "zero-matched-target112", "audio": str(matched_zero),
                  "sha256": digest(matched_zero.read_bytes()), "surah":112,"reciter":None,
                  "split":"negative","control":"zero-matched-positive-duration"})
    shifted = np.concatenate((np.zeros(9600, dtype=np.float32), original))
    wav = out / "shift600-shatri112.wav"
    sf.write(wav, shifted, 16000, subtype="FLOAT")
    cases.append({**existing["shatri-112"], "id": "shift600-shatri112", "audio": str(wav),
                  "sha256": digest(wav.read_bytes()), "control": "shift", "split": "shift",
                  "expected_shift_ms": 600})
    write_json(out / "manifest.json", cases)
    predict_qpc.TASK = out
    predict_qpc.corpus = lambda: cases
    sys.argv = [sys.argv[0], "--variants", "current", "no_delimiter", "phones", "free_edges", "blank_minus1"]
    predict_qpc.main()


if __name__ == "__main__":
    main()
