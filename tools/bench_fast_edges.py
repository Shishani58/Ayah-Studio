"""Derived-audio contract probes; no reference or claimed listening verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from surah_cutter.core import align, candidates, decode, model_store
from surah_cutter.core.fast_executor import align_surah_sequential
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, validate, verify_dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = decode.decode(str(args.audio))
    # Fixed sample edits of surah 108. They are reproducible stress cases,
    # not independently verified ground-truth recordings or another reciter.
    cases = {
        "crop_after_intro_candidate_6320ms": original[101120:],
        "leading_silence_12s": np.concatenate((np.zeros(192000, dtype=np.float32), original)),
        "repeat_segment_6320_10460ms": np.concatenate((original[:167360], original[101120:167360], original[167360:])),
        "truncated_at_8s": original[:128000],
        "very_short_500ms": original[:8000],
    }
    words = quran_text.surah_words(108)
    decisions = {name: candidates.pause_pipeline_decisions(pcm, 16000, surah=108)
                 for name, pcm in cases.items()}
    outputs = []
    for model in ("quran-phonetics", "whisper-tarteel-tiny", "whisper-tarteel-base"):
        model_store.ACTIVE_ID = model
        align._ONNX_CACHE.clear()
        release_runtimes()
        bundle = align.default_bundle()
        for name, pcm in cases.items():
            trace = {}
            started = time.monotonic()
            with dispatch_spy(bundle, pcm) as calls:
                rows = align_surah_sequential(pcm, 16000, words, bundle=bundle,
                                             trace=trace, pause_decisions=decisions[name])
            verify_dispatch(calls, pcm, trace)
            assert not validate(rows, trace, [1, 2, 3])
            if name == "very_short_500ms":
                assert not calls and not rows and trace["unresolved_ayahs"] == [1, 2, 3]
            outputs.append({"case": name, "model": model, "seconds": time.monotonic()-started,
                            "pcm_sha256": hashlib.sha256(memoryview(pcm)).hexdigest(),
                            "rows": rows, "trace": trace, "actual_dispatches": calls})
            args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2)+"\n")
            print(model, name, "confirmed", sum(conf > 0 for _, _, conf in rows), flush=True)
    release_runtimes()


if __name__ == "__main__":
    main()
