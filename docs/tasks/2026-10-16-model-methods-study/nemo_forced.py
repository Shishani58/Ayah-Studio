"""Known-text CTC: fixed SentencePiece versus equivalent acoustic BPE on one matrix."""
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import SOURCE, TASK, digest, prediction_guard, write_json
from surah_cutter.core import align, model_runtime, model_store
from surah_cutter.core.model_worker import Acoustic
from surah_cutter.data import quran_text


def main():
    prediction_guard()
    torch.set_num_threads(4)
    model_store.ACTIVE_ID = "fastconformer"
    entry = next(c for c in json.loads((SOURCE / "surah_cutter/core/model_catalog.json").read_text())
                 if c["id"] == "fastconformer")
    folder = Path.home() / ".surah-cutter/models/fastconformer" / entry["revision"]
    acoustic = Acoustic("fastconformer", folder)
    out = TASK / "nemo-forced"
    out.mkdir(exist_ok=True)
    write_json(out / "identity.json", dict(revision=entry["revision"],
               checkpoint_sha=digest(next(folder.rglob("*.nemo")).read_bytes()),
               torch=torch.__version__, source=str(SOURCE)))
    for case in json.loads((TASK / "secondary-inputs.json").read_text()):
        if case["surah"] is None:
            continue
        pcm = np.load(case["input"], allow_pickle=False)
        assert digest(pcm.tobytes()) == case["pcm_sha"]
        matrix_path = out / f"{case['id']}-matrix.npy"
        if matrix_path.exists():
            matrix = np.load(matrix_path, allow_pickle=False)
            meta = json.loads(matrix_path.with_suffix(".json").read_text())
            assert meta["pcm_sha"] == case["pcm_sha"]
            variants = meta["bpe_variants"]
        else:
            start = time.perf_counter()
            matrix = acoustic.emissions(pcm)
            variants = acoustic.bpe_variants
            meta = dict(pcm_sha=case["pcm_sha"], matrix_sha=digest(matrix.tobytes()),
                        acoustic_s=time.perf_counter()-start, bpe_variants=variants)
            np.save(matrix_path, matrix, allow_pickle=False)
            write_json(matrix_path.with_suffix(".json"), meta)
        words = quran_text.surah_words(case["surah"])
        for method in ("forced_default_bpe", "forced_acoustic_bpe"):
            destination = out / f"{case['id']}-{method}.json"
            if destination.exists():
                continue
            acoustic.bpe_variants = {} if method == "forced_default_bpe" else variants

            def build_targets(selected, intro=False):
                return acoustic.targets([w.text for w in selected], selected[0].surah, intro,
                                        renderings=model_runtime.inference_words(selected, "fastconformer"))

            emitter = SimpleNamespace(model_id="fastconformer", text_kind="fastconformer",
                                      vocab=acoustic.vocab, blank_id=acoustic.blank_id,
                                      emissions=lambda _pcm: matrix, build_targets=build_targets)
            trace = {}
            start = time.perf_counter()
            result = dict(case=case, method=method, matrix_sha=meta["matrix_sha"],
                          acoustic_s=meta["acoustic_s"], known_text_supplied=True)
            try:
                ids, _ = build_targets(words, intro=case["surah"] not in (1, 9))
                result.update(target_sha=digest(np.asarray(ids,dtype=np.int64).tobytes()), target_count=len(ids))
                result["rows"] = align.align_surah(pcm,16000,words,model=align._OnnxBundle(emitter),
                                                    surah=case["surah"],trace=trace)
                assert len(result["rows"]) == quran_text.ayah_count(case["surah"])
            except Exception:
                result["error"] = traceback.format_exc()
            result.update(trace=trace, alignment_s=time.perf_counter()-start)
            write_json(destination, result)
            print(case["id"],method,"ERROR" if "error" in result else "OK",flush=True)


if __name__ == "__main__":
    main()
