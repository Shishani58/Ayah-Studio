"""One Muaalem acoustic matrix, four phonetic reference constructions."""
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import quran_transcript

from common import SOURCE, TASK, digest, prediction_guard, write_json
from surah_cutter.core import align, model_store, model_runtime
from surah_cutter.core.model_worker import Acoustic
from surah_cutter.data import quran_text


def main():
    prediction_guard()
    model_store.ACTIVE_ID = "muaalem"
    entry = next(c for c in json.loads((SOURCE / "surah_cutter/core/model_catalog.json").read_text())
                 if c["id"] == "muaalem")
    folder = Path.home() / ".surah-cutter/models/muaalem" / entry["revision"]
    acoustic = Acoustic("muaalem", folder)
    actual_attrs = quran_transcript.MoshafAttributes
    state = {"madd": 4}

    def attrs(**kwargs):
        kwargs["madd_aared_len"] = state["madd"]
        return actual_attrs(**kwargs)

    quran_transcript.MoshafAttributes = attrs
    out = TASK / "muaalem"
    out.mkdir(exist_ok=True)
    for case in json.loads((TASK / "secondary-inputs.json").read_text()):
        if case["surah"] is None:
            continue
        pcm = np.load(case["input"], allow_pickle=False)
        matrix_path = out / f"{case['id']}-matrix.npy"
        if matrix_path.exists():
            matrix = np.load(matrix_path, allow_pickle=False)
            acoustic_s = json.loads(matrix_path.with_suffix(".json").read_text())["acoustic_s"]
        else:
            start = time.perf_counter()
            matrix = acoustic.emissions(pcm)
            acoustic_s = time.perf_counter() - start
            np.save(matrix_path, matrix, allow_pickle=False)
            write_json(matrix_path.with_suffix(".json"), {"acoustic_s": acoustic_s,
                                                          "PCM_sha": digest(pcm.tobytes()),
                                                          "matrix_sha": digest(matrix.tobytes())})
        words = quran_text.surah_words(case["surah"])
        for method in ("madd2", "madd4", "madd6", "connected4"):
            destination = out / f"{case['id']}-{method}.json"
            if destination.exists():
                continue
            state["madd"] = int(method[-1])

            def build_targets(selected, intro=False):
                ayahs = [1] * len(selected) if method == "connected4" else [w.ayah for w in selected]
                return acoustic.targets([w.text for w in selected], selected[0].surah, intro,
                                        renderings=model_runtime.inference_words(selected, "muaalem"), ayahs=ayahs)

            emitter = SimpleNamespace(model_id="muaalem", text_kind="muaalem", vocab=acoustic.vocab,
                                      blank_id=acoustic.blank_id, emissions=lambda _pcm: matrix,
                                      build_targets=build_targets)
            bundle = align._OnnxBundle(emitter)
            trace = {}
            start = time.perf_counter()
            result = {"case": case, "method": method, "matrix_sha": digest(matrix.tobytes()),
                      "acoustic_s": acoustic_s, "profile": {"madd_aared": state["madd"],
                      "other_madd": 4, "groups": "connected" if method == "connected4" else "ayah"}}
            try:
                ids, spans = build_targets(words, intro=case["surah"] not in (1, 9))
                result.update(target_sha=digest(np.asarray(ids, dtype=np.int64).tobytes()), target_count=len(ids))
                rows = align.align_surah(pcm, 16000, words, model=bundle, surah=case["surah"], trace=trace)
                assert len(rows) == quran_text.ayah_count(case["surah"])
                result["rows"] = rows
            except Exception:
                result["error"] = traceback.format_exc()
            result.update(alignment_s=time.perf_counter() - start, trace=trace)
            write_json(destination, result)
            print(case["id"], method, "ERROR" if "error" in result else "OK", flush=True)


if __name__ == "__main__":
    main()
