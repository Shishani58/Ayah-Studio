"""Known-text controls on zeros and wrong Quran audio, with one positive control."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["muaalem", "fastconformer", "whisper-tarteel-base"])
    args = parser.parse_args()
    prediction_guard()
    torch.set_num_threads(4)
    model_store.ACTIVE_ID = args.model
    entry = next(c for c in json.loads((SOURCE / "surah_cutter/core/model_catalog.json").read_text())
                 if c["id"] == args.model)
    folder = Path.home() / ".surah-cutter/models" / args.model / entry["revision"]
    acoustic = Acoustic(args.model,folder)
    out = TASK / "known-controls"
    out.mkdir(exist_ok=True)
    tests = [("zero4",112,"zero"), ("zero_matched",112,"zero-matched"),
             ("shatri-112",114,"wrong-surah"), ("shatri-112",112,"positive")]
    for record, target, control in tests:
        destination = out / f"{args.model}-{control}.json"
        if destination.exists():
            continue
        pcm = np.load(TASK / "secondary-inputs" / f"{'shatri-112' if record == 'zero_matched' else record}.npy", allow_pickle=False)
        if record == "zero_matched":
            pcm = np.zeros_like(pcm)
        words = quran_text.surah_words(target)
        trace = {}
        start = time.perf_counter()
        result = dict(model=args.model, revision=entry["revision"], input=record, control=control,
                      target=target, pcm_sha=digest(pcm.tobytes()), known_text_supplied=True)
        try:
            if args.model.startswith("whisper"):
                intro = model_runtime.inference_words(quran_text.surah_words(1)[:4],args.model)
                rendered = intro + model_runtime.inference_words(words,args.model)
                word_rows = acoustic.whisper_align(pcm,rendered,strict_words=False)
                owners = [0]*len(intro) + [w.ayah for w in words]
                first = {}
                for a, row in zip(owners,word_rows):
                    if a:
                        first.setdefault(a,(a,row[0],row[1]))
                rows = list(first.values())
                checks = [dict(ayah=a,onset_ms=ms,conf=c,reliable=c>=.25) for a,ms,c in rows]
                result["word_rows"] = word_rows
                trace["ayahs"] = checks
            else:
                matrix = acoustic.emissions(pcm)

                def targets(selected,intro=False):
                    return acoustic.targets([w.text for w in selected],target,intro,
                                            renderings=model_runtime.inference_words(selected,args.model),
                                            ayahs=[w.ayah for w in selected])

                emitter = SimpleNamespace(model_id=args.model,text_kind=args.model,vocab=acoustic.vocab,
                                          blank_id=acoustic.blank_id,emissions=lambda _:matrix,
                                          build_targets=targets)
                rows = align.align_surah(pcm,16000,words,model=align._OnnxBundle(emitter),
                                         surah=target,trace=trace)
            result.update(rows=rows,expected_ayahs=quran_text.ayah_count(target),
                          reliable=sum(bool(c.get("reliable")) for c in trace.get("ayahs",[])),
                          nonincreasing_pairs=sum(b[1]<=a[1] for a,b in zip(rows,rows[1:])))
        except Exception:
            result["error"] = traceback.format_exc()
        result.update(trace=trace,wall_s=time.perf_counter()-start)
        write_json(destination,result)
        print(args.model,control,"ERROR" if "error" in result else "OK",flush=True)


if __name__ == "__main__":
    main()
