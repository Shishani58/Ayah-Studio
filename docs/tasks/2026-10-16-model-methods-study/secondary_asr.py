"""Same installed weights, independent decoders on identical FULL recordings."""
import argparse
import copy
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from common import SOURCE, TASK, digest, prediction_guard, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["whisper", "nemo"])
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    prediction_guard()
    torch.set_num_threads(4)
    inputs = json.loads((TASK / "secondary-inputs.json").read_text())
    catalog = json.loads((SOURCE / "surah_cutter/core/model_catalog.json").read_text())
    mid = "whisper-tarteel-base" if args.model == "whisper" else "fastconformer"
    entry = next(c for c in catalog if c["id"] == mid)
    folder = Path.home() / ".surah-cutter/models" / mid / entry["revision"]
    if args.model == "whisper":
        from surah_cutter.core.model_worker import Acoustic
        acoustic = Acoustic(mid, folder)
        model = acoustic.model
        methods = ["greedy", "token_timestamps", "teacher_attention", "greedy_sdpa", "beam5", "greedy_overlap", "nbest5"]
    else:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        model = EncDecHybridRNNTCTCBPEModel.restore_from(str(next(folder.rglob("*.nemo"))), map_location="cpu").eval()
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        methods = ["ctc", "rnnt"]
    if args.methods:
        assert set(args.methods) <= set(methods)
        methods = args.methods
    out = TASK / ("asr-" + args.model)
    out.mkdir(exist_ok=True)
    assets = {str(p.relative_to(folder)): digest(p.read_bytes()) for p in folder.rglob("*")
              if p.is_file() and p.suffix in (".bin", ".nemo", ".json") and ".cache" not in p.parts}
    write_json(out / "identity.json", {"model": mid, "revision": entry["revision"], "assets": assets,
                                       "torch": torch.__version__, "source": str(SOURCE)})
    identity = json.loads((out / "identity.json").read_text())
    identity["packages"] = {}
    for package in ("transformers", "nemo_toolkit", "onnxruntime"):
        try:
            identity["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(out / "identity.json", identity)
    for case in inputs:
        pcm = np.load(case["input"], allow_pickle=False)
        assert digest(pcm.tobytes()) == case["pcm_sha"]
        for method in methods:
            destination = out / f"{case['id']}-{method}.json"
            if destination.exists():
                continue
            result = {"case": case, "method": method, "model": mid, "chunks": [],
                      "known_text_supplied": method == "teacher_attention"}
            start = time.perf_counter()
            try:
                if args.model == "whisper":
                    model.set_attn_implementation("sdpa" if method == "greedy_sdpa" else "eager")
                    result["attention_implementation"] = model.config._attn_implementation
                    result["generation_use_cache"] = model.generation_config.use_cache
                if args.model == "nemo":
                    from omegaconf import OmegaConf, open_dict
                    cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.aux_ctc.decoding
                                                                  if method == "ctc" else model.cfg.decoding))
                    with open_dict(cfg):
                        cfg.strategy = "greedy_batch"
                        cfg.compute_timestamps = True
                        cfg.preserve_alignments = True
                        if method == "rnnt":
                            cfg.rnnt_timestamp_type = "all"
                            cfg.greedy.use_cuda_graph_decoder = False
                    model.change_decoding_strategy(cfg, decoder_type=method)
                    with torch.inference_mode():
                        hyp = model.transcribe([str(Path(case["input"]).with_suffix(".wav"))], batch_size=1,
                                               return_hypotheses=True, verbose=False)[0]
                    stride = float(model.cfg.preprocessor.window_stride) * int(model.encoder.subsampling_factor) * 1000
                    result.update(text=hyp.text, words=[{"word": w.get("word", w.get("char")),
                                                        "start_ms": w["start_offset"] * stride,
                                                        "end_ms": w["end_offset"] * stride}
                                                       for w in hyp.timestamp.get("word", [])], stride_ms=stride)
                elif method == "teacher_attention":
                    # Canonical text is intentionally supplied only to this alignment method.
                    truth = next((c for c in json.loads((TASK / "secondary-truth.json").read_text())
                                  if c["id"] == case["id"]), None)
                    if truth is None:
                        result["not_applicable"] = "zero control has no canonical alignment target"
                    else:
                        rendered = truth["intro"] + truth["words"]
                        rows = acoustic.whisper_align(pcm, rendered, strict_words=False)
                        result.update(word_rows=rows, owners=[0] * len(truth["intro"]) + truth["owners"],
                                      canonical_renderings=rendered)
                else:
                    ranges = []
                    if method == "greedy_overlap":
                        for core in range(0,len(pcm),320000):
                            lo,hi = max(0,core-80000),min(len(pcm),core+400000)
                            ranges.append((lo,hi,(hi-lo)/16000,(core-lo)/16000,
                                           0 if hi == len(pcm) else 5))
                            if hi == len(pcm):
                                break
                        cores = [(lo+round(left*16000),hi-round(right*16000))
                                 for lo,hi,_,left,right in ranges]
                        assert cores[0][0] == 0 and cores[-1][1] == len(pcm)
                        assert all(a[1] == b[0] for a,b in zip(cores,cores[1:]))
                        assert all(0 < hi-lo <= 480000 for lo,hi,*_ in ranges)
                    else:
                        ranges = [(lo,min(len(pcm),lo+480000),0,0,0) for lo in range(0,len(pcm),480000)]
                    for lo,hi,length,left,right in ranges:
                        crop = pcm[lo:hi]
                        processed = acoustic.processor(crop, sampling_rate=16000, return_tensors="pt",
                                                       return_attention_mask=True)
                        with torch.inference_mode():
                            if method == "nbest5":
                                from transformers.generation.utils import GenerationMixin
                                generation = copy.deepcopy(model.generation_config)
                                generation.forced_decoder_ids = None
                                encoded = model.model.encoder(processed.input_features, return_dict=True)
                                generated = GenerationMixin.generate(
                                    model,encoder_outputs=encoded,attention_mask=processed.attention_mask,
                                    decoder_input_ids=torch.tensor([acoustic.processor.tokenizer.prefix_tokens]),
                                    generation_config=generation,max_new_tokens=256,do_sample=False,
                                    num_beams=5,num_return_sequences=5,return_dict_in_generate=True,output_scores=False)
                            else:
                                generated = model.generate(processed.input_features, attention_mask=processed.attention_mask,
                                                       language="ar", task="transcribe", return_timestamps=False,
                                                       return_token_timestamps=method == "token_timestamps",
                                                       max_new_tokens=256, do_sample=False,
                                                       num_beams=5 if method == "beam5" else 1,
                                                       num_return_sequences=1,
                                                       return_dict_in_generate=True, output_scores=False)
                        sequences = generated["sequences"] if isinstance(generated, dict) else generated.sequences
                        chunk = {"start_ms": lo / 16, "end_ms": (lo + len(crop)) / 16,
                                 "texts": acoustic.processor.batch_decode(sequences, skip_special_tokens=True),
                                 "tokens": sequences.tolist()}
                        if method == "nbest5":
                            chunk["distinct_token_sequences"] = len({tuple(s) for s in chunk["tokens"]})
                        if method == "greedy_overlap":
                            chunk["stride_s"] = [length,left,right]
                        if method == "token_timestamps":
                            timestamps = generated["token_timestamps"] if isinstance(generated, dict) else generated.token_timestamps
                            chunk["token_timestamps_s"] = timestamps.tolist()
                        result["chunks"].append(chunk)
                    result["text"] = " ".join(c["texts"][0] for c in result["chunks"])
                    if method == "greedy_overlap":
                        outputs = [{"tokens":np.asarray(c["tokens"]),"stride":tuple(c["stride_s"])}
                                   for c in result["chunks"]]
                        result["text"] = acoustic.processor.tokenizer._decode_asr(
                            outputs,return_timestamps=False,return_language=False,time_precision=.02)[0]
            except Exception:
                result["error"] = traceback.format_exc()
            result["wall_s"] = time.perf_counter() - start
            write_json(destination, result)
            print(args.model, case["id"], method, "ERROR" if "error" in result else "OK",
                  round(result["wall_s"], 2), flush=True)


if __name__ == "__main__":
    main()
