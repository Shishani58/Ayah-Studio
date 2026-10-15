"""Independent ASR audit; references never enter the recognition call.

prepare: app .venv; fastconformer: isolated NeMo Python; whisper: model Python.
Inputs live in /tmp; outputs are audit JSON, never application databases.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
INPUTS = Path("/tmp/surah-cutter-asr-probe-20261014")
sys.path.insert(0, str(ROOT))


def prepare():
    import soundfile as sf

    from surah_cutter.core import decode
    from surah_cutter.core.model_runtime import inference_words
    from surah_cutter.data import quran_text, references

    INPUTS.mkdir(exist_ok=True)
    sources = [("abu_bakr_ash_shatri", 1, Path.home() / "Desktop/Abu_Bakr_Ash-Shatri/001.mp3")]
    diagnostics = json.loads((TASK.parent / "2026-10-14-full-accuracy/diagnostics.json").read_text())
    sources.extend((d["reciter"], d["surah"], Path(d["audio"]))
                   for d in diagnostics if d["surah"] == 112)
    cases = []
    for reciter, surah, path in sources:
        pcm = decode.decode(str(path))
        words = quran_text.surah_words(surah)
        gt = references.load_reference(reciter)
        base = {"reciter": reciter, "surah": surah, "audio": str(path),
                "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "total_ms": len(pcm) / 16,
                "reference_text": " ".join(inference_words(words, "fastconformer")),
                "reference_ayahs": [dict(ayah=a, onset_ms=gt[surah, a],
                                         text=" ".join(inference_words(
                                             [w for w in words if w.ayah == a], "fastconformer")))
                                    for a in range(1, quran_text.ayah_count(surah) + 1)]}
        name = f"{reciter}-{surah}-full"
        sf.write(INPUTS / f"{name}.wav", pcm, 16000, subtype="FLOAT")
        np.save(INPUTS / f"{name}.npy", pcm)
        cases.append(dict(base, id=name, kind="FULL-ASR", start_ms=0, end_ms=len(pcm) / 16))
        # Fixed short local probes; GT is used later for evaluation only.
        start, end = (6300, 10300) if surah == 1 else (0, min(6000, len(pcm) / 16 - 1))
        crop = pcm[int(start * 16):int(end * 16)]
        name = f"{reciter}-{surah}-local"
        sf.write(INPUTS / f"{name}.wav", crop, 16000, subtype="FLOAT")
        np.save(INPUTS / f"{name}.npy", crop)
        cases.append(dict(base, id=name, kind="LOCAL-ASR", start_ms=start, end_ms=end))
        if surah == 1:
            name = "shatri-1-local-shift600"
            sf.write(INPUTS / f"{name}.wav", np.concatenate((np.zeros(9600), crop)),
                     16000, subtype="FLOAT")
            np.save(INPUTS / f"{name}.npy", np.concatenate((np.zeros(9600), crop)))
            cases.append(dict(base, id=name, kind="LOCAL-ASR-shift", start_ms=start,
                              end_ms=end, padding_ms=600))
    name = "silence-4s"
    sf.write(INPUTS / f"{name}.wav", np.zeros(64000), 16000, subtype="FLOAT")
    np.save(INPUTS / f"{name}.npy", np.zeros(64000))
    cases.append(dict(id=name, kind="SILENCE", start_ms=0, end_ms=4000))
    (INPUTS / "manifest.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(f"Prepared {len(cases)} cases; full and local experiments are separate.")


def recognize(model_id):
    import importlib.metadata

    import torch

    torch.set_num_threads(4)
    checkpoint_id = model_id.removesuffix("-ctc")
    catalog = json.loads((ROOT / "surah_cutter/core/model_catalog.json").read_text())
    entry = next(e for e in catalog if e["id"] == checkpoint_id)
    folder = Path.home() / ".surah-cutter/models" / checkpoint_id / entry["revision"]
    stage = time.perf_counter()
    if checkpoint_id == "fastconformer":
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        from omegaconf import OmegaConf, open_dict

        model = EncDecHybridRNNTCTCBPEModel.restore_from(
            str(next(folder.rglob("*.nemo"))), map_location="cpu").eval()
        decoder = "ctc" if model_id.endswith("-ctc") else "rnnt"
        source_cfg = model.cfg.aux_ctc.decoding if decoder == "ctc" else model.cfg.decoding
        cfg = OmegaConf.create(OmegaConf.to_container(source_cfg))
        with open_dict(cfg):
            cfg.strategy = "greedy_batch"
            cfg.compute_timestamps = True
            cfg.preserve_alignments = True
            if decoder == "rnnt":
                cfg.rnnt_timestamp_type = "all"
                cfg.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(cfg, decoder_type=decoder)
        model.preprocessor.featurizer.dither = 0.0
        stride_s = float(model.cfg.preprocessor.window_stride) * int(model.encoder.subsampling_factor)
    else:
        from surah_cutter.core.model_worker import Acoustic

        acoustic = Acoustic(model_id, folder)
        model = acoustic.model
    versions = {}
    for package in ("torch", "nemo_toolkit", "transformers", "onnx", "ml_dtypes", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = dict(model=model_id, revision=entry["revision"], versions=versions,
                  load_s=time.perf_counter() - stage, decoder=decoder if checkpoint_id == "fastconformer"
                  else "autoregressive-greedy", known_text_supplied=False, cases=[])
    for case in json.loads((INPUTS / "manifest.json").read_text()):
        stage = time.perf_counter()
        path = INPUTS / f"{case['id']}.wav"
        record = dict(case)
        try:
            with torch.inference_mode():
                if checkpoint_id == "fastconformer":
                    hypothesis = model.transcribe([str(path)], batch_size=1,
                                                  return_hypotheses=True, verbose=False)[0]
                    record["text"] = hypothesis.text
                    record["words"] = [dict(word=w.get("word", w.get("char")),
                                            start_ms=w["start_offset"] * stride_s * 1000,
                                            end_ms=w["end_offset"] * stride_s * 1000)
                                       for w in hypothesis.timestamp.get("word", [])]
                    record["stride_ms"] = stride_s * 1000
                else:
                    pcm, sr = np.load(path.with_suffix(".npy")).astype(np.float32), 16000
                    # Long FULL experiments use independent 30s blocks, never FAST.
                    texts = []
                    for begin in range(0, len(pcm), sr * 30):
                        features = acoustic.processor(pcm[begin:begin + sr * 30],
                                                      sampling_rate=sr, return_tensors="pt",
                                                      return_attention_mask=True)
                        tokens = model.generate(features.input_features, attention_mask=features.attention_mask,
                                                language="ar", task="transcribe",
                                                max_new_tokens=256, do_sample=False, num_beams=1,
                                                return_timestamps=False)
                        texts.append(acoustic.processor.batch_decode(tokens, skip_special_tokens=True)[0])
                    record["text"] = " ".join(texts)
            record["elapsed_s"] = time.perf_counter() - stage
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        result["cases"].append(record)
        (TASK / f"{model_id}-asr.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(case["id"], record.get("text", record.get("error")), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        recognize(sys.argv[1])
