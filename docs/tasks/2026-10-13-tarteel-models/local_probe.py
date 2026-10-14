"""Sparse pause-adjacent ASR experiment; not a production FAST adapter."""

import argparse
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def plain(text):
    text = "".join(c for c in text if unicodedata.category(c) not in ("Mn", "Me"))
    return text.replace("ٱ", "ا").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")


def identify(text, phrases):
    observed = plain(text).split()
    scores = sorted(
        (
            (difflib.SequenceMatcher(None, phrase, observed, autojunk=False).ratio(), ayah)
            for ayah, phrase in phrases
        ),
        reverse=True,
    )
    best, ayah = scores[0]
    gap = best - scores[1][0] if len(scores) > 1 else best
    return dict(
        ayah=ayah,
        similarity=best,
        margin=gap,
        confirmed=best >= 0.8 and gap >= 0.2 and len(observed) >= 2,
    )


def worker(folder, model_id, work):
    sys.path.insert(0, str(ROOT / "surah_cutter/core"))
    from model_worker import Acoustic

    started = time.monotonic()
    acoustic = Acoustic(model_id, Path(folder))
    load = time.monotonic() - started
    data = json.loads((work / "input.json").read_text())
    report = []
    for item in data:
        pcm = np.load(work / f"{item['surah']}.npy", allow_pickle=False)
        reads = []
        for pause in item["pauses"]:
            for speech_ms in (1300, 3000):
                start = max(pause["start_ms"], pause["end_ms"] - 200)
                end = min(len(pcm) / 16, pause["end_ms"] + speech_ms)
                clip = pcm[int(start * 16) : int(end * 16)]
                started = time.monotonic()
                processed = acoustic.processor(
                    clip, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
                )
                features = processed.input_features
                with acoustic.torch.inference_mode():
                    output = acoustic.model.generate(
                        features,
                        attention_mask=processed.attention_mask,
                        language="ar",
                        task="transcribe",
                        return_timestamps=False,
                        max_new_tokens=64,
                        do_sample=False,
                        num_beams=1,
                    )
                text = acoustic.processor.batch_decode(output, skip_special_tokens=True)[0]
                identity = identify(text, item["phrases"])
                reads.append(
                    dict(
                        start_ms=start,
                        end_ms=end,
                        input_ms=end - start,
                        pause_center_ms=(pause["start_ms"] + pause["end_ms"]) / 2,
                        seconds=time.monotonic() - started,
                        text=text,
                        **identity,
                        reason="minimal_phrase" if speech_ms == 1300 else "unconfirmed_phrase",
                    )
                )
                if identity["confirmed"]:
                    break
        intervals = sorted((r["start_ms"], r["end_ms"]) for r in reads)
        union = []
        for a, b in intervals:
            if union and a <= union[-1][1]:
                union[-1][1] = max(b, union[-1][1])
            else:
                union.append([a, b])
        covered = sum(b - a for a, b in union)
        assert covered < len(pcm) / 16 * 0.65
        report.append(
            {
                **item,
                "model": model_id,
                "load_seconds": load,
                "model_reads": reads,
                "coverage": covered / (len(pcm) / 16),
                "covered_audio_ms": covered,
                "audio_ms": sum(r["input_ms"] for r in reads),
                "inference_seconds": sum(r["seconds"] for r in reads),
                "runtime_versions": {
                    p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")
                },
                "generation_config_source": next(
                    m
                    for m in json.loads((ROOT / "surah_cutter/core/model_catalog.json").read_text())
                    if m["id"] == model_id
                )["generation_config_source"],
            }
        )
    (work / "output.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=3)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], args.worker[1], Path(args.worker[2]))
        return
    sys.path.insert(0, str(ROOT))
    from surah_cutter.core import candidates, decode, model_store
    from surah_cutter.data import quran_text
    from surah_cutter.core.model_runtime import inference_words, ensure_runtime

    data = []
    with tempfile.TemporaryDirectory(prefix="tarteel-local-") as directory:
        work = Path(directory)
        for surah in (114, 112, 78):
            audio = Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri") / f"{surah:03d}.mp3"
            started = time.monotonic()
            pcm = decode.decode(str(audio))
            decode_s = time.monotonic() - started
            started = time.monotonic()
            pauses = candidates.detect_energy_pauses(pcm, 16000)
            dsp_s = time.monotonic() - started
            pauses = [p for p in pauses if p.start_ms > 500 and p.end_ms < len(pcm) / 16 - 3000]
            selected = [pauses[i] for i in sorted(set([0, len(pauses) // 2, len(pauses) - 1]))]
            words = quran_text.surah_words(surah)
            renderings = inference_words(words, "whisper-tarteel-tiny")
            phrases = [
                (
                    a,
                    plain(" ".join(t for w, t in zip(words, renderings) if w.ayah == a)).split()[
                        :2
                    ],
                )
                for a in range(1, quran_text.ayah_count(surah) + 1)
            ]
            np.save(work / f"{surah}.npy", pcm)
            data.append(
                dict(
                    surah=surah,
                    audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                    decode_seconds=decode_s,
                    dsp_seconds=dsp_s,
                    phrases=phrases,
                    duration_ms=len(pcm) / 16,
                    pauses=[vars(p) for p in selected],
                )
            )
        (work / "input.json").write_text(json.dumps(data, ensure_ascii=False))
        for size in ("tiny", "base"):
            model_id = f"whisper-tarteel-{size}"
            subprocess.run(
                [
                    str(ensure_runtime(model_id)),
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(model_store.installed_path(model_store.spec(model_id))),
                    model_id,
                    str(work),
                ],
                check=True,
            )
            report = json.loads((work / "output.json").read_text())
            # Reference is consulted only after all local inference for this model.
            ref = ROOT / "data/references/abu_bakr_ash_shatri.db"
            with sqlite3.connect(f"file:{ref}?mode=ro", uri=True) as conn:
                for item in report:
                    truth = list(
                        conn.execute("SELECT ayah,time FROM timings WHERE sura=?", (item["surah"],))
                    )
                    item["reference_sha256"] = hashlib.sha256(ref.read_bytes()).hexdigest()
                    item["revision"] = model_store.spec(model_id)["revision"]
                    for read in item["model_reads"]:
                        a, ms = min(truth, key=lambda row: abs(row[1] - read["pause_center_ms"]))
                        read["reference_ayah"] = a
                        read["reference_distance_ms"] = abs(ms - read["pause_center_ms"])
                        read["correct_ayah"] = a == read["ayah"]
            output = Path(__file__).with_name(f"{size}-local.json")
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(
                size,
                [(x["surah"], x["coverage"], x["inference_seconds"]) for x in report],
                flush=True,
            )


if __name__ == "__main__":
    main()
