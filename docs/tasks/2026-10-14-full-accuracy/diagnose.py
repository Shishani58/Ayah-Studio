"""Read-only FULL token/signal audit against frozen runtime; GT is evaluation only."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
PREVIOUS = TASK.parent / "2026-10-14-full-other-reciters"
RUNTIME = Path(json.loads((PREVIOUS / "runtime-manifest.json").read_text())["runtime"])
sys.path.insert(0, str(RUNTIME))
from surah_cutter.core import align, candidates, decode, emissions, full_windows  # noqa: E402
from surah_cutter.data import quran_text, references  # noqa: E402

CACHE = Path.home() / ".cache/surah-cutter/full-accuracy"
CACHE.mkdir(exist_ok=True)
emitter = emissions.load_emitter(ROOT / "models/wav2vec2-quran-phonetics.onnx")
inverse = {v: k for k, v in emitter.vocab.items()}
original = align._check_ayah_onsets
details = []


def capture(words, word_token_spans, target_spans, frame_scores, stride_ms, **kwargs):
    checks = original(words, word_token_spans, target_spans, frame_scores, stride_ms, **kwargs)
    from surah_cutter.data import transliteration
    all_trs = transliteration.surah_word_tr(words[0].surah)
    targets, _ = align._build_targets(words, [all_trs[w.index] for w in words],
                                     bundle, align._basmala_trs(transliteration)
                                     if words[0].index == 0 else None)
    base_ms = kwargs.get("base_ms", 0)
    seen = set()
    for wi, (word, (ts, te)) in enumerate(zip(words, word_token_spans)):
        if word.ayah in seen:
            continue
        seen.add(word.ayah)
        context_end = min(len(words), wi + 3)
        token_end = word_token_spans[context_end - 1][1]
        tokens = []
        for ti in range(ts, token_end):
            sf, ef = target_spans[ti]
            tokens.append({"token": inverse[targets[ti]], "start_ms": base_ms + sf * stride_ms,
                           "end_ms": base_ms + (ef + 1) * stride_ms,
                           "conf": align._span_confidence(frame_scores, sf, ef)})
        details.append({"ayah": word.ayah, "first_word": word.text,
                        "raw_onset_ms": checks[word.ayah]["raw_onset_ms"],
                        "first_word_tokens": te - ts, "tokens": tokens})
    return checks


align._check_ayah_onsets = capture
recordings = json.loads((PREVIOUS / "data-manifest.json").read_text())["recordings"]
recordings.append({"name": "abu_bakr_ash_shatri", "audio_dir": str(Path.home() / "Desktop/Abu_Bakr_Ash-Shatri"),
                   "reference": str(ROOT / "data/references/abu_bakr_ash_shatri.db")})
results = []
for rec in recordings:
    surahs = [42, 83, 95, 112] if rec["name"] == "abu_bakr_ash_shatri" else [80, 98, 112]
    for surah in surahs:
        details.clear()
        path = Path(rec["audio_dir"]) / f"{surah:03d}.mp3"
        pcm = decode.decode(str(path))
        stem = f"{rec['name']}-{surah}"
        cache_file = CACHE / (stem + ".npy")
        trace_file = CACHE / (stem + "-windows.json")
        if cache_file.exists():
            matrix = np.load(cache_file)
            windows = json.loads(trace_file.read_text())
        else:
            windows = {}
            matrix = full_windows.generate_pause_emissions(emitter, pcm, window_s=20, workers=3,
                         strong_pauses=True, max_window_s=60, trace=windows)
            np.save(cache_file, matrix)
            trace_file.write_text(json.dumps(windows))
        bundle = align._OnnxBundle(SimpleNamespace(vocab=emitter.vocab, blank_id=emitter.blank_id,
                                                  emissions=lambda _: matrix))
        trace = {}
        rows = align.align_surah(pcm, 16000, quran_text.surah_words(surah), model=bundle,
                                 surah=surah, trace=trace)
        align._review_pause_full_onsets(trace["ayahs"], windows)
        # All predictions completed before opening reference.
        reference = references.load_reference(rec["reference"])
        rms = np.sqrt(np.mean(pcm[:len(pcm)//320*320].reshape(-1, 320)**2, axis=1))
        np.save(CACHE / (stem + "-rms.npy"), rms)
        result = {"reciter": rec["name"], "surah": surah, "rows": rows,
                  "errors_ms": [ms-reference[surah, a] for a, ms, _ in rows],
                  "checks": trace["ayahs"], "details": [next(d for d in reversed(details)
                      if d["ayah"] == c["ayah"] and d["raw_onset_ms"] == c["raw_onset_ms"])
                      for c in trace["ayahs"]], "windows": windows,
                  "audio": str(path), "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "emissions_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                  "stride_ms": align._stride_ms(len(pcm), 16000, len(matrix))}
        result["pauses"] = [{"start_ms": p.start_ms, "end_ms": p.end_ms, "center_ms": p.center_ms}
                            for p in candidates.detect_energy_pauses(pcm, 16000, use_ten_vad=False)]
        if rec["name"] != "abu_bakr_ash_shatri":
            previous = json.loads((PREVIOUS / (rec["name"] + ".json")).read_text())
            expected = next(s for c in previous["cases"] if c["case"] == "strong:20:3:0"
                            for s in c["surahs"] if s["surah"] == surah)
            assert rows == [tuple(r) for r in expected["rows"]]
            assert result["emissions_sha256"] == expected["emissions_sha256"]
        results.append(result)
        (TASK / "diagnostics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
        print(stem, "reproduced", flush=True)
