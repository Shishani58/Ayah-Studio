"""Read-only replay of one logged FULL chunk, preserving acoustic windows."""

import json
from pathlib import Path
import sys
import hashlib
import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT))
from surah_cutter.core import align, decode, emissions
from surah_cutter.core._ctc import forced_align, aligned_segments
from surah_cutter.data import quran_text, transliteration

AUDIO = Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/002.mp3")
run = json.loads(
    AUDIO.with_name("Abu_Bakr_Ash-Shatri.logs").joinpath("sura_002_full.json").read_text()
)
trace = run["trace"]
pcm = decode.decode(str(AUDIO))
emitter = emissions.load_emitter(ROOT / "models/wav2vec2-quran-phonetics.onnx")
bundle = align._OnnxBundle(emitter)
wav = emissions._normalize(pcm)
chunk = next(c for c in trace["chunks"] if c["ayahs"] == [217, 220])
stride = align._stride_ms(len(pcm), 16000, trace["emissions_frames"])
s0, s1 = [round(t * 16) for t in chunk["window_ms"]]
f0, f1 = [round(s * trace["emissions_frames"] / len(pcm)) for s in [s0, s1]]
parts = []
for w in trace["full_windows"]["windows"]:
    left, right = [round(w[k] * 16) for k in ["start_ms", "end_ms"]]
    lf, rf = left // 320, left // 320 + w["frames"]
    if rf <= f0 or lf >= f1:
        continue
    clip = wav[left - 32000 : right + 32000]
    logits = emissions._run(emitter.session, clip[None], emitter._input_name, emitter._output_name)[
        0
    ]
    matrix = emissions._log_softmax(logits[100 : 100 + w["frames"]])
    parts.append(matrix[max(0, f0 - lf) : min(w["frames"], f1 - lf)])
    print("window", w["start_ms"], w["end_ms"], flush=True)
matrix = np.ascontiguousarray(np.concatenate(parts))
np.save(TASK / "chunk-emissions.npy", matrix)
words = [w for w in quran_text.surah_words(2) if 215 <= w.ayah <= 223]
trs = transliteration.surah_word_tr(2)
targets, word_spans = align._build_targets(words, [trs[w.index] for w in words], bundle, None)
path, scores = forced_align(matrix[None], np.asarray(targets)[None], blank=bundle.blank_id)
segs = aligned_segments(path.reshape(-1), targets, blank=bundle.blank_id)
spans = [(s.start, s.end) for s in segs]
scores = scores.reshape(-1)
inverse = {v: k for k, v in bundle.vocab.items()}
base = f0 * stride
checks = align._check_ayah_onsets(
    words, word_spans, spans, scores, stride, pauses=[], base_ms=base, lag_ms=200
)
result = {
    "source": run["source"],
    "chunk": chunk,
    "stride_ms": stride,
    "base_ms": base,
    "checks": checks,
    "words": [],
    "greedy": [],
    "local": [],
    "audio_sha256": hashlib.sha256(AUDIO.read_bytes()).hexdigest(),
}
for w, (ts, te) in zip(words, word_spans):
    if w.ayah not in [219, 220]:
        continue
    tokens = [
        {
            "char": inverse[targets[i]],
            "start_ms": base + spans[i][0] * stride,
            "end_ms": base + spans[i][1] * stride,
            "confidence": align._span_confidence(scores, *spans[i]),
        }
        for i in range(ts, te)
    ]
    result["words"].append({"ayah": w.ayah, "text": w.text, "tr": trs[w.index], "tokens": tokens})
ids = matrix.argmax(axis=1)
for i, token in enumerate(ids):
    t = base + i * stride
    if 4840000 <= t <= 4854000 and token != bundle.blank_id and (i == 0 or token != ids[i - 1]):
        result["greedy"].append(
            {"char": inverse[int(token)], "ms": t, "prob": float(np.exp(matrix[i, token]))}
        )
for lo, hi in [(4843000, 4850000), (4845600, 4850000)]:
    crop = pcm[lo * 16 : hi * 16]
    m = emitter.emissions(crop)
    ids = m.argmax(axis=1)
    greedy = "".join(
        inverse[int(t)]
        for i, t in enumerate(ids)
        if t != bundle.blank_id and (i == 0 or t != ids[i - 1])
    )
    selected = [w for w in words if w.ayah == 220][:3]
    tar, ws = align._build_targets(selected, [trs[w.index] for w in selected], bundle, None)
    p, sc = forced_align(m[None], np.asarray(tar)[None], blank=bundle.blank_id)
    sg = aligned_segments(p.reshape(-1), tar, blank=bundle.blank_id)
    ck = align._check_ayah_onsets(
        selected,
        ws,
        [(s.start, s.end) for s in sg],
        sc.reshape(-1),
        align._stride_ms(len(crop), 16000, len(m)),
        pauses=[],
        base_ms=lo,
        lag_ms=200,
    )
    result["local"].append({"crop_ms": [lo, hi], "greedy": greedy, "check": ck})
(TASK / "probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("220", checks[220], "local", result["local"], flush=True)
