"""Diagnostic target variants only; canonical text and runtime stay untouched."""

import json, sys
from pathlib import Path
import numpy as np

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK.parents[2]))
from surah_cutter.core import align, emissions
from surah_cutter.core._ctc import forced_align, aligned_segments
from surah_cutter.data import quran_text, transliteration

probe = json.loads((TASK / "probe.json").read_text())
matrix = np.load(TASK / "chunk-emissions.npy")
emitter = emissions.load_emitter(TASK.parents[2] / "models/wav2vec2-quran-phonetics.onnx")
bundle = align._OnnxBundle(emitter)
trs = transliteration.surah_word_tr(2)
words = [w for w in quran_text.surah_words(2) if 215 <= w.ayah <= 223]
wi = next(i for i, w in enumerate(words) if w.ayah == 220)
results = []
for name, selected in [
    ("baseline", words),
    ("repeat_previous_last_two_words", words[:wi] + words[wi - 2 : wi] + words[wi:]),
]:
    targets, ws = align._build_targets(selected, [trs[w.index] for w in selected], bundle, None)
    p, sc = forced_align(matrix[None], np.asarray(targets)[None], blank=bundle.blank_id)
    segs = aligned_segments(p.reshape(-1), targets, blank=bundle.blank_id)
    spans = [(s.start, s.end) for s in segs]
    scores = sc.reshape(-1)
    inverse = {v: k for k, v in bundle.vocab.items()}
    first = []
    for i, w in enumerate(selected):
        if i and selected[i - 1].ayah == w.ayah:
            continue
        ts, te = ws[i]
        v = [float(np.exp(scores[s : e + 1]).mean()) for s, e in spans[ts:te]]
        l = np.concatenate([scores[s : e + 1] for s, e in spans[ts:te]])
        first.append(
            {
                "ayah": w.ayah,
                "text": w.text,
                "tr": trs[w.index],
                "start_ms": round(probe["base_ms"] + spans[ts][0] * probe["stride_ms"] - 200),
                "geometric": float(np.exp(l.mean())),
                "arithmetic": float(np.exp(l).mean()),
                "tokens": [
                    {
                        "char": inverse[targets[j]],
                        "confidence": align._span_confidence(scores, *spans[j]),
                    }
                    for j in range(ts, te)
                ],
            }
        )
    results.append({"variant": name, "first_words": first})
    print(name, first, flush=True)
(TASK / "counterfactual.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
