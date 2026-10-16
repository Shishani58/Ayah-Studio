"""Read-only baseline and local two-sided QPC probes; no project DB writes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from surah_cutter.core import align, decode, emissions, full_pause_center
from surah_cutter.core._ctc import forced_align, aligned_segments
from surah_cutter.core.full_boundary import _consonants
from surah_cutter.data import quran_text, transliteration
import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = Path.home() / "Desktop/Abu_Bakr_Ash-Shatri"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suras", nargs="+", type=int, default=[1, 112, 114, 80, 98])
    args = parser.parse_args()
    bundle = align._OnnxBundle(emissions.load_emitter(Path("models/wav2vec2-quran-phonetics.onnx")))
    inverse = {v: k for k, v in bundle.vocab.items()}
    original = full_pause_center.record_edges

    def edges(checks, words, ws, spans, scores, stride, **kw):
        original(checks, words, ws, spans, scores, stride, **kw)
        by_ayah = defaultdict(list)
        for w, (ts, te) in zip(words, ws):
            by_ayah[w.ayah].append(dict(index=w.index, start_ms=kw.get("base_ms", 0) + spans[ts][0] * stride,
                                        end_ms=kw.get("base_ms", 0) + (spans[te - 1][1] + 1) * stride))
        for a, items in by_ayah.items():
            checks[a]["probe_head"] = items[:2]
            checks[a]["probe_tail"] = items[-2:]

    full_pause_center.record_edges = edges
    for surah in args.suras:
        pcm = decode.decode(ROOT / f"{surah:03d}.mp3")
        words = quran_text.surah_words(surah)
        trs = transliteration.surah_word_tr(surah)
        trace = {}
        start = time.monotonic()
        rows = align.align_surah(pcm, 16000, words, surah=surah, model=bundle, trace=trace)
        baseline = dict(surah=surah, rows=rows, trace=trace, elapsed_s=time.monotonic() - start)
        (TASK / f"exploration-{surah}.json").write_text(json.dumps(baseline, ensure_ascii=False) + "\n")
        probes = []
        for prev, check in zip(trace["ayahs"], trace["ayahs"][1:]):
            if check.get("repaired") or check["pause_center_reason"] == "positioned":
                continue
            left, right = prev["probe_tail"], check["probe_head"]
            lo = max(0, round((left[0]["start_ms"] - 400) / 20) * 20)
            hi = min(len(pcm) / 16, round((right[-1]["end_ms"] + 400) / 20) * 20)
            if not 1000 <= hi - lo <= 12000:
                continue
            matrix = bundle.emitter.emissions(pcm[round(lo * 16):round(hi * 16)])
            ids = matrix.argmax(axis=1)
            decoded = "".join(inverse[int(t)] for i, t in enumerate(ids)
                              if t != bundle.blank_id and (i == 0 or t != ids[i - 1]))
            targets = [[trs[w["index"]] for w in group] for group in (left, right)]
            selected = [words[w["index"]] for group in (left, right) for w in group]
            tokens, ws = align._build_targets(selected, sum(targets, []), bundle, None)
            path, scores = forced_align(matrix[None], np.asarray(tokens)[None], blank=bundle.blank_id)
            segments = aligned_segments(path.reshape(-1), tokens, blank=bundle.blank_id)
            spans = [(s.start, s.end) for s in segments]
            stride = align._stride_ms(round((hi - lo) * 16), 16000, len(matrix))
            local = align._check_ayah_onsets(selected, ws, spans, scores.reshape(-1), stride,
                                            pauses=[], base_ms=lo, lag_ms=bundle.lag_ms)
            original(local, selected, ws, spans, scores.reshape(-1), stride, base_ms=lo)
            probes.append(dict(ayah=check["ayah"], reason=check["pause_center_reason"],
                               lo=lo, hi=hi, decoded=decoded, targets=targets,
                               consonants=_consonants(decoded),
                               expected=[_consonants("".join(t)) for t in targets], local=local))
        (TASK / f"probe-{surah}.json").write_text(json.dumps(probes, ensure_ascii=False, indent=2) + "\n")
        print(surah, len(rows), round(baseline["elapsed_s"], 2), json.dumps(probes[:6], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
