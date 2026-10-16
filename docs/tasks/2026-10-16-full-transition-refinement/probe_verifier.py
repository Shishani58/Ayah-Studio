"""Independent installed verifier, joint tail/head crops; no project writes."""
import json
from pathlib import Path

import numpy as np

from surah_cutter.core import align, decode
from surah_cutter.core._ctc import aligned_segments, forced_align
from surah_cutter.core.full_boundary import _muaalem_verifier
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text

TASK = Path(__file__).resolve().parent


def main():
    items = json.loads((TASK / "verification.json").read_text())
    bundle, source = _muaalem_verifier()
    results = []
    try:
        for item in items:
            reads = [r for r in item["trace"].get("full_boundary_reads", []) if r.get("role") == "two-sided-cut"]
            if not reads:
                continue
            audio = next(Path(p) for p in item["protected_sha256"] if p.endswith(".mp3"))
            pcm = decode.decode(audio)
            words = quran_text.surah_words(item["surah"])
            for read in reads:
                a = read["ayah"]
                selected = [w for w in words if w.ayah == a - 1][-2:] + [w for w in words if w.ayah == a][:2]
                lo, hi = read["start_ms"], read["end_ms"]
                # Phonetize complete canonical ayahs before slicing words:
                # artificial start/waqf on partial phrases changes the targets.
                complete = [w for w in words if w.ayah in (a - 1, a)]
                full_targets, full_ws = align._build_targets(complete, [], bundle, None)
                start_word = next(i for i, w in enumerate(complete) if w.index == selected[0].index)
                stop_word = next(i for i, w in enumerate(complete) if w.index == selected[-1].index)
                start_token, stop_token = full_ws[start_word][0], full_ws[stop_word][1]
                targets = full_targets[start_token:stop_token]
                ws = [(s - start_token, e - start_token) for s, e in full_ws[start_word:stop_word + 1]]
                if len([w for w in words if w.ayah == a]) <= 2:
                    following = next((c for c in item["before_checks"] if c["ayah"] == a + 1), None)
                    if following:
                        hi = min(len(pcm) / 16, max(hi, following["onset_ms"] + 400))
                assert hi - lo <= 12000
                crop = pcm[round(lo * 16):round(hi * 16)]
                matrix = bundle.emitter.emissions(crop)
                ids = matrix.argmax(axis=1)
                greedy = [int(t) for i, t in enumerate(ids) if t != bundle.blank_id and (i == 0 or t != ids[i - 1])]
                matches = [i for i in range(len(greedy) - len(targets) + 1) if greedy[i:i + len(targets)] == targets]
                path, scores = forced_align(matrix[None], np.asarray(targets)[None], blank=bundle.blank_id)
                spans = aligned_segments(path.reshape(-1), targets, blank=bundle.blank_id)
                stride = align._stride_ms(len(crop), 16000, len(matrix))
                quality = [align._span_confidence(scores.reshape(-1), s.start, s.end) for s in spans]
                split = next(i for i, w in enumerate(selected) if w.ayah == a)
                first = lo + spans[ws[split][0]].start * stride
                tail = lo + (spans[ws[split - 1][1] - 1].end + 1) * stride
                result = dict(surah=item["surah"], ayah=a, source=source, matches=matches, lo=lo, hi=hi,
                              first_ms=first, tail_ms=tail, min_quality=min(quality),
                              targets=targets, greedy=greedy, quality=quality, lag_ms=bundle.lag_ms)
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k not in {"quality", "targets", "greedy"}}), flush=True)
                (TASK / "verifier-probe.json").write_text(json.dumps(results, ensure_ascii=False) + "\n")
    finally:
        release_runtimes()


if __name__ == "__main__":
    main()
