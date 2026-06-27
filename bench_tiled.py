"""Quick validation: tiled vs full vs fast vs GT on the Shatri reference.

Measures per-sura accuracy (median/p90/max error vs ground truth) and wall time
for each mode, so we can see whether `tiled` matches `full` accuracy at ~1x cost.
Run: python bench_tiled.py 078 080 083 087 089 098
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from surah_cutter.core import align, align_fast, align_tiled, decode
from surah_cutter.data import quran_text, references

AUDIO = Path(os.environ.get("SHATRI_DIR", str(Path.home() / "Desktop" / "Abu_Bakr_Ash-Shatri")))
REF = references.load_reference("abu_bakr_ash_shatri")


def err_stats(rows, surah):
    es = []
    for ayah, ms, *_ in rows:
        gt = REF.get((surah, ayah))
        if gt is not None:
            es.append(abs(ms - gt))
    es.sort()
    if not es:
        return None
    n = len(es)
    return dict(
        med=es[n // 2], p90=es[min(n - 1, int(0.9 * n))], mx=es[-1],
        le150=round(100 * sum(e <= 150 for e in es) / n),
        le500=round(100 * sum(e <= 500 for e in es) / n), n=n,
    )


def run(argv):
    suras = argv or ["078", "080", "083", "087", "089", "098"]
    print(f"audio={AUDIO}")
    print(f"{'sura':>4} {'mode':>6} {'sec':>6} {'med':>5} {'p90':>5} {'max':>6} {'<=150':>5} {'<=500':>5}")
    for s in suras:
        n = int(s)
        path = AUDIO / f"{s}.mp3"
        if not path.exists():
            print(f"{s} MISSING {path}")
            continue
        pcm = decode.decode(str(path))
        sr = decode.DEFAULT_SR
        words = quran_text.surah_words(n)
        dur = len(pcm) / sr
        print(f"--- sura {s}  ({dur:.0f}s audio, {len(words)} words) ---")
        for mode in ("full", "tiled", "fast"):
            t0 = time.time()
            if mode == "full":
                rows = align.align_surah(pcm, sr, words, surah=n)
            elif mode == "tiled":
                rows = align_tiled.align_surah_tiled(pcm, sr, words, surah=n)
            else:
                rows = align_fast.align_surah_fast(pcm, sr, words, surah=n)
            dt = time.time() - t0
            st = err_stats(rows, n)
            if st:
                print(f"{s:>4} {mode:>6} {dt:6.1f} {st['med']:5d} {st['p90']:5d} "
                      f"{st['mx']:6d} {st['le150']:4d}% {st['le500']:4d}%")
            else:
                print(f"{s:>4} {mode:>6} {dt:6.1f}  (no GT)")
        sys.stdout.flush()


if __name__ == "__main__":
    run(sys.argv[1:])
