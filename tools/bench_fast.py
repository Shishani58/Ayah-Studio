"""Bench the two-phase fast aligner vs full vs ground truth (Shatri juz).

Run: python -m tools.bench_fast 093 100 080 087 090 078
Presets:  (no args) → the curated hard set; ``long`` → all suras >60 s that
actually exercise the windowed path; ``all`` → 78-114.
Add ``fast-only`` to skip the full reference (it never changes — only the fast
path is under iteration), roughly halving wall time. Audio dir via $SHATRI_DIR.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from surah_cutter.core import align, align_fast, decode
from surah_cutter.data import quran_text, references

AUDIO = Path(os.environ.get("SHATRI_DIR", str(Path.home() / "Desktop" / "Abu_Bakr_Ash-Shatri")))
REF = references.load_reference("abu_bakr_ash_shatri")

# Suras <60 s fall back to the full path (align_fast._FAST_MIN_MS) → no fast
# signal. These are the ones that DO exercise the windowed path; the default is a
# curated subset spanning the failure modes (78/83 melodic off-by-one, 80 clean
# high-coverage, 87 the canonical broken sura, 89 wasl-heavy, 98 ~18 s ayahs).
LONG = [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 96, 98]
HARD = ["078", "080", "083", "087", "089", "098"]


def err_stats(rows, surah):
    es = []
    for ayah, ms, _c in rows:
        gt = REF.get((surah, ayah))
        if gt is not None:
            es.append(abs(ms - gt))
    es.sort()
    if not es:
        return None
    n = len(es)
    return {
        "med": es[n // 2],
        "p90": es[min(n - 1, int(0.9 * n))],
        "max": es[-1],
        "le150": round(100 * sum(e <= 150 for e in es) / n),
        "le500": round(100 * sum(e <= 500 for e in es) / n),
        "n": n,
    }


def main(argv):
    fast_only = "fast-only" in argv
    argv = [a for a in argv if a != "fast-only"]
    if argv == ["all"]:
        suras = [f"{n:03d}" for n in range(78, 115)]
    elif argv == ["long"]:
        suras = [f"{n:03d}" for n in LONG]
    else:
        suras = argv or HARD
    bundle = align._get_onnx_bundle()  # warm + reuse
    all_fast_err, all_full_err = [], []
    tot_tf = tot_tu = 0.0
    for s in suras:
        surah = int(s)
        path = AUDIO / f"{int(s):03d}.mp3"
        if not path.exists():
            print(f"sura {surah}: NO AUDIO {path}")
            continue
        pcm = decode.decode(str(path))
        sr = decode.DEFAULT_SR
        words = quran_text.surah_words(surah)

        trace: dict = {}
        t0 = time.perf_counter()
        fast = align_fast.align_surah_fast(pcm, sr, words, surah=surah, bundle=bundle, trace=trace)
        tf = time.perf_counter() - t0

        for ayah, ms, _c in fast:
            gt = REF.get((surah, ayah))
            if gt is not None:
                all_fast_err.append(abs(ms - gt))
        tot_tf += tf
        ef = err_stats(fast, surah)
        print(
            f"sura {surah:>3} ({trace.get('total_ms',0)/1000:5.1f}s, {len(words)}w, "
            f"pause={trace.get('n_pause')}/wasl={trace.get('n_wasl')}, "
            f"cover={trace.get('coverage')}):"
        )
        print(
            f"   FAST med={ef['med']:>5} p90={ef['p90']:>5} max={ef['max']:>6} "
            f"<=150={ef['le150']}% <=500={ef['le500']}%  {tf:5.1f}s"
        )
        if not fast_only:
            t0 = time.perf_counter()
            full = align.align_surah(pcm, sr, words, surah=surah, model=bundle)
            tu = time.perf_counter() - t0
            for ayah, ms, _c in full:
                gt = REF.get((surah, ayah))
                if gt is not None:
                    all_full_err.append(abs(ms - gt))
            tot_tu += tu
            eu = err_stats(full, surah)
            speed = tu / tf if tf else 0.0
            print(
                f"   FULL med={eu['med']:>5} p90={eu['p90']:>5} max={eu['max']:>6} "
                f"<=150={eu['le150']}% <=500={eu['le500']}%  {tu:5.1f}s   speed={speed:.2f}x"
            )

    def agg(es, label):
        es = sorted(es)
        n = len(es)
        if not n:
            return
        print(
            f"{label}: med={es[n//2]:>4} p90={es[int(0.9*n)]:>5} p95={es[int(0.95*n)]:>5} "
            f"<=150={round(100*sum(e<=150 for e in es)/n)}% "
            f"<=500={round(100*sum(e<=500 for e in es)/n)}%  (n={n})"
        )
    print("\n===== AGGREGATE =====")
    agg(all_fast_err, "FAST")
    if not fast_only:
        agg(all_full_err, "FULL")
        print(f"TOTAL time: fast={tot_tf:.0f}s full={tot_tu:.0f}s  speedup={tot_tu/tot_tf:.2f}x")
    else:
        print(f"TOTAL fast time: {tot_tf:.0f}s  (full reference skipped)")


if __name__ == "__main__":
    main(sys.argv[1:])
