"""Local repair pass for suspicious automatic ayah timings."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def repair_surah_timings(
    pcm,
    sr: int,
    words: list,
    rows: list[tuple[int, int, int]],
    *,
    surah: int | None = None,
    bundle: Any = None,
    frozen: set[int] | None = None,
    max_shift_ms: int = 5000,
) -> list[tuple[int, int, int]]:
    """Repair recheck-suspicious non-frozen ayahs with local CTC alignment.

    ``rows`` are pipeline-shaped ``(sura, ayah, start_ms)``. Only ayahs flagged
    by :mod:`surah_cutter.core.recheck` as sounding like a neighbour are touched,
    and only if the repaired start stays monotonic between its neighbours.
    """
    if not words or not rows:
        return rows

    import numpy as np

    from surah_cutter.core import align as A
    from surah_cutter.core import recheck
    from surah_cutter.core._ctc import forced_align, merge_repeats
    from surah_cutter.core.align_fast import _default_bundle
    from surah_cutter.data import transliteration

    if surah is None:
        surah = words[0].surah
    if bundle is None:
        bundle = _default_bundle()
    frozen = frozen or set()

    timings = {ayah: start for _s, ayah, start in rows}
    checks = recheck.recheck_surah(pcm, sr, words, timings, surah=surah, bundle=bundle)
    suspects = [
        int(r["ayah"])
        for r in checks
        if r.get("verdict") == "shift" and int(r["ayah"]) not in frozen
    ]
    if not suspects:
        return rows

    word_trs = transliteration.surah_word_tr(surah)
    ayah_words: "OrderedDict[int, list[int]]" = OrderedDict()
    for i, w in enumerate(words):
        ayah_words.setdefault(w.ayah, []).append(i)
    ayahs = list(ayah_words)
    idx_of = {a: i for i, a in enumerate(ayahs)}
    total_ms = len(pcm) * 1000.0 / sr if sr else 0.0

    def align_window(lo_i: int, hi_i: int) -> dict[int, tuple[int, float]]:
        lo_a = ayahs[lo_i]
        hi_a = ayahs[hi_i]
        w0 = ayah_words[lo_a][0]
        w1 = ayah_words[hi_a][-1] + 1
        prev_t = timings.get(ayahs[lo_i - 1], 0) if lo_i > 0 else 0
        next_t = timings.get(ayahs[hi_i + 1], int(total_ms)) if hi_i + 1 < len(ayahs) else int(total_ms)
        t0 = 0.0 if lo_i == 0 else max(0.0, float(prev_t) - 800.0)
        t1 = min(total_ms, float(next_t) + 800.0)
        if t1 - t0 < 1000.0:
            return {}

        s0 = max(0, int(round(t0 * sr / 1000.0)))
        s1 = min(len(pcm), int(round(t1 * sr / 1000.0)))
        if s1 - s0 < int(0.3 * sr):
            return {}
        sub_words = words[w0:w1]
        sub_trs = word_trs[w0:w1]
        basmala = lo_i == 0 and surah not in (1, 9)
        basmala_trs = A._basmala_trs(transliteration) if basmala else None
        targets, spans = A._build_targets(sub_words, sub_trs, bundle, basmala_trs)
        if not targets:
            return {}

        emis = A._emissions(bundle, np.ascontiguousarray(pcm[s0:s1], dtype=np.float32), np)
        nf = int(emis.shape[0])
        if nf < 2:
            return {}
        stride = A._stride_ms(s1 - s0, sr, nf)
        base = s0 * 1000.0 / sr
        path, fscores = forced_align(
            np.ascontiguousarray(emis[None, :, :], dtype=np.float32),
            np.asarray(targets, dtype=np.int64)[None, :],
            blank=bundle.blank_id,
        )
        path = np.asarray(path).reshape(-1)
        fscores = np.asarray(fscores).reshape(-1)
        segs = [
            s for s in merge_repeats(path.tolist(), {k: k for k in range(emis.shape[1])})
            if s.label != bundle.blank_id
        ]
        tspans = [(s.start, s.end) for s in segs]
        while len(tspans) < len(targets):
            tspans.append(tspans[-1] if tspans else (nf - 1, nf - 1))

        out: dict[int, tuple[int, float]] = {}
        for a in ayahs[lo_i : hi_i + 1]:
            wi = ayah_words[a][0] - w0
            ts, te = spans[wi]
            if te <= ts or ts >= len(tspans):
                continue
            sf = tspans[ts][0]
            ef = tspans[te - 1][1] if te - 1 < len(tspans) else sf
            start = max(0, int(round(base + sf * stride)) - A.ALIGN_LAG_MS)
            seg = fscores[sf:max(sf + 1, ef + 1)]
            conf = float(np.exp(seg.mean())) if seg.size else 0.0
            out[a] = (start, conf)
        return out

    repaired = dict(timings)
    cache: dict[tuple[int, int], dict[int, tuple[int, float]]] = {}
    for a in suspects:
        k = idx_of.get(a)
        if k is None:
            continue
        lo_i = max(0, k - 2)
        hi_i = min(len(ayahs) - 1, k + 2)
        win_key = (lo_i, hi_i)
        local = cache.get(win_key)
        if local is None:
            local = align_window(lo_i, hi_i)
            cache[win_key] = local
        cand = local.get(a)
        if cand is None:
            continue
        new_t, conf = cand
        old_t = timings[a]
        prev_t = repaired.get(ayahs[k - 1], 0) if k > 0 else 0
        next_t = repaired.get(ayahs[k + 1], int(total_ms)) if k + 1 < len(ayahs) else int(total_ms)
        if conf < 0.15 or abs(new_t - old_t) > max_shift_ms:
            continue
        if prev_t <= new_t <= next_t:
            repaired[a] = int(new_t)

    return [(s, a, repaired.get(a, t)) for s, a, t in rows]
