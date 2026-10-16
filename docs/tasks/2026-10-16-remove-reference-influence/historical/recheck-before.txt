"""Re-check existing per-ayah timings against the audio (read-only verification).

Where forced alignment *places* boundaries, re-check *audits* boundaries that
already exist. For each ayah whose start time is stored, it greedy-decodes the
audio at that timecode — what the acoustic model actually *hears* there — and
compares it to that ayah's expected opening words. A timecode that sits on the
right words scores high; a drifted / off-by-one one scores low, and often a
neighbouring ayah's opening matches better, which pinpoints the slip.

This reuses the very signal the fast aligner uses as its placement-verification
fallback (``_VERIFY_SIM`` in :mod:`surah_cutter.core.align_fast`), here as an
explicit review aid. It is **strictly read-only**: it never writes timings or
review status, so it cannot clobber manual work — it only reports.

Robustness note: absolute similarity is a noisy per-ayah signal — on slow
murattal it ranges ~0.14–0.78 even on *correct* boundaries (greedy decode is
messy and confounds with *wasl* density; see the fast-mode failure landscape), so
no absolute threshold separates right from wrong. Validation on a known-correct
reciter confirmed this: an absolute floor flagged ~45% of correct ayahs, while
the *relative* test below produced **zero** false positives there yet caught
off-by-one (30/40), multi-second drift (+4 s → 20, +8 s → 26) and a localised
slip (precisely the corrupted ayahs). So the verdict is **purely relative**: does
a neighbouring ayah's opening match what's heard here better than this ayah's own
opening? If so it is flagged ``shift`` ("звучит как аят N"); otherwise ``ok``.
Repeated refrains (e.g. ar-Rahman's "fa-bi-ayyi ālā'i…") are acoustically
identical across ayahs and so score ``ok`` even when shifted — an inherent limit;
those are fixed by ear.

Cost mirrors one full-sura emissions pass (the model runs once over the sura);
the per-ayah greedy decodes are cheap frame slices of that single pass.
"""

from __future__ import annotations

import os
from typing import Any

#: A neighbouring ayah's opening must beat this ayah's own score by this margin to
#: be flagged as "звучит как аят N" (off-by-one / drift). Below the margin the lead
#: is melodic-stretch noise, not a real mismatch. Calibrated on a known-correct
#: reciter: 0.10 gave zero false positives while still catching off-by-one and
#: multi-second drift (the lead there is large). Env-tunable.
_SHIFT_MARGIN = float(os.environ.get("SC_RECHECK_SHIFT_MARGIN", "0.10"))
#: How many ayahs either side to test for a better match (drift is local).
_NEIGHBOR_BAND = int(os.environ.get("SC_RECHECK_BAND", "2"))
#: Window of audio greedy-decoded at each timecode (ms). Long enough to reach a
#: word that distinguishes ayahs sharing an opening phrase under slow melisma.
_DECODE_WIN_MS = 2500.0
#: The decode starts slightly *before* the stored onset so the first phoneme is
#: not clipped (the same -50 ms align_fast uses).
_DECODE_LEAD_MS = 50.0


def recheck_surah(
    pcm,
    sr: int,
    words: list,
    timings: dict[int, int],
    *,
    surah: int | None = None,
    bundle: Any = None,
) -> list[dict[str, Any]]:
    """Audit a sura's stored timings against the audio; one verdict per ayah.

    Args:
        pcm: Mono PCM at ``sr`` Hz (from :mod:`surah_cutter.core.decode`).
        sr: Sample rate of ``pcm`` (must equal the model's expected rate).
        words: The sura's :class:`~surah_cutter.data.quran_text.Word` objects in
            recitation order (``data.quran_text.surah_words``).
        timings: ``ayah -> start_ms`` for the ayahs to check (from the project
            DB). Ayahs absent here / mapped to ``None`` are reported ``no_time``.
        surah: Sura number; inferred from ``words[0].surah`` when omitted.
        bundle: Pre-loaded emitter bundle to reuse; loaded lazily otherwise.

    Returns:
        One dict per ayah (ordered by ayah):
        ``{ayah, time, score, verdict, match_ayah, match_score, hint}`` where
        ``score`` is the self-similarity ``[0,1]``, ``verdict`` is one of
        ``"ok" | "shift" | "no_time" | "insufficient_evidence"``, and ``match_ayah`` / ``match_score`` /
        ``hint`` describe a better-matching neighbour when one exists (``hint``
        empty for ``ok``). ``score`` is informational only — the verdict is
        purely relative (see the module docstring on why no absolute floor works).
    """
    import numpy as np

    from surah_cutter.core import align as A
    from surah_cutter.core.phonetics import onset_similarity as _sim
    from surah_cutter.data import transliteration

    if not words:
        return []
    if sr != A.EXPECTED_SR:
        raise ValueError(f"recheck_surah expects {A.EXPECTED_SR} Hz PCM, got {sr}")
    total_ms = len(pcm) * 1000.0 / sr
    if surah is None:
        surah = words[0].surah
    if bundle is None:
        bundle = A.default_bundle()
    if getattr(bundle, "text_kind", "phonetic") != "phonetic":
        raise ValueError("Проверка фонетических начал доступна только для Quran Phonetics.")

    word_trs = transliteration.surah_word_tr(surah)

    # ayah -> ordered global word indices (recitation order).
    ayah_words: dict[int, list[int]] = {}
    for i, w in enumerate(words):
        ayah_words.setdefault(w.ayah, []).append(i)
    ayah_list = sorted(ayah_words)
    idx_of = {a: k for k, a in enumerate(ayah_list)}

    # Expected opening (first ~4 words) per ayah — the comparison target.
    openings = {
        a: "".join(word_trs[i] for i in ayah_words[a][:4] if i < len(word_trs))
        for a in ayah_list
    }

    # One emissions pass over the whole sura; per-ayah decodes slice its frames.
    emissions = (A._emissions(bundle, np.ascontiguousarray(pcm, dtype=np.float32), np)
                 if len(pcm) else np.empty((0, len(bundle.vocab)), dtype=np.float32))
    n_frames = int(emissions.shape[0])
    stride_ms = A._stride_ms(len(pcm), sr, n_frames)
    inv_vocab = {v: k for k, v in bundle.vocab.items()}
    blank_id = bundle.blank_id

    def greedy(t0_ms: float, t1_ms: float) -> str:
        """Greedy CTC decode of audio ``[t0_ms, t1_ms]`` -> transliteration."""
        f0 = max(0, int(round(t0_ms / stride_ms))) if stride_ms else 0
        f1 = min(n_frames, int(round(t1_ms / stride_ms))) if stride_ms else 0
        if f1 - f0 < 2:
            return ""
        ids = emissions[f0:f1].argmax(-1)
        out: list[str] = []
        prev = -1
        for i in ids.tolist():
            if i != prev and i != blank_id:
                token = inv_vocab.get(int(i), "")
                if not token.startswith(("<", "[")):
                    out.append(token)
            prev = i
        return "".join(out)

    results: list[dict[str, Any]] = []
    for a in ayah_list:
        t = timings.get(a)
        if t is None:
            results.append({
                "ayah": a, "time": None, "score": 0.0, "verdict": "no_time",
                "match_ayah": None, "match_score": 0.0, "hint": "нет тайм-кода",
            })
            continue

        valid_time = type(t) is int and 0 <= t < total_ms
        dec = (greedy(t - _DECODE_LEAD_MS, t - _DECODE_LEAD_MS + _DECODE_WIN_MS)
               if valid_time else "")
        if not any(char.isalnum() for char in dec) or not openings.get(a):
            results.append({
                "ayah": a, "time": t, "score": 0.0, "verdict": "insufficient_evidence",
                "match_ayah": None, "match_score": 0.0,
                "hint": "недостаточно акустических данных",
            })
            continue
        self_sim = _sim(dec, openings.get(a, ""))

        # Relative test: does a nearby ayah's opening match what's heard better?
        k = idx_of[a]
        best_other_ayah: int | None = None
        best_other_sim = 0.0
        for kk in range(max(0, k - _NEIGHBOR_BAND), min(len(ayah_list), k + _NEIGHBOR_BAND + 1)):
            if kk == k:
                continue
            b = ayah_list[kk]
            s = _sim(dec, openings.get(b, ""))
            if s > best_other_sim:
                best_other_sim, best_other_ayah = s, b

        if best_other_ayah is not None and best_other_sim > self_sim + _SHIFT_MARGIN:
            verdict = "shift"
            hint = f"звучит как аят {best_other_ayah}"
        else:
            verdict = "ok"
            hint = ""

        results.append({
            "ayah": a,
            "time": int(t),
            "score": round(float(self_sim), 3),
            "verdict": verdict,
            "match_ayah": best_other_ayah,
            "match_score": round(float(best_other_sim), 3),
            "hint": hint,
        })

    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count verdicts for the UI summary line."""
    out = {"ok": 0, "shift": 0, "no_time": 0, "insufficient_evidence": 0}
    for r in results:
        out[r["verdict"]] = out.get(r["verdict"], 0) + 1
    out["checked"] = out["ok"] + out["shift"]
    out["suspect"] = out["shift"]
    out["unchecked"] = out["no_time"] + out["insufficient_evidence"]
    return out
