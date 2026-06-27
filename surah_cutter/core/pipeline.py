"""Per-sura orchestration: known-reciter shortcut, fast, hybrid, or full align.

Three paths today:

1. **Known reciter** — if the reciter has a reference timing set, use it
   directly (no audio model needed).
2. **Hybrid** (``mode='hybrid'``, default) — decode the audio, get cheap
   pause/boundary candidates (Silero VAD + structural prior, see
   :mod:`surah_cutter.core.candidates`), take every ayah whose expected
   boundary sits on a clear pause directly, and run the NN forced aligner only
   when the cheap stage is unreliable (missing/ambiguous pauses, or the pause
   count disagrees with the ayah count). Full-sura alignment is the accuracy
   fallback.
3. **Fast** (``mode='fast'``) — decode once, place rough boundaries from pause
   onsets, then refine each ayah start with short growing NN windows.
4. **Full** (``mode='full'``) — always decode + run the forced aligner over the
   sura's whole known word sequence (the original Phase-3 path).

Output rows ``(sura, ayah, start_ms)`` are shaped to feed straight into
``data.project_db.write_timings`` (which takes confidence as a separate arg).

**Resumption:** pass ``project_db`` (a :class:`~data.project_db.ProjectDB` or a
path to one) to skip ayahs already finalised by a human — any ayah whose
``review_meta.status`` is ``edited``/``verified`` keeps its stored start and is
not recomputed. ``auto`` rows and absent rows are (re)computed normally.

Heavy work (decode, alignment, VAD) is imported lazily so this module stays
importable without torch/onnxruntime.
"""

from __future__ import annotations

#: Default minimum candidate confidence for taking a pause-backed boundary
#: directly (no NN). Below this the ayah is "hard" and goes to the aligner.
PAUSE_ACCEPT_CONFIDENCE = 0.5


def process_surah(
    audio_path: str,
    surah: int,
    *,
    db_path: str | None = None,
    reciter: str | None = None,
    mode: str = "hybrid",
    project_db: object | None = None,
    trace: dict | None = None,
) -> list[tuple[int, int, int]]:
    """Produce per-ayah start timings for one sura.

    Args:
        audio_path: Path to the sura's audio file (one file = one sura).
        surah: Sura number (1..114).
        db_path: Optional path to the Quran text DB (``data/quran_mushaf.db``),
            used to read the sura's known word sequence and the structural
            prior. ``None`` uses the bundled default
            (:data:`quran_text.DEFAULT_DB_PATH`).
        reciter: Optional reciter identifier. If it matches a reference set, the
            stored timings are returned and no audio is processed.
        mode: ``'hybrid'`` (default), ``'fast'`` or ``'full'``. ``'hybrid'`` uses
            cheap pause/prior candidates and only aligns hard ayahs; ``'fast'``
            uses pause onsets plus growing NN windows; ``'full'`` always runs
            full-sura forced alignment.
        project_db: Optional working :class:`~data.project_db.ProjectDB` (or a
            path to one) for resumption. Ayahs whose review status is
            ``edited``/``verified`` keep their stored start and are not
            recomputed.

    Returns:
        Rows ``(sura, ayah, start_ms)``, one per ayah of the sura, ordered by
        ayah. Ready to pass to ``data.project_db.write_timings`` (confidence is
        a separate argument there).

    Raises:
        ValueError: If ``mode`` is not ``'hybrid'``, ``'fast'`` or ``'full'``.
    """
    if mode not in ("hybrid", "fast", "full"):
        raise ValueError(f"unknown mode {mode!r}; use 'hybrid', 'fast' or 'full'")

    # Lazy imports: keep module import light and free of heavy deps.
    from surah_cutter.data import quran_text, references

    # 1. Known-reciter shortcut.
    if reciter is not None:
        name = references.find_reciter(reciter)
        if name is not None:
            # Reference start times for the whole DB; keep this sura, sorted.
            timings = references.load_reference(name)
            return sorted(
                (sura, ayah, start_ms)
                for (sura, ayah), start_ms in timings.items()
                if sura == surah
            )

    # Resumption: ayahs already finalised by a human keep their stored start.
    frozen = _frozen_ayahs(project_db, surah)

    # Sura word sequence (also drives the structural prior in hybrid mode).
    if db_path is None:
        words = quran_text.surah_words(surah)
    else:
        words = quran_text.surah_words(surah, db_path=db_path)

    if mode == "full":
        rows = _full_align(audio_path, surah, words)
    elif mode == "fast":
        rows = _fast_align(audio_path, surah, words, trace=trace)
    else:
        rows = _hybrid(audio_path, surah, words, db_path=db_path)

    # Overlay frozen (edited/verified) ayahs: never overwrite manual work.
    if frozen:
        rows = [(s, a, frozen.get(a, t)) for (s, a, t) in rows]

    return rows


# --- full alignment (original Phase-3 path) -------------------------------


def _full_align(
    audio_path: str,
    surah: int,
    words: list,
) -> list[tuple[int, int, int]]:
    """Decode + full-sura forced alignment; one row per ayah.

    Lazy heavy imports (torch/onnx live behind ``core.align``).
    """
    from surah_cutter.core import align, decode

    pcm = decode.decode(audio_path)
    # The acoustic model emits a Latin transliteration; align_surah pulls the
    # known per-word transliterations via transliteration.surah_word_tr(surah).
    rows = align.align_surah(pcm, decode.DEFAULT_SR, words, surah=surah)
    # align_surah yields (ayah, start_ms, confidence); shape to write_timings'
    # (sura, ayah, time_ms) and drop confidence (a separate write_timings arg).
    return [(surah, ayah, start_ms) for ayah, start_ms, _conf in rows]


# --- fast alignment -------------------------------------------------------


def _fast_align(
    audio_path: str,
    surah: int,
    words: list,
    *,
    trace: dict | None = None,
) -> list[tuple[int, int, int]]:
    """Decode + two-phase fast alignment; one row per ayah."""
    from surah_cutter.core import align_fast, decode

    pcm = decode.decode(audio_path)
    rows = align_fast.align_surah_fast(
        pcm, decode.DEFAULT_SR, words, surah=surah, trace=trace
    )
    return [(surah, ayah, start_ms) for ayah, start_ms, _conf in rows]


# --- hybrid path ----------------------------------------------------------


def _hybrid(
    audio_path: str,
    surah: int,
    words: list,
    *,
    db_path: str | None = None,
) -> list[tuple[int, int, int]]:
    """Cheap pause/prior candidates first; NN-align only the hard ayahs.

    Strategy:

    1. Decode audio once; estimate total duration in ms.
    2. Detect speech pauses with Silero VAD (degrades to "no pauses" if the
       model is absent — see :func:`candidates.detect_pauses`).
    3. Build per-ayah boundary candidates from the structural prior matched to
       those pauses.
    4. Take every ayah whose candidate is pause-backed with high confidence
       directly (no NN). If *any* ayah is left hard — or the cheap stage is
       structurally unreliable (pause count != ayah count) — run a single
       full-sura forced alignment and fill the hard ayahs from it.

    Running one full-sura alignment for the hard set (rather than per-window
    crops) keeps the implementation minimal and avoids re-deriving frame strides
    on sliced audio; the easy, pause-backed ayahs still skip nothing-but-NN work
    on the common case where the cheap stage already nailed them.
    """
    from surah_cutter.core import candidates, decode

    pcm = decode.decode(audio_path)
    sr = decode.DEFAULT_SR
    total_ms = len(pcm) * 1000.0 / sr if sr else 0.0

    # Cheap pause detection (no NN). Absent VAD model -> [] (all ayahs hard).
    pauses = candidates.detect_pauses(pcm, sr)

    cand_kwargs = {} if db_path is None else {"db_path": db_path}
    cands = candidates.boundary_candidates(surah, total_ms, pauses, **cand_kwargs)

    n_ayahs = len(cands)
    # Structural sanity: межаятных пауз ровно (n_ayahs - 1)? Если число пауз
    # сильно расходится с числом границ — доверять сопоставлению нельзя.
    pause_count_ok = len(pauses) == max(n_ayahs - 1, 0)

    easy: dict[int, int] = {}
    hard: set[int] = set()
    for c in cands:
        if (
            c.from_pause
            and c.confidence >= PAUSE_ACCEPT_CONFIDENCE
            and pause_count_ok
        ):
            easy[c.ayah] = int(round(c.time_ms))
        else:
            # Including the first ayah: its true start follows the intro-basmala
            # (~6 s of audio not in `words`), which only the aligner recovers —
            # never assume the sura starts at 0 ms.
            hard.add(c.ayah)

    if not hard:
        # Cheap stage covered every boundary; no NN needed.
        rows = [(surah, a, easy[a]) for a in sorted(easy)]
        return _monotonic(rows)

    # Some ayahs are hard -> one full-sura forced alignment as the accuracy
    # source. Take aligner starts for the hard ayahs, keep cheap starts for the
    # easy ones.
    aligned = dict(_full_align_starts(audio_path, surah, words, pcm=pcm, sr=sr))

    rows: list[tuple[int, int, int]] = []
    for c in cands:
        a = c.ayah
        if a in easy:
            rows.append((surah, a, easy[a]))
        else:
            rows.append((surah, a, aligned.get(a, int(round(c.expected_ms)))))
    return _monotonic(rows)


def _full_align_starts(
    audio_path: str,
    surah: int,
    words: list,
    *,
    pcm=None,
    sr: int | None = None,
) -> list[tuple[int, int]]:
    """Run full-sura forced alignment; return ``(ayah, start_ms)`` per ayah.

    Reuses an already-decoded ``pcm`` when provided (the hybrid path decodes
    once) so we never decode the same file twice.
    """
    from surah_cutter.core import align, decode

    if pcm is None:
        pcm = decode.decode(audio_path)
        sr = decode.DEFAULT_SR
    rows = align.align_surah(pcm, sr or decode.DEFAULT_SR, words, surah=surah)
    return [(ayah, start_ms) for ayah, start_ms, _conf in rows]


# --- resumption & shaping -------------------------------------------------


def _frozen_ayahs(project_db: object | None, surah: int) -> dict[int, int]:
    """Map ``ayah -> stored start_ms`` for finalised (edited/verified) ayahs.

    ``project_db`` may be a :class:`~data.project_db.ProjectDB` or a path to a
    working DB. Returns an empty dict when no project DB is given or it has no
    finalised ayahs for ``surah``. Such ayahs must keep their stored start so
    regeneration never clobbers manual work.
    """
    if project_db is None:
        return {}

    # Duck-typed ProjectDB handle: read via its API (no mutation).
    if hasattr(project_db, "get_timings") and hasattr(project_db, "get_review"):
        timings = {
            (s, a): t for (s, a, t) in project_db.get_timings() if s == surah
        }
        frozen: dict[int, int] = {}
        for (s, a), t in timings.items():
            review = project_db.get_review(s, a)
            if review and review.get("status") in ("edited", "verified"):
                frozen[a] = int(t)
        return frozen

    # Path to a working DB: pure read-only sqlite so we never mutate it (a
    # ProjectDB.open would re-run the schema/db_version upsert).
    return _frozen_from_path(str(project_db), surah)


def _frozen_from_path(path: str, surah: int) -> dict[int, int]:
    """Read ``ayah -> start_ms`` for edited/verified ayahs of ``surah``.

    Read-only sqlite over an existing working DB; a missing/invalid DB yields an
    empty dict so resumption is simply skipped rather than erroring.
    """
    import os
    import sqlite3

    if not os.path.exists(path):
        return {}
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT t.ayah, t.time FROM timings t "
                "JOIN review_meta r ON r.sura = t.sura AND r.ayah = t.ayah "
                "WHERE t.sura = ? AND r.status IN ('edited', 'verified')",
                (surah,),
            ).fetchall()
    except sqlite3.Error:
        return {}
    return {int(a): int(t) for a, t in rows}


def _monotonic(
    rows: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Sort by ayah and clamp ``start_ms`` to be non-decreasing.

    Mixing cheap (pause) and NN starts can yield a small backward step at a
    boundary; clamp it so the output is monotonic like the aligner's own rows.
    """
    rows = sorted(rows, key=lambda r: r[1])  # by ayah
    out: list[tuple[int, int, int]] = []
    prev = 0
    for sura, ayah, start_ms in rows:
        if start_ms < prev:
            start_ms = prev
        prev = start_ms
        out.append((sura, ayah, start_ms))
    return out
