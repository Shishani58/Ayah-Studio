"""Pure-numpy CTC forced alignment — vendored to avoid heavy deps.

``ctc_forced_aligner.forced_align`` pulls in librosa/numba/scipy/llvmlite (~250 MB)
just for two small functions. Since the runtime ships an int8 ONNX model and does
not need any of that, we vendor the CTC forced-alignment Viterbi here in numpy.

API mirrors what :mod:`surah_cutter.core.align` consumes:

- :func:`forced_align` (log_probs ``(1, T, V)``, targets ``(1, L)``, blank id) ->
  ``(path (1, T), scores (1, T))`` where ``path`` is the per-frame vocab id of the
  aligned token (or blank) and ``scores`` are the per-frame log-probs of that id.
- :func:`merge_repeats` collapses consecutive equal ids into :class:`Segment`
  runs with inclusive ``end``.

This is forced alignment, not decoding: the token sequence is fixed; we only find
its best monotonic time path through the CTC trellis (blanks inserted between
tokens; a blank is mandatory between two identical adjacent tokens).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from surah_cutter.core.cancellation import check_cancelled

_NEG = -1.0e30


@dataclass
class Segment:
    """A run of one label over an inclusive frame range ``[start, end]``."""

    label: int
    start: int
    end: int


def forced_align(log_probs, targets, blank: int = 0):
    """Viterbi forced alignment of a fixed token sequence to CTC emissions.

    Args:
        log_probs: ``(1, T, V)`` log-softmax emissions over the vocab.
        targets: ``(1, L)`` target vocab ids (the fixed sequence to place).
        blank: CTC blank id.

    Returns:
        ``(path, scores)`` each ``(1, T)``: ``path[0, t]`` is the vocab id the
        frame is assigned to (a target token or ``blank``); ``scores[0, t]`` is
        the emission log-prob of that id at frame ``t``.
    """
    emis = np.asarray(log_probs, dtype=np.float64)[0]  # (T, V)
    tgt = np.asarray(targets).reshape(-1).astype(np.int64)  # (L,)
    t_len, _ = emis.shape
    n_tok = tgt.shape[0]
    if t_len == 0:
        raise ValueError("CTC requires audio frames")
    if n_tok == 0:
        path = np.full((1, t_len), blank, dtype=np.int64)
        return path, emis[np.arange(t_len), blank][None, :]

    # Extended state sequence: blank between every token -> [b, t0, b, t1, ..., b].
    n_st = 2 * n_tok + 1
    ext = np.full(n_st, blank, dtype=np.int64)
    ext[1::2] = tgt

    # A skip (s-2 -> s) is allowed only into a non-blank state whose token differs
    # from the one two states back (can't skip the mandatory blank between equal
    # adjacent tokens).
    allowed_skip = np.zeros(n_st, dtype=bool)
    allowed_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    dp = np.full(n_st, _NEG)
    dp[0] = emis[0, ext[0]]
    if n_st > 1:
        dp[1] = emis[0, ext[1]]

    # Each predecessor differs by only 0, 1 or 2 states. Store that choice,
    # not a full int64 state index (8x smaller, with identical tie breaking).
    back = np.zeros((t_len, n_st), dtype=np.uint8)
    idx = np.arange(n_st)
    prev1 = np.empty(n_st)
    prev2 = np.empty(n_st)
    best = np.empty(n_st)
    take = np.empty(n_st, dtype=bool)
    disallowed_skip = ~allowed_skip
    nonfinite = not np.isfinite(emis).all()

    from surah_cutter.core.progress import advance

    for t in range(1, t_len):
        if t % 64 == 0:
            check_cancelled()
            advance("alignment", t, t_len)
        prev1[0], prev1[1:] = _NEG, dp[:-1]
        prev2[:2], prev2[2:] = _NEG, dp[:-2]
        prev2[disallowed_skip] = _NEG
        if nonfinite:
            # Preserve argmax's first-NaN and infinity arithmetic behaviour too.
            stacked = np.stack((dp, prev1, prev2))
            choice = np.argmax(stacked, axis=0)
            dp = stacked[choice, idx] + emis[t, ext]
            back[t] = choice
        else:
            # Strict > keeps the original tie order: stay, then -1, then -2.
            # Reuse O(states) scratch instead of stacking/reducing every frame.
            choice = back[t]
            np.greater(prev1, dp, out=take)
            np.copyto(choice, np.uint8(1), where=take)
            np.maximum(dp, prev1, out=best)
            np.greater(prev2, best, out=take)
            np.copyto(choice, np.uint8(2), where=take)
            np.maximum(best, prev2, out=best)
            np.add(best, emis[t, ext], out=dp)

    if max(dp[n_st - 1], dp[n_st - 2]) <= _NEG / 2:
        raise ValueError("CTC target sequence cannot fit in audio")
    s = n_st - 1 if dp[n_st - 1] >= dp[n_st - 2] else n_st - 2
    states = np.zeros(t_len, dtype=np.int64)
    for t in range(t_len - 1, -1, -1):
        if t % 64 == 0:
            check_cancelled()
        states[t] = s
        s -= int(back[t, s])

    path = ext[states]
    scores = emis[np.arange(t_len), path]
    advance("alignment", t_len, t_len)
    return path[None, :], scores[None, :]


def forced_align_banded(log_probs, targets, blank: int = 0, radius: int = 256):
    """Banded Viterbi forced alignment with fallback-compatible output.

    The search is restricted to states near a linear token/time trajectory. This
    is an optimisation for tiled chunks, not a different alignment objective. If
    the band is too narrow and the final CTC state is unreachable, ``ValueError``
    is raised so callers can retry with :func:`forced_align`.
    """
    if radius <= 0:
        raise ValueError("band radius must be positive")

    emis = np.asarray(log_probs, dtype=np.float64)[0]
    tgt = np.asarray(targets).reshape(-1).astype(np.int64)
    t_len, _ = emis.shape
    n_tok = tgt.shape[0]
    if n_tok == 0:
        path = np.full((1, t_len), blank, dtype=np.int64)
        return path, emis[np.arange(t_len), blank][None, :]

    n_st = 2 * n_tok + 1
    ext = np.full(n_st, blank, dtype=np.int64)
    ext[1::2] = tgt

    allowed_skip = np.zeros(n_st, dtype=bool)
    allowed_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    idx = np.arange(n_st)

    def band_mask(t: int) -> np.ndarray:
        if t_len <= 1:
            center = n_st - 1
        else:
            center = (n_st - 1) * (t / (t_len - 1))
        lo = max(0, int(center) - radius)
        hi = min(n_st, int(center) + radius + 1)
        mask = np.zeros(n_st, dtype=bool)
        mask[lo:hi] = True
        return mask

    dp = np.full(n_st, _NEG)
    dp[0] = emis[0, ext[0]]
    if n_st > 1:
        dp[1] = emis[0, ext[1]]
    dp = np.where(band_mask(0), dp, _NEG)

    choices = np.zeros((t_len, n_st), dtype=np.uint8)

    for t in range(1, t_len):
        if t % 64 == 0:
            check_cancelled()
        stay = dp
        prev1 = np.concatenate(([_NEG], dp[:-1]))
        prev2 = np.concatenate(([_NEG, _NEG], dp[:-2]))
        prev2 = np.where(allowed_skip, prev2, _NEG)
        stacked = np.stack((stay, prev1, prev2))
        choice = np.argmax(stacked, axis=0).astype(np.uint8)
        best = stacked[choice, idx]
        dp = best + emis[t, ext]
        dp = np.where(band_mask(t), dp, _NEG)
        choices[t] = choice

    end_a = dp[n_st - 1]
    end_b = dp[n_st - 2] if n_st > 1 else _NEG
    if end_a <= _NEG / 2 and end_b <= _NEG / 2:
        raise ValueError("banded CTC path unreachable")

    s = n_st - 1 if end_a >= end_b else n_st - 2
    states = np.zeros(t_len, dtype=np.int64)
    for t in range(t_len - 1, -1, -1):
        if t % 64 == 0:
            check_cancelled()
        states[t] = s
        s -= int(choices[t, s])
        if s < 0:
            raise ValueError("banded CTC backtrace left trellis")

    path = ext[states]
    scores = emis[np.arange(t_len), path]
    return path[None, :], scores[None, :]


def merge_repeats(path, id_map=None):
    """Collapse consecutive equal ids in ``path`` into :class:`Segment` runs.

    ``id_map`` (optional) maps raw path values to labels; defaults to identity.
    ``end`` is inclusive (the last frame index of the run).
    """
    seq = list(path)
    segments: list[Segment] = []
    i = 0
    n = len(seq)
    while i < n:
        j = i
        while j + 1 < n and seq[j + 1] == seq[i]:
            j += 1
        label = seq[i] if id_map is None else id_map.get(seq[i], seq[i])
        segments.append(Segment(int(label), i, j))
        i = j + 1
    return segments


def aligned_segments(path, targets, *, blank: int = 0) -> list[Segment]:
    """Require one non-blank segment for every fixed target, in exact order."""
    segments = [segment for segment in merge_repeats(np.asarray(path).reshape(-1))
                if segment.label != blank]
    if [segment.label for segment in segments] != list(targets):
        raise ValueError("Не удалось сопоставить полный текст с аудио")
    return segments
