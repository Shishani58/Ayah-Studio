"""Forced alignment of known Quran text to recitation audio.

This is **forced alignment, not transcription**: the word sequence is fixed (the
Quran text is known) and we only recover *where in time* each word falls. There
is no argmax decoding.

Acoustic model: :data:`DEFAULT_MODEL_ID`
(``TBOGamer22/wav2vec2-quran-phonetics``), a ``Wav2Vec2ForCTC`` whose 38-symbol
vocabulary is a **Latin transliteration** (built from the ``word_tr`` column of
``Buraaq/quran-md-words``; ``'|'`` is the word delimiter, the PAD id is the CTC
blank). The known text is mapped to model tokens via
:func:`surah_cutter.data.transliteration.surah_word_tr`, so no G2P is needed.

Pipeline per call:

1. ``emissions = log_softmax(model(processor(pcm)).logits)`` over the vocab.
2. Build the fixed target id sequence: each word's transliteration chars -> vocab
   ids, words joined by the ``'|'`` id; chars absent from the vocab are dropped
   (counted, not guessed).
3. Forced-align that fixed sequence with
   :func:`ctc_forced_aligner.forced_align` (CTC, no decoding) to get a per-frame
   path, then per-token frame spans.
4. Convert frames to milliseconds via the model's true stride (computed from
   ``emission_len`` vs audio length), take each ayah's first word's start frame,
   and emit ``(ayah, start_ms, confidence)`` for every ayah, sorted and
   monotonic non-decreasing.

BASMALA: for every sura except al-Fatiha (1) and at-Tawba (9) the audio opens
with an intro-basmala that is **not** part of ``surah_words``. Leaving it out
makes forced alignment stretch word 1 back over those ~6 s (empirically a ~5.6 s
error on ayah 1 while ayahs 2+ stay within ~200 ms), so we **prepend** the
basmala's transliteration tokens (al-Fatiha ayah 1) as a prefix that absorbs the
leading audio; the prefix is not reported as a word/ayah. al-Fatiha's basmala
*is* ayah (1, 1) and is already in its words; at-Tawba has no basmala.

Heavy deps (``torch``, ``transformers``, ``ctc_forced_aligner``) are imported
lazily inside :func:`align_surah`; ``core`` stays importable without them. The
loaded model/processor bundle is cached across calls.

For long inputs, :func:`align_surah` estimates the exact Viterbi backtrace size
before running the acoustic model.  Jobs above the configured memory budget are
transparently routed through bounded exact tiles; explicit ``full`` mode and
automatic fallbacks therefore cannot allocate a whole-sura ``T × (2L+1)``
matrix.
"""

from __future__ import annotations

import os
from typing import Any

from surah_cutter.resources import external_model_path

from surah_cutter.data.quran_text import Word

#: Acoustic model for forced alignment (Wav2Vec2ForCTC, char-level CTC).
DEFAULT_MODEL_ID = "TBOGamer22/wav2vec2-quran-phonetics"

#: Sample rate the model expects (wav2vec2 family); ``core.decode`` produces it.
EXPECTED_SR = 16_000

#: CTC emission-lag compensation (ms), subtracted from every word start.
#: A CTC model emits each token's probability spike slightly *after* the acoustic
#: onset, so the raw first-token frame runs systematically late versus a human
#: onset mark — measured ~230 ms median (100% of ayahs late) on a labelled
#: reciter. The lag is a property of the model's receptive field, not the
#: reciter, so a fixed compensation is reciter-agnostic. Subtracting ~200 ms cut
#: the median error ~246 -> ~58 ms in validation. Override via the
#: ``SURAH_CUTTER_ALIGN_LAG_MS`` env var (set 0 to disable).
ALIGN_LAG_MS = int(os.environ.get("SURAH_CUTTER_ALIGN_LAG_MS", "200"))

#: Word-delimiter symbol in the model vocabulary.
WORD_DELIMITER = "|"

#: Backend that runs the acoustic model to produce emissions. ``"onnx"``
#: (default) runs the int8 ONNX export via :mod:`surah_cutter.core.emissions`
#: (no torch at inference — the shipped runtime path); ``"torch"`` keeps the
#: original oracle path (the ``align`` dev extra). The forced-alignment math is
#: identical either way — only the emission source differs. Overridable per call
#: via ``align_surah(..., backend=...)`` or the ``SURAH_CUTTER_ALIGN_BACKEND`` env.
DEFAULT_BACKEND = os.environ.get("SURAH_CUTTER_ALIGN_BACKEND", "onnx").strip().lower()

#: Keep the previous int64-sized routing estimate conservative even though the
#: backtrace now stores uint8 choices, so this optimisation never widens jobs.
#: Above this budget the same exact text is aligned through bounded tiles instead
#: of allocating a whole-sura backtrace matrix.  The estimate uses wav2vec2's
#: stable ~20 ms frame stride and is deliberately conservative.
_MAX_FULL_CTC_BACKTRACE_BYTES = int(
    float(os.environ.get("SC_MAX_FULL_CTC_BACKTRACE_MB", "1024")) * 1024 * 1024
)
_CTC_FRAME_STRIDE_MS_ESTIMATE = 20.0

#: Default path to the int8 ONNX model (the artefact written by
#: :mod:`surah_cutter.tools.export_onnx`). Overridable via the
#: ``SURAH_CUTTER_ONNX_MODEL`` env var. Resolved lazily so the import is free.
#: align.py -> core/ -> surah_cutter/ -> <repo>/models/...
_DEFAULT_ONNX_MODEL = os.environ.get(
    "SURAH_CUTTER_ONNX_MODEL",
    str(external_model_path("wav2vec2-quran-phonetics.onnx")),
)

_ORACLE_HINT = (
    "Forced alignment needs torch + transformers + ctc-forced-aligner. "
    "Install the optional extra with:\n"
    "    pip install -e '.[oracle]'"
)

# Process-wide cache of (model, processor, vocab) keyed by model id, so repeated
# per-sura calls don't reload weights.
_BUNDLE_CACHE: dict[str, "_Bundle"] = {}

# Process-wide cache of ONNX emitters keyed by model path.
_ONNX_CACHE: dict[str, "_OnnxBundle"] = {}


class _Bundle:
    """A loaded model + processor + resolved vocab/blank ids."""

    __slots__ = ("model", "processor", "vocab", "blank_id", "delim_id")

    def __init__(self, model: Any, processor: Any, vocab: dict[str, int]) -> None:
        self.model = model
        self.processor = processor
        self.vocab = vocab
        self.blank_id = _resolve_blank_id(processor, vocab)
        self.delim_id = vocab.get(WORD_DELIMITER)


class _OnnxBundle:
    """ONNX-backed parity surrogate for :class:`_Bundle`.

    Exposes the **same** ``vocab`` / ``blank_id`` / ``delim_id`` interface the
    target-building and forced-alignment code reads, but produces emissions via
    :mod:`surah_cutter.core.emissions` (int8 ONNX) instead of torch. Holds no
    torch state, so the inference path needs no torch/transformers.
    """

    __slots__ = ("emitter", "vocab", "blank_id", "delim_id", "text_kind", "lag_ms")

    def __init__(self, emitter: Any) -> None:
        self.emitter = emitter
        self.vocab = emitter.vocab
        self.blank_id = emitter.blank_id
        self.delim_id = emitter.vocab.get(WORD_DELIMITER)
        self.text_kind = "phonetic"
        self.lag_ms = ALIGN_LAG_MS
        if hasattr(emitter, "text_kind"):
            self.text_kind = emitter.text_kind
            self.lag_ms = 0


def _estimated_ctc_backtrace_bytes(
    n_samples: int,
    sr: int,
    n_targets: int,
) -> int:
    """Conservatively estimate the exact CTC Viterbi backtrace allocation."""
    if n_samples <= 0 or sr <= 0 or n_targets <= 0:
        return 0
    samples_per_frame = max(
        1,
        int(round(sr * _CTC_FRAME_STRIDE_MS_ESTIMATE / 1000.0)),
    )
    n_frames = (int(n_samples) + samples_per_frame - 1) // samples_per_frame
    n_states = 2 * int(n_targets) + 1
    return n_frames * n_states * 8


def _full_ctc_requires_tiling(
    n_samples: int,
    sr: int,
    n_targets: int,
    *,
    max_backtrace_bytes: int = _MAX_FULL_CTC_BACKTRACE_BYTES,
) -> bool:
    """Whether exact whole-sura CTC exceeds the configured memory budget."""
    return (
        _estimated_ctc_backtrace_bytes(n_samples, sr, n_targets)
        > max(0, int(max_backtrace_bytes))
    )


def align_surah(
    pcm,
    sr: int,
    words: list[Word],
    *,
    model: Any | None = None,
    surah: int | None = None,
    backend: str | None = None,
    trace: dict | None = None,
    log_event: Any | None = None,
    ayah_progress: Any | None = None,
) -> list[tuple[int, int, float]]:
    """Forced-align a sura's known words to its audio; return per-ayah starts.

    Args:
        pcm: Mono PCM samples at ``sr`` Hz (float32, from ``core.decode``).
        sr: Sample rate of ``pcm``; must equal :data:`EXPECTED_SR`.
        words: The sura's :class:`Word` objects in recitation order (from
            ``data.quran_text.surah_words``). Each carries its ayah number.
        model: Optional pre-loaded bundle (a private ``_Bundle`` for the torch
            backend or ``_OnnxBundle`` for the onnx backend) to reuse. When
            ``None`` the default model is loaded lazily and cached process-wide.
        surah: Optional sura number; inferred from ``words[0].surah`` if omitted.
            Used to look up the per-word transliterations.
        backend: Emission backend, ``"torch"`` (default) or ``"onnx"``. Only
            selects how emissions are computed; the forced-alignment math is
            identical. ``None`` uses :data:`DEFAULT_BACKEND`. Ignored when an
            explicit ``model`` bundle is passed (its type decides the backend).

    Returns:
        Rows ``(ayah, start_ms, confidence)`` for every ayah of the sura, ordered
        by ayah, with strictly increasing ``start_ms`` and ``confidence`` in
        ``[0, 1]`` (first-word token probability, excluding blank frames).
        Weak first-token evidence and pause rechecks are recorded in ``trace``.

    Raises:
        ImportError: If the alignment extra is not installed.
        ValueError: If ``sr`` does not match :data:`EXPECTED_SR` or ``words`` is
            empty.
    """
    if sr != EXPECTED_SR:
        raise ValueError(f"align_surah expects {EXPECTED_SR} Hz PCM, got {sr}")
    if not words:
        raise ValueError("align_surah got an empty word list")

    if surah is None:
        surah = words[0].surah

    # Resolve the backend: an explicit bundle decides it, else the argument /
    # module default. The onnx backend avoids torch entirely.
    if isinstance(model, _OnnxBundle):
        use_onnx = True
    elif isinstance(model, _Bundle):
        use_onnx = False
    else:
        use_onnx = (backend or DEFAULT_BACKEND).strip().lower() == "onnx"
        from surah_cutter.core import model_store

        if model_store.ACTIVE_ID != "quran-phonetics":
            if backend is not None and backend.strip().lower() != "onnx":
                raise ValueError("Выбранная модель не поддерживает этот backend.")
            # A legacy oracle environment must never replace the selected
            # weights with DEFAULT_MODEL_ID.
            use_onnx = True

    # Forced-alignment math is pure numpy (vendored in core._ctc), backend-
    # agnostic and torch-free; torch is only pulled in for the torch emission
    # backend (see _torch_emissions / _get_bundle).
    import numpy as np

    from surah_cutter.core._ctc import forced_align, aligned_segments
    from surah_cutter.data import transliteration

    if use_onnx:
        bundle = model if isinstance(model, _OnnxBundle) else _get_onnx_bundle()
    else:
        bundle = model if isinstance(model, _Bundle) else _get_bundle(DEFAULT_MODEL_ID)

    pause_full = _uses_pause_full(bundle)
    if pause_full and trace is None:
        trace = {}
    if pause_full and log_event is not None:
        log_event("FULL: участки около 20 с по сильным паузам, 3 одновременных вызова")
    if trace is not None:
        trace["model_source"] = bundle_source(bundle)

    if getattr(bundle, "text_kind", "phonetic") == "whisper":
        word_rows = bundle.emitter.align_words(pcm, words, intro=surah not in (1, 9), strict_words=False)
        first = {}
        for word, (_, start, confidence) in zip(words, word_rows):
            if ayah_progress is not None and word.ayah not in first:
                ayah_progress(word.ayah)
            first.setdefault(word.ayah, (word.ayah, start, confidence))
        rows = list(first.values())
        if any(b[1] <= a[1] for a, b in zip(rows, rows[1:])):
            raise ValueError("Границы аятов Whisper не различимы.")
        if trace is not None:
            trace.update({"mode": "full", "backend": "whisper-attention-dtw",
                          "align_lag_ms": 0, "ayahs": [
                              {"ayah": a, "onset_ms": ms, "conf": c, "reliable": c >= .25}
                              for a, ms, c in rows]})
        return rows

    # --- 1. build the fixed target id sequence ------------------------------
    custom_targets = (hasattr(getattr(bundle, "emitter", None), "build_targets")
                      or getattr(bundle, "text_kind", "phonetic") == "arabic")
    word_trs = [] if custom_targets else transliteration.surah_word_tr(surah)
    # Every sura except al-Fatiha (1, whose basmala IS ayah 1) and at-Tawba (9,
    # which has no basmala) opens with an intro-basmala in the audio that is not
    # in ``words``. Prepend its tokens so word 1 isn't dragged back over it.
    basmala_trs = ([] if custom_targets else _basmala_trs(transliteration)) if surah not in (1, 9) else None
    targets, word_token_spans = _build_targets(words, word_trs, bundle, basmala_trs)

    # Whole-sura Viterbi is O(frames × target states) in both time and memory.
    # Route oversized exact jobs through bounded exact tiles before computing
    # emissions, so a long sura cannot allocate hundreds of GiB merely because
    # a caller explicitly selected the full path or a fallback escalated here.
    estimated_targets = len(targets)
    if getattr(bundle, "text_kind", "").startswith("fastconformer"):
        estimated_targets = max(estimated_targets, 2 * sum(len(w.text) for w in words) + 256)
    if targets and _full_ctc_requires_tiling(len(pcm), sr, estimated_targets):
        from surah_cutter.core import align_tiled

        kwargs = {"surah": surah, "bundle": bundle, "check_onsets": True}
        if pause_full:
            kwargs["parallel_full"] = True
        if trace is not None:
            kwargs["trace"] = trace
        if log_event is not None:
            kwargs["log_event"] = log_event
        if ayah_progress is not None:
            kwargs["ayah_progress"] = ayah_progress
        rows = align_tiled.align_surah_tiled(pcm, sr, words, **kwargs)
        if pause_full:
            _review_pause_full_onsets(trace["ayahs"], trace["full_windows"], log_event=log_event)
        return rows

    # --- 2. emissions: log-probs over the vocab, shape (T, V) ---------------
    emissions = _full_emissions(bundle, pcm, np, trace=trace)  # float32 (T, V)
    if getattr(bundle, "text_kind", "").startswith("fastconformer"):
        targets, word_token_spans = _build_targets(words, word_trs, bundle, basmala_trs)
    n_frames = emissions.shape[0]
    stride_ms = _stride_ms(len(pcm), sr, n_frames)

    if not targets:
        raise ValueError("Нет текста для точного выравнивания суры")
    if any(end <= start for start, end in word_token_spans):
        raise ValueError("Не все слова имеют токены для точного выравнивания")

    targets_arr = np.asarray(targets, dtype=np.int64)[None, :]  # (1, L)
    log_probs = np.ascontiguousarray(emissions[None, :, :], dtype=np.float32)

    # --- 3. forced alignment (CTC, no decoding) -----------------------------
    path, frame_scores = forced_align(log_probs, targets_arr, blank=bundle.blank_id)
    path = np.asarray(path).reshape(-1)
    frame_scores = np.asarray(frame_scores).reshape(-1)

    # Collapse the per-frame path into non-blank segments in target order. Each
    # non-blank segment corresponds to one target position (CTC separates equal
    # consecutive targets with blanks, so counts line up).
    nonblank = aligned_segments(path, targets, blank=bundle.blank_id)
    target_spans: list[tuple[int, int]] = [(s.start, s.end) for s in nonblank]

    # --- per-token frames -> per-word spans -> per-ayah first-word start ----
    rows = _ayah_rows(
        words,
        word_token_spans,
        target_spans,
        frame_scores,
        stride_ms,
        np,
    )
    from surah_cutter.core import candidates

    pauses = candidates.detect_energy_pauses(pcm, sr, use_ten_vad=False)
    checks = _check_ayah_onsets(
        words, word_token_spans, target_spans, frame_scores, stride_ms, pauses=pauses,
        lag_ms=getattr(bundle, "lag_ms", ALIGN_LAG_MS),
        ayah_progress=ayah_progress,
    )
    if pause_full:
        _review_pause_full_onsets(checks.values(), trace["full_windows"])
    rows = [(a, checks[a]["onset_ms"], checks[a]["conf"]) for a, _, _ in rows]
    if any(right[1] <= left[1] for left, right in zip(rows, rows[1:])):
        raise ValueError("Границы аятов не различимы; требуется проверка записи")
    if trace is not None:
        trace.update({
            "mode": "full", "total_ms": len(pcm) * 1000.0 / sr,
            "emission_passes": 1, "emissions_frames": n_frames,
            "backend": getattr(getattr(bundle, "emitter", None), "model_id", "onnx") if use_onnx else "torch",
            "batch_size": 1 if use_onnx else None,
            "align_lag_ms": getattr(bundle, "lag_ms", ALIGN_LAG_MS),
            "ayahs": list(checks.values()),
        })
    if log_event is not None:
        for check in checks.values():
            if check["repaired"] or not check["reliable"]:
                reason = {
                    "weak-onset": "начало слова не подтверждено, требуется ревью",
                    "weak-word": "слово распознано неуверенно, требуется ревью",
                    "weak-prefix-across-pause": "слабое начало перепроверено по паузе",
                    "weak-word-token": "звук первого слова не подтверждён, требуется ревью",
                    "late-onset-in-pause": "сомнительное начало в длинной паузе, требуется ревью",
                }.get(check["reason"], check["reason"])
                log_event(
                    f"Аят {check['ayah']}: {reason}, "
                    f"исходный старт {check['raw_onset_ms'] / 1000:.3f} сек., "
                    f"старт {check['onset_ms'] / 1000:.3f} сек., "
                    f"conf={check['conf']:.4f}, первый звук={check['first_token_conf']:.4f}"
                )
    return rows


def _check_ayah_onsets(
    words, word_token_spans, target_spans, frame_scores, stride_ms,
    *, pauses, base_ms: float = 0.0, lag_ms: int | None = None,
    ayah_progress: Any | None = None,
) -> dict[int, dict]:
    """Check first-word evidence without letting blank frames hide weak tokens.

    A weak prefix stretched across a strict DSP pause can belong to the
    preceding sound. Move to the pause centre only if every token before that
    pause is weak and a later token of the same word is strongly supported.
    No pause/no strong suffix/competing pauses means review, not a guessed move.
    ``base_ms`` lets FAST reuse this check on saved emission slices.
    """
    import numpy as np

    checks: dict[int, dict] = {}
    for word, (ts, te) in zip(words, word_token_spans):
        if word.ayah in checks:
            continue
        if ayah_progress is not None:
            ayah_progress(word.ayah)
        spans = target_spans[ts:te]
        if not spans or len(spans) != te - ts:
            raise ValueError("Неполный текст начала аята")
        token_confs = [_span_confidence(frame_scores, sf, ef) for sf, ef in spans]
        token_scores = np.concatenate([
            frame_scores[sf:ef + 1] for sf, ef in spans
        ])
        conf = _span_confidence(token_scores, 0, len(token_scores) - 1)
        raw_ms = max(0, round(base_ms + spans[0][0] * stride_ms)
                     - (ALIGN_LAG_MS if lag_ms is None else lag_ms))
        check = {
            "ayah": word.ayah, "raw_onset_ms": raw_ms, "onset_ms": raw_ms,
            "conf": conf, "first_token_conf": token_confs[0],
            "reliable": token_confs[0] >= 0.25 and conf >= 0.25,
            "repaired": False,
            "reason": ("weak-onset" if token_confs[0] < 0.25 else
                       "weak-word" if conf < 0.25 else "ctc"),
            "token_confs": token_confs,
        }
        if token_confs[0] < 0.25:
            strong = next((i for i, c in enumerate(token_confs) if c >= 0.45), None)
            if strong is not None:
                first_ms = base_ms + spans[0][0] * stride_ms
                strong_ms = base_ms + spans[strong][0] * stride_ms
                supported = [
                    p for p in pauses
                    if first_ms < p.start_ms < p.end_ms <= strong_ms
                    and all(
                        c < 0.25 for (sf, _), c in zip(spans, token_confs)
                        if base_ms + sf * stride_ms < p.end_ms
                    )
                ]
                if len(supported) == 1:
                    pause = supported[0]
                    check.update({
                        "onset_ms": round(pause.center_ms), "repaired": True,
                        "reason": "weak-prefix-across-pause",
                        "pause_start_ms": pause.start_ms, "pause_end_ms": pause.end_ms,
                    })
        checks[word.ayah] = check
    return checks


# --- emissions & stride ---------------------------------------------------


def _uses_pause_full(bundle: Any) -> bool:
    from surah_cutter.core.emissions import OnnxEmitter

    return (isinstance(bundle, _OnnxBundle)
            and isinstance(bundle.emitter, OnnxEmitter)
            and bundle.text_kind == "phonetic")


def _full_emissions(bundle: Any, pcm, np, *, trace: dict | None = None) -> Any:
    """Apply the QPC FULL policy without changing FAST/local or other emitters."""
    if not _uses_pause_full(bundle):
        return _emissions(bundle, pcm, np)
    from surah_cutter.core import full_windows

    window_trace = {"algorithm": full_windows.FULL_POLICY,
                    "review_policy": full_windows.FULL_REVIEW_POLICY}
    if trace is not None:
        trace["full_windows"] = window_trace
    matrix = full_windows.generate_pause_emissions(
        bundle.emitter, np.ascontiguousarray(pcm, dtype=np.float32),
        window_s=full_windows.FULL_WINDOW_S, workers=full_windows.FULL_WORKERS,
        strong_pauses=True, max_window_s=full_windows.FULL_MAX_WINDOW_S,
        trace=window_trace,
    )
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _review_pause_full_onsets(checks, windows: dict, *, log_event=None) -> None:
    """Conservative FULL warnings; never move a boundary or change confidence.

    High mean word probability can hide an unsupported phoneme. A late mark
    inside a long silence is also ambiguous despite high CTC probability.
    These gates address the observed unflagged 83:10/95:8 without ayah IDs,
    reference timings, extra inference, or changing FAST's onset checker.
    """
    windows.update(review_min_token_conf=.05, review_long_pause_ms=700,
                   review_late_ms=200)
    pauses = [p for p in windows.get("pause_decisions", []) if p["accepted"]]
    for check in checks:
        if not check.get("reliable"):
            continue
        tokens = check.get("token_confs", [])
        reason = None
        if tokens and min(tokens) < windows["review_min_token_conf"]:
            reason = "weak-word-token"
        else:
            onset = check["onset_ms"]
            if any(p["duration_ms"] >= windows["review_long_pause_ms"]
                   and p["start_ms"] <= onset <= p["end_ms"]
                   and onset - p["center_ms"] > windows["review_late_ms"] for p in pauses):
                reason = "late-onset-in-pause"
        if reason:
            check.update(reliable=False, reason=reason, review_policy=windows.get("review_policy"))
            if log_event is not None:
                log_event(f"Аят {check['ayah']}: сомнительная граница ({reason}), требуется ревью")


def _emissions(bundle: Any, pcm, np) -> Any:
    """Run the model and return log-softmax emissions of shape ``(T, V)``.

    Dispatches on the bundle type: ``_OnnxBundle`` runs the int8 ONNX export via
    :mod:`surah_cutter.core.emissions`; ``_Bundle`` runs the torch model. Both
    return the identical ``(T, V)`` log-softmax over the same vocab order, so the
    downstream forced alignment is unaffected by the choice.
    """
    from surah_cutter.core.cancellation import check_cancelled

    check_cancelled()
    if isinstance(bundle, _OnnxBundle):
        emissions = bundle.emitter.emissions(np.ascontiguousarray(pcm, dtype=np.float32))
        return np.ascontiguousarray(emissions, dtype=np.float32)
    return _torch_emissions(bundle, pcm, np)


def _torch_emissions(bundle: "_Bundle", pcm, np) -> Any:
    """Torch backend: run the model and return log-softmax emissions ``(T, V)``.

    Done under ``torch.inference_mode`` on CPU; the result is detached to numpy
    float32. The processor normalises the raw waveform as the model expects.
    """
    import torch

    inputs = bundle.processor(
        pcm,
        sampling_rate=EXPECTED_SR,
        return_tensors="pt",
    )
    input_values = inputs.input_values
    with torch.inference_mode():
        logits = bundle.model(input_values).logits  # (1, T, V)
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[0].to(torch.float32).cpu().numpy()


def _stride_ms(n_samples: int, sr: int, n_frames: int) -> float:
    """Milliseconds of audio per emission frame, from true ratios.

    Computed as ``(n_samples / sr) * 1000 / n_frames`` rather than assuming the
    nominal ~20 ms, so any model-specific receptive field / padding is absorbed.
    """
    if n_frames <= 0:
        return 20.0
    return (n_samples / sr) * 1000.0 / n_frames


# --- target construction --------------------------------------------------

_BASMALA_TRS_CACHE: list[str] | None = None


def _basmala_trs(transliteration: Any) -> list[str]:
    """Transliterations of the basmala words (al-Fatiha ayah 1), cached.

    The intro-basmala recited before most suras is textually al-Fatiha's first
    ayah; we reuse its per-word transliterations as the prefix tokens.
    """
    global _BASMALA_TRS_CACHE
    if _BASMALA_TRS_CACHE is None:
        from surah_cutter.data import quran_text

        fatiha_words = quran_text.surah_words(1)
        fatiha_trs = transliteration.surah_word_tr(1)
        _BASMALA_TRS_CACHE = [
            tr for w, tr in zip(fatiha_words, fatiha_trs) if w.ayah == 1
        ]
    return _BASMALA_TRS_CACHE


def _build_targets(
    words: list[Word],
    word_trs: list[str],
    bundle: "_Bundle",
    basmala_trs: list[str] | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Build the flat target id sequence and per-word target-position spans.

    For each word: map its transliteration chars to vocab ids (dropping unknown
    chars), then append a single ``'|'`` delimiter id *between* words (not after
    the last). Returns:

    - ``targets``: the flat list of vocab ids (the fixed CTC target sequence);
    - ``word_token_spans``: for each input word, a ``(start, end)`` half-open
      range of indices into ``targets`` covering that word's *letter* tokens
      (delimiters excluded). Words with no usable tokens get an empty span
      ``(k, k)`` so per-ayah grouping can still find a neighbour.

    ``basmala_trs`` (when given) are transliterations of the intro-basmala words
    that the reciter utters *before* the sura but that are **not** part of
    ``words``. Their tokens are prepended to ``targets`` (each followed by a
    delimiter, including before the first real word) so forced alignment assigns
    the leading basmala audio to them; they get **no** entry in
    ``word_token_spans``, so they are never emitted as a word/ayah — they only
    keep word 1 from being stretched back over the basmala.
    """
    if hasattr(getattr(bundle, "emitter", None), "build_targets"):
        return bundle.emitter.build_targets(words, intro=basmala_trs is not None)
    if getattr(bundle, "text_kind", "phonetic") == "arabic":
        from surah_cutter.data import quran_text

        word_trs = [_arabic_target(w.text, bundle.vocab) for w in words]
        if basmala_trs is not None:
            basmala_trs = [_arabic_target(w.text, bundle.vocab)
                           for w in quran_text.surah_words(1) if w.ayah == 1]
    vocab = bundle.vocab
    delim_id = bundle.delim_id

    targets: list[int] = []
    word_token_spans: list[tuple[int, int]] = []

    # Intro-basmala prefix (consumed by the leading audio, not reported).
    for btr in basmala_trs or ():
        for ch in btr:
            tid = vocab.get(ch)
            if tid is not None:
                targets.append(tid)
        if delim_id is not None:
            targets.append(delim_id)

    for w_idx, word in enumerate(words):
        tr = word_trs[w_idx] if w_idx < len(word_trs) else ""
        start = len(targets)
        for ch in tr:
            tid = vocab.get(ch)
            if tid is None:
                # Char not in the transliteration vocab; skip it (do not guess).
                continue
            targets.append(tid)
        end = len(targets)
        word_token_spans.append((start, end))
        # Word delimiter between words (skip after the final word).
        if delim_id is not None and w_idx < len(words) - 1:
            targets.append(delim_id)

    return targets, word_token_spans


# --- frame/ms mapping -----------------------------------------------------


def _ayah_rows(
    words: list[Word],
    word_token_spans: list[tuple[int, int]],
    target_spans: list[tuple[int, int]],
    frame_scores,
    stride_ms: float,
    np,
) -> list[tuple[int, int, float]]:
    """Collapse aligned word spans into per-ayah ``(ayah, start_ms, conf)`` rows.

    For each word we take the first-token start frame as the word's start frame
    (or, for words with no tokens, fall back to the previous word's start). The
    first word of each ayah gives the ayah start. Confidence is the mean
    per-frame path probability over the ayah-first word's token span.
    """
    n_words = len(words)

    # word -> (start_frame, end_frame) using its first/last letter token spans.
    word_start_frame: list[int | None] = [None] * n_words
    word_end_frame: list[int | None] = [None] * n_words
    for w_idx, (ts, te) in enumerate(word_token_spans):
        if te > ts:  # word has at least one usable token
            sf = target_spans[ts][0]
            ef = target_spans[te - 1][1]
            word_start_frame[w_idx] = sf
            word_end_frame[w_idx] = ef

    # Fill token-less words (rare) by carrying the previous word's frame forward.
    last_frame = 0
    for w_idx in range(n_words):
        if word_start_frame[w_idx] is None:
            word_start_frame[w_idx] = last_frame
            word_end_frame[w_idx] = last_frame
        else:
            last_frame = word_start_frame[w_idx]  # type: ignore[assignment]

    # First word index per ayah (recitation order => first occurrence wins).
    first_word_of_ayah: dict[int, int] = {}
    for w_idx, word in enumerate(words):
        if word.ayah not in first_word_of_ayah:
            first_word_of_ayah[word.ayah] = w_idx

    scores = np.asarray(frame_scores, dtype=np.float32)

    rows: list[tuple[int, int, float]] = []
    for ayah in sorted(first_word_of_ayah):
        w_idx = first_word_of_ayah[ayah]
        sf = int(word_start_frame[w_idx] or 0)
        ef = int(word_end_frame[w_idx] or sf)
        # Compensate the CTC emission lag (see ALIGN_LAG_MS); clamp at 0.
        start_ms = max(0, int(round(sf * stride_ms)) - ALIGN_LAG_MS)
        conf = _span_confidence(scores, sf, ef)
        rows.append((ayah, start_ms, conf))

    # Enforce monotonic non-decreasing start_ms (numerical/alignment slack).
    rows.sort(key=lambda r: r[0])
    prev = 0
    fixed: list[tuple[int, int, float]] = []
    for ayah, start_ms, conf in rows:
        if start_ms < prev:
            start_ms = prev
        prev = start_ms
        fixed.append((ayah, start_ms, conf))
    return fixed


def _span_confidence(scores, start_frame: int, end_frame: int) -> float:
    """Mean per-frame probability over ``[start_frame, end_frame]`` -> ``[0,1]``.

    ``scores`` holds per-frame log-probs from the forced path; we exponentiate
    and average. Empty/degenerate spans yield ``0.0``.
    """
    import math

    if len(scores) == 0 or end_frame < start_frame:
        return 0.0
    lo = max(0, start_frame)
    hi = min(len(scores) - 1, end_frame)
    if hi < lo:
        return 0.0
    window = scores[lo : hi + 1]
    mean_logp = float(window.mean())
    # Clamp to avoid math range issues on extreme log-probs.
    mean_logp = max(min(mean_logp, 0.0), -50.0)
    return round(math.exp(mean_logp), 4)


def _uniform_fallback(
    words: list[Word],
    n_frames: int,
    stride_ms: float,
) -> list[tuple[int, int, float]]:
    """Last-resort even split when no target tokens could be built.

    Distributes ayah starts uniformly across the audio so the pipeline still
    emits one row per ayah. Confidence is ``0.0`` to flag the degeneracy.
    """
    ayahs = sorted({w.ayah for w in words})
    total_ms = int(round(n_frames * stride_ms))
    rows: list[tuple[int, int, float]] = []
    n = len(ayahs)
    for i, ayah in enumerate(ayahs):
        start_ms = int(round(total_ms * i / n)) if n else 0
        rows.append((ayah, start_ms, 0.0))
    return rows


# --- model loading & vocab resolution -------------------------------------


def _arabic_target(text: str, vocab: dict[str, int]) -> str:
    """Normalise inference text only; the canonical Quran text stays untouched."""
    import unicodedata

    result = []
    normalised = unicodedata.normalize("NFC", text).translate(str.maketrans({
        "ٱ": "ا", "ۥ": "و", "ۦ": "ي", "۞": "", "۩": "",
    }))
    for ch in normalised:
        if ch in vocab and ch != "ـ":
            result.append(ch)
        elif unicodedata.category(ch) in ("Mn", "Me", "Cf") or ch.isspace() or ch == "ـ":
            continue
        else:
            raise ValueError(f"Символ {ch!r} не представлен в словаре арабской модели.")
    if not result:
        raise ValueError("Слово не представлено в словаре арабской модели.")
    return "".join(result)


def default_bundle() -> "_OnnxBundle | _Bundle":
    """Resolve the chosen weights once; an oracle setting cannot replace them."""
    from surah_cutter.core import model_store

    if DEFAULT_BACKEND == "onnx" or model_store.ACTIVE_ID != "quran-phonetics":
        return _get_onnx_bundle()
    return _get_bundle(DEFAULT_MODEL_ID)


def _get_onnx_bundle(model_path: str | None = None) -> "_OnnxBundle":
    """Load (and cache) the int8 ONNX emitter wrapped as an ``_OnnxBundle``.

    ``model_path`` defaults to :data:`_DEFAULT_ONNX_MODEL` (overridable via the
    ``SURAH_CUTTER_ONNX_MODEL`` env var). Raises ``FileNotFoundError`` with a
    pointer to the export tool when the artefact is missing.
    """
    from surah_cutter.core import model_store

    selected = model_store.runtime_path(model_store.ACTIVE_ID) if model_path is None else None
    if model_path is None and model_store.ACTIVE_ID != "quran-phonetics" and selected is None:
        raise ValueError("Выбранная модель не установлена или не подключена к разметке.")
    path = model_path or str(selected or _DEFAULT_ONNX_MODEL)
    cached = _ONNX_CACHE.get(path)
    if cached is not None:
        process = getattr(cached.emitter, "process", None)
        if process is None or process.poll() is None:
            return cached
        del _ONNX_CACHE[path]

    import os as _os

    if not _os.path.exists(path):
        raise FileNotFoundError(
            f"ONNX alignment model not found at {path!r}. Export it first:\n"
            "    python -m surah_cutter.tools.export_onnx"
        )

    if model_path is None:
        emitter = model_store.load_emitter(model_store.ACTIVE_ID)
    else:
        from surah_cutter.core import emissions as _emissions_mod

        emitter = _emissions_mod.load_emitter(path)
    bundle = _OnnxBundle(emitter)
    if model_path is None and model_store.ACTIVE_ID == "arabic-ctc":
        bundle.text_kind = "arabic"
        bundle.lag_ms = 0
    _ONNX_CACHE[path] = bundle
    return bundle


def bundle_source(bundle: "_OnnxBundle | _Bundle") -> str:
    """Record the backend and revision that supplied the actual emissions."""
    if isinstance(bundle, _Bundle):
        config = getattr(bundle.model, "config", None)
        model_id = getattr(config, "_name_or_path", None) or DEFAULT_MODEL_ID
        revision = getattr(config, "_commit_hash", None) or "unknown"
        return f"torch-ctc:{model_id}:revision={revision}:lag={ALIGN_LAG_MS}"
    from surah_cutter.core import model_store

    return model_store.source()


def _get_bundle(model_id: str) -> "_Bundle":
    """Load (and cache) the model + processor + vocab for ``model_id``."""
    revision = os.environ.get("SURAH_CUTTER_TORCH_REVISION")
    key = f"{model_id}:{revision}" if revision else model_id
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_id, revision=revision)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, revision=revision)
    model.eval()
    torch.set_grad_enabled(False)

    vocab = _resolve_vocab(processor)
    bundle = _Bundle(model, processor, vocab)
    _BUNDLE_CACHE[key] = bundle
    return bundle


def _resolve_vocab(processor: Any) -> dict[str, int]:
    """Extract the ``token -> id`` vocab from a Wav2Vec2 processor/tokenizer."""
    tokenizer = getattr(processor, "tokenizer", processor)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return dict(get_vocab())
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, dict):
        return dict(vocab)
    raise RuntimeError("Could not resolve the model vocabulary from the processor")


def _resolve_blank_id(processor: Any, vocab: dict[str, int]) -> int:
    """CTC blank id: the tokenizer's PAD id (Wav2Vec2 convention)."""
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_id, int):
        return pad_id
    pad_tok = getattr(tokenizer, "pad_token", None)
    if pad_tok is not None and pad_tok in vocab:
        return vocab[pad_tok]
    # Fallback to the common Wav2Vec2 PAD symbol or id 0.
    for cand in ("<pad>", "[PAD]", "PAD"):
        if cand in vocab:
            return vocab[cand]
    return 0
