"""ONNX runtime path for the alignment model's emission matrix.

Produces the **same** log-softmax emission matrix as the torch path in
:mod:`surah_cutter.core.align` (``log_softmax(Wav2Vec2ForCTC(...).logits)``),
but via an int8 ONNX export run with :mod:`onnxruntime` instead of torch. This
lets forced alignment run without torch/transformers at inference time; the
forced-alignment math downstream is unchanged.

Parity contract with the torch path:

- **Vocab order**: the ONNX logits are emitted in the *exact* column order of
  the exported model's classifier, which is the Wav2Vec2 vocab order. The
  caller resolves token ids from the same vocab JSON exported alongside the
  model (see :mod:`surah_cutter.tools.export_onnx`), so column ``i`` means the
  same token in both paths. We do **not** append any ``<star>`` column.
- **Stride**: wav2vec2 emits one frame per ~20 ms; the true per-frame stride is
  recovered downstream from ``len(pcm) / n_frames`` exactly as the torch path
  does, so no stride assumption leaks in here.
- **Normalisation**: ``Wav2Vec2Processor`` for this model applies zero-mean /
  unit-variance normalisation over the *whole* clip. The torch path normalises
  the full waveform in one pass; to keep parity we normalise the full waveform
  **once, globally**, *before* windowing — not per-window — so window seams do
  not change the statistics.

Long audio is processed in overlapping windows with discarded context frames
(the same scheme as ``ctc_forced_aligner.generate_emissions``): pad with
``context`` seconds on each side, slide a ``window``-second frame with
``context`` of look-around, run each window, then trim the context frames back
off so the concatenation reconstructs the contiguous emission sequence.
Production inference processes one window per call: dynamic INT8 scales must
not depend on another window or a mostly padded tail in the same batch.

Heavy deps (:mod:`onnxruntime`) are imported lazily so ``core`` stays importable
without them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "EXPECTED_SR",
    "FRAME_STRIDE_MS",
    "OnnxEmitter",
    "load_emitter",
    "generate_emissions",
]

#: Sample rate the model expects (wav2vec2 family); matches ``core.decode``.
EXPECTED_SR = 16_000

#: Nominal wav2vec2 frame stride in milliseconds. Used only to convert the
#: context/window seconds into a count of frames to trim; the *real* stride used
#: for ms conversion is recovered from ``len(pcm)/n_frames`` downstream (parity
#: with the torch path's :func:`surah_cutter.core.align._stride_ms`).
FRAME_STRIDE_MS = 20

#: Default window / context lengths (seconds), mirroring
#: ``ctc_forced_aligner.generate_emissions`` defaults.
DEFAULT_WINDOW_S = 30
DEFAULT_CONTEXT_S = 2

# Dynamic INT8 activation scales depend on all samples in a batch. Feeding a
# mostly padded tail alongside a speech window can move an ayah onset by ~800 ms
# (Shatri 114). Keep production inference independent of neighbouring windows.
DEFAULT_BATCH_SIZE = 1


def _time_to_frame(seconds: float) -> int:
    """Number of emission frames spanning ``seconds`` of audio (~20 ms/frame)."""
    return int(seconds * (1000.0 / FRAME_STRIDE_MS))


class OnnxEmitter:
    """A loaded int8 ONNX model + its vocab, producing log-softmax emissions.

    Parity surrogate for the torch ``_Bundle`` in :mod:`surah_cutter.core.align`:
    exposes the same ``vocab`` mapping and a ``blank_id`` so the alignment code
    can build the identical target id sequence regardless of backend.
    """

    __slots__ = (
        "session",
        "vocab",
        "blank_id",
        "_input_name",
        "_output_name",
        "_do_normalize",
        "_loaded_runtime_identity",
    )

    def __init__(
        self,
        session: Any,
        vocab: dict[str, int],
        *,
        blank_id: int,
        do_normalize: bool = True,
        weights_sha256: str | None = None,
    ) -> None:
        self.session = session
        self.vocab = vocab
        self.blank_id = blank_id
        self._do_normalize = do_normalize
        # ONNX exports name the audio input "input_values" and the head "logits"
        # (see export_onnx.py); resolve defensively from the graph regardless.
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        import json
        from importlib import metadata

        options = session.get_session_options() if hasattr(session, "get_session_options") else None
        self._loaded_runtime_identity = json.dumps({
            "weights_sha256": weights_sha256, "vocab": dict(vocab), "blank_id": blank_id,
            "do_normalize": do_normalize, "input_name": self._input_name,
            "output_name": self._output_name, "providers": session.get_providers() if hasattr(session, "get_providers") else [],
            "onnxruntime": metadata.version("onnxruntime"), "numpy": np.__version__,
            "session_options": {name: str(getattr(options, name)) for name in
                ("graph_optimization_level", "execution_mode", "intra_op_num_threads", "inter_op_num_threads")}
                if options is not None else {},
        }, sort_keys=True, allow_nan=False)

    @property
    def runtime_identity(self):
        import json

        return json.loads(self._loaded_runtime_identity)

    def emissions(
        self,
        pcm: np.ndarray,
        *,
        window_s: int = DEFAULT_WINDOW_S,
        context_s: int = DEFAULT_CONTEXT_S,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Return log-softmax emissions of shape ``(T, V)`` for ``pcm``.

        ``pcm`` is mono float32 at :data:`EXPECTED_SR`. The result matches the
        torch path: ``log_softmax`` over the vocab, one row per ~20 ms frame, no
        extra ``<star>`` column.
        """
        return generate_emissions(
            self.session,
            np.ascontiguousarray(pcm, dtype=np.float32),
            input_name=self._input_name,
            output_name=self._output_name,
            window_s=window_s,
            context_s=context_s,
            batch_size=batch_size,
            do_normalize=self._do_normalize,
        )


def load_emitter(model_path: str | Path, *, vocab_path: str | Path | None = None) -> OnnxEmitter:
    """Load an int8 ONNX model and its vocab into an :class:`OnnxEmitter`.

    Args:
        model_path: Path to the (quantised) ``.onnx`` file exported by
            :mod:`surah_cutter.tools.export_onnx`.
        vocab_path: Path to the JSON ``token -> id`` vocab written next to the
            model. Defaults to ``<model_path>.vocab.json`` then ``vocab.json``
            in the model's directory.

    Lazy: imports :mod:`onnxruntime` only here.
    """
    import onnxruntime as ort  # lazy heavy dep

    model_path = Path(model_path)
    import hashlib

    before = model_path.stat()
    weights_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    vocab, do_normalize = _load_sidecar(model_path, vocab_path)
    blank_id = _resolve_blank_id(vocab)

    # CPU-only, single deterministic provider (parity over throughput).
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    if model_path.stat() != before:
        raise ValueError("Model weights changed during loading")
    return OnnxEmitter(session, vocab, blank_id=blank_id, do_normalize=do_normalize,
                       weights_sha256=weights_sha256)


def generate_emissions(
    session: Any,
    audio_waveform: np.ndarray,
    *,
    input_name: str = "input_values",
    output_name: str = "logits",
    window_s: int = DEFAULT_WINDOW_S,
    context_s: int = DEFAULT_CONTEXT_S,
    batch_size: int = DEFAULT_BATCH_SIZE,
    do_normalize: bool = True,
) -> np.ndarray:
    """Windowed ONNX inference -> ``(T, V)`` log-softmax emissions.

    Mirrors ``ctc_forced_aligner.generate_emissions`` windowing (pad with
    ``context`` seconds, slide ``window``-second frames with context look-around,
    trim context frames after inference), but:

    - normalises the **whole** waveform once before windowing (parity with the
      torch ``Wav2Vec2Processor`` single-pass), when ``do_normalize`` is set;
    - returns the bare ``(T, V)`` log-softmax — **no** ``<star>`` column — so the
      vocab columns line up 1:1 with the torch path.
    """
    if audio_waveform.ndim != 1:
        raise ValueError("audio_waveform must be a 1-D mono array")
    if window_s <= 0 or context_s < 0 or batch_size <= 0:
        raise ValueError("window and batch must be positive; context must be nonnegative")
    if audio_waveform.size == 0:
        raise ValueError("audio_waveform must contain samples")

    from surah_cutter.core.progress import advance

    wav = np.ascontiguousarray(audio_waveform, dtype=np.float32)
    if do_normalize:
        wav = _normalize(wav)

    context = context_s * EXPECTED_SR
    window = window_s * EXPECTED_SR
    n = wav.shape[0]

    # Short clip: a single forward pass, no context trimming needed. This also
    # matches the torch path's behaviour for typical per-sura recitations.
    if n <= window:
        advance("acoustic", 0, 1)
        logits = _run(session, wav[None, :], input_name, output_name)[0]
        advance("acoustic", 1, 1)
        return _log_softmax(logits)

    # Pad so the signal length is a whole number of windows, plus ``context`` of
    # look-around on each side.
    extension = math.ceil(n / window) * window - n
    padded = np.pad(wav, (context, context + extension), mode="constant")

    num_windows = (padded.shape[0] - 2 * context) // window
    ctx_frames = _time_to_frame(context_s)

    advance("acoustic", 0, num_windows)
    emissions_chunks: list[np.ndarray] = []
    for i in range(0, num_windows, batch_size):
        batch = np.stack([
            padded[j * window:j * window + window + 2 * context]
            for j in range(i, min(num_windows, i + batch_size))
        ])
        logits = _run(session, batch, input_name, output_name)  # (B, Tw, V)
        # Trim the context frames off each window so the kept frames are exactly
        # the ``window``-second core; the ``+ 1`` keeps the seam aligned with the
        # reference implementation's framing.
        trimmed = logits[:, ctx_frames:-ctx_frames + 1, :] if ctx_frames else logits
        emissions_chunks.append(trimmed)
        advance("acoustic", min(num_windows, i + batch_size), num_windows)

    emissions = np.concatenate(emissions_chunks, axis=0)  # (num_windows, Tc, V)
    emissions = emissions.reshape(-1, emissions.shape[-1])  # (T_padded, V)

    # Drop the frames corresponding to the tail padding we added.
    ext_frames = _time_to_frame(extension / EXPECTED_SR)
    if ext_frames > 0:
        emissions = emissions[:-ext_frames, :]

    return _log_softmax(emissions)


# --- helpers --------------------------------------------------------------


def _run(session: Any, batch: np.ndarray, input_name: str, output_name: str) -> np.ndarray:
    """Run one ONNX forward pass; return logits ``(B, T, V)`` as float32."""
    inputs = {input_name: np.ascontiguousarray(batch, dtype=np.float32)}
    from surah_cutter.core.cancellation import current_event, interruptible

    if current_event() is None:
        outputs = session.run([output_name], inputs)
    else:
        import onnxruntime as ort

        options = ort.RunOptions()
        with interruptible(lambda: setattr(options, "terminate", True)):
            outputs = session.run([output_name], inputs, options)
    return np.asarray(outputs[0], dtype=np.float32)


def _normalize(wav: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance over the whole clip (Wav2Vec2Processor parity).

    Uses the population standard deviation with a small epsilon, matching the
    ``(x - mean) / sqrt(var + 1e-7)`` the HF feature extractor applies.
    """
    mean = wav.mean()
    var = wav.var()
    return ((wav - mean) / np.sqrt(var + 1e-7)).astype(np.float32)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis -> float32 ``(T, V)``."""
    x = logits.astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    out = x - np.log(exp.sum(axis=-1, keepdims=True))
    return out.astype(np.float32)


def _load_sidecar(
    model_path: Path,
    vocab_path: str | Path | None,
) -> tuple[dict[str, int], bool]:
    """Load the vocab JSON and the ``do_normalize`` flag exported with the model.

    The export writes ``<model>.vocab.json`` (``token -> id``) and an optional
    ``<model>.meta.json`` carrying ``do_normalize``. Falls back to ``vocab.json``
    in the model directory and ``do_normalize=True`` (the model's default).
    """
    if vocab_path is not None:
        vp = Path(vocab_path)
    else:
        cand = model_path.with_suffix(model_path.suffix + ".vocab.json")
        vp = cand if cand.exists() else model_path.parent / "vocab.json"
    if not vp.exists():
        raise FileNotFoundError(
            f"Vocab JSON not found for ONNX model {model_path!r}; expected "
            f"{vp!r}. Re-run the export tool to write it."
        )
    with vp.open("r", encoding="utf-8") as fh:
        vocab = {str(k): int(v) for k, v in json.load(fh).items()}

    do_normalize = True
    meta = model_path.with_suffix(model_path.suffix + ".meta.json")
    if meta.exists():
        try:
            with meta.open("r", encoding="utf-8") as fh:
                do_normalize = bool(json.load(fh).get("do_normalize", True))
        except (json.JSONDecodeError, OSError, ValueError):
            do_normalize = True
    return vocab, do_normalize


def _resolve_blank_id(vocab: dict[str, int]) -> int:
    """CTC blank id: the PAD symbol's id (Wav2Vec2 convention), else 0."""
    for cand in ("<pad>", "[PAD]", "PAD"):
        if cand in vocab:
            return vocab[cand]
    return 0
