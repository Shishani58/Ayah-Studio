"""Export the alignment acoustic model to int8 ONNX for the runtime path.

Exports ``TBOGamer22/wav2vec2-quran-phonetics`` (a ``Wav2Vec2ForCTC``) to ONNX,
then applies **dynamic int8 quantisation** via
:mod:`onnxruntime.quantization`. The quantised model is what
:mod:`surah_cutter.core.emissions` loads at inference time, so the heavy
torch/transformers stack is only needed *here*, at export, never at runtime.

What is written (under ``models/``, default ``models/wav2vec2-quran-phonetics``
basename):

- ``<out>.fp32.onnx`` — the raw float32 export (kept for debugging/parity).
- ``<out>.onnx`` — the int8 dynamically-quantised model (the runtime artefact).
- ``<out>.onnx.vocab.json`` — the tokenizer ``token -> id`` vocab, so the
  emissions loader resolves the *same* column order the torch path uses.
- ``<out>.onnx.meta.json`` — small metadata (model id, ``do_normalize`` flag).

The exported graph has a single dynamic-length audio input named
``input_values`` (shape ``[batch, samples]``) and a single output named
``logits`` (shape ``[batch, frames, vocab]``) — the same tensors the windowed
ONNX runner in :mod:`emissions` expects.

Heavy deps are imported lazily inside :func:`export`; the module imports without
torch so it can be referenced/inspected freely.

CLI::

    python -m surah_cutter.tools.export_onnx \
        --model-id TBOGamer22/wav2vec2-quran-phonetics \
        --out models/wav2vec2-quran-phonetics

This module is a dev tool; it is **not** invoked by the runtime pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Default acoustic model to export (kept in sync with
#: :data:`surah_cutter.core.align.DEFAULT_MODEL_ID`).
DEFAULT_MODEL_ID = "TBOGamer22/wav2vec2-quran-phonetics"

#: Default output basename under ``models/``. ``.fp32.onnx`` / ``.onnx`` /
#: ``.onnx.vocab.json`` / ``.onnx.meta.json`` are appended/derived from it.
#: export_onnx.py -> tools/ -> surah_cutter/ -> <repo>/
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "models" / "wav2vec2-quran-phonetics"

#: ONNX opset for the export. 14 covers the wav2vec2 conv/attention stack and is
#: supported by the installed onnxruntime.
OPSET = 14

#: A representative number of samples for the dummy export input (~1.6 s at
#: 16 kHz). Dynamic axes make the actual length free at runtime.
_DUMMY_SAMPLES = 16_000


def export(
    model_id: str = DEFAULT_MODEL_ID,
    out: str | Path = DEFAULT_OUT,
    *,
    opset: int = OPSET,
) -> Path:
    """Export ``model_id`` to fp32 ONNX, quantise to int8, write sidecars.

    Returns the path to the int8 ``.onnx`` artefact (the runtime model). Heavy
    deps (torch, transformers, onnxruntime.quantization) are imported here.
    """
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = out.with_suffix(out.suffix + ".fp32.onnx")
    int8_path = out.with_suffix(out.suffix + ".onnx")
    vocab_path = int8_path.with_suffix(int8_path.suffix + ".vocab.json")
    meta_path = int8_path.with_suffix(int8_path.suffix + ".meta.json")

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    torch.set_grad_enabled(False)

    # --- 1. fp32 ONNX export -------------------------------------------------
    wrapped = _LogitsOnly(model)
    dummy = torch.zeros(1, _DUMMY_SAMPLES, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(fp32_path),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "samples"},
            "logits": {0: "batch", 1: "frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # --- 2. dynamic int8 quantisation ---------------------------------------
    # Pre-process the graph (shape inference + optimisation) then quantise only
    # MatMul ops: those are the transformer attention/FFN weights — the bulk of
    # the 94M params. The small conv feature-encoder is left fp32 because its
    # dynamic-quantiser path crashes on this graph ("expected initializer"), and
    # it is a negligible fraction of the size.
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.preprocess import quant_pre_process

    prep_path = out.with_suffix(out.suffix + ".prep.onnx")
    # skip_symbolic_shape: ORT's symbolic shape inference crashes on this graph's
    # Range node (_infer_Range AssertionError); MatMul-only dynamic quant doesn't
    # need symbolic shapes anyway.
    quant_pre_process(
        str(fp32_path), str(prep_path), skip_symbolic_shape=True
    )
    quantize_dynamic(
        model_input=str(prep_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    prep_path.unlink(missing_ok=True)

    # --- 3. sidecars: vocab order + normalisation flag -----------------------
    vocab = _resolve_vocab(processor)
    with vocab_path.open("w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)

    feat = getattr(processor, "feature_extractor", None)
    do_normalize = bool(getattr(feat, "do_normalize", True))
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"model_id": model_id, "do_normalize": do_normalize, "opset": opset},
            fh,
            ensure_ascii=False,
        )

    return int8_path


class _LogitsOnly:
    """Wrap ``Wav2Vec2ForCTC`` so ONNX export sees a single ``logits`` tensor.

    The HF model returns a ``CausalLMOutput``; ONNX export wants plain tensors.
    Defined as a thin ``torch.nn.Module`` subclass *lazily* (torch imported by
    the caller) to keep this file importable without torch.
    """

    def __new__(cls, model):  # noqa: D102 - construct the real nn.Module
        import torch.nn as nn

        class _Mod(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_values):
                return self.m(input_values).logits

        return _Mod(model)


def _resolve_vocab(processor) -> dict[str, int]:
    """Extract ``token -> id`` from a Wav2Vec2 processor/tokenizer (parity w/ align)."""
    tokenizer = getattr(processor, "tokenizer", processor)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return {str(k): int(v) for k, v in get_vocab().items()}
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, dict):
        return {str(k): int(v) for k, v in vocab.items()}
    raise RuntimeError("Could not resolve the model vocabulary from the processor")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, run :func:`export`, print the artefact path."""
    parser = argparse.ArgumentParser(
        prog="python -m surah_cutter.tools.export_onnx",
        description="Export the wav2vec2 alignment model to int8 ONNX.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HF model id to export (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output basename under models/ (default: %(default)s). "
        "'.onnx' (int8), '.fp32.onnx', and sidecar JSONs derive from it.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=OPSET,
        help=f"ONNX opset version (default: {OPSET}).",
    )
    args = parser.parse_args(argv)

    int8_path = export(args.model_id, args.out, opset=args.opset)
    print(int8_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    raise SystemExit(main())
