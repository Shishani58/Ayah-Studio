"""Developer tools for surah-cutter (model export, cache warming, etc.).

These are *offline prep* utilities, not part of the runtime path. They pull in
heavy dev-only deps (torch, transformers, onnxruntime.quantization) lazily, so
importing this package does not drag those into ``core``/``data``/GUI.
"""
