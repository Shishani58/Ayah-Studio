"""Fetch a checksum-pinned runtime-model ZIP for CI (four files at ZIP root)."""

import hashlib
import io
import os
import urllib.request
import zipfile
from pathlib import Path

NAMES = (
    "wav2vec2-quran-phonetics.onnx",
    "wav2vec2-quran-phonetics.onnx.vocab.json",
    "wav2vec2-quran-phonetics.onnx.meta.json",
    "silero_vad.onnx",
)


def main() -> None:
    url = os.environ.get("AYAH_MODELS_URL", "")
    checksum = os.environ.get("AYAH_MODELS_SHA256", "")
    if not url.startswith("https://") or len(checksum) != 64:
        raise SystemExit("Set AYAH_MODELS_URL (HTTPS ZIP) and AYAH_MODELS_SHA256 in CI")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != checksum.lower():
        raise SystemExit("Runtime model archive checksum mismatch")
    destination = Path(__file__).resolve().parents[1] / "models"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        # Read exact filenames; never extract arbitrary archive paths.
        files = {name: archive.read(name) for name in NAMES}
    for name, content in files.items():
        (destination / name).write_bytes(content)


if __name__ == "__main__":
    main()
