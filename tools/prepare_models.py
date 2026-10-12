"""Build runtime models from immutable upstream revisions before packaging.

Run from the repository root: python -m tools.prepare_models.
Install tools/model-export-requirements.txt first. Export in a temporary directory
so a failed download/export does not replace existing runtime models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

from surah_cutter.tools.export_onnx import export

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "wav2vec2-quran-phonetics.onnx",
    "wav2vec2-quran-phonetics.onnx.vocab.json",
    "wav2vec2-quran-phonetics.onnx.meta.json",
    "silero_vad.onnx",
)


def download_checked(url: str, destination: Path, expected: str) -> None:
    """Reject corrupt or substituted weights before they reach the output directory."""
    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != expected:
        destination.unlink()
        raise ValueError(f"SHA256 mismatch for {url}: expected {expected}, got {actual}")


def prepare(output: Path, manifest: Path = ROOT / "tools/model_sources.json") -> None:
    sources = json.loads(manifest.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="ayah-models-") as temporary:
        stage = Path(temporary)
        silero = sources["silero"]
        download_checked(silero["url"], stage / "silero_vad.onnx", silero["sha256"])
        acoustic = sources["acoustic"]
        export(
            acoustic["model_id"], stage / "wav2vec2-quran-phonetics",
            revision=acoustic["revision"], opset=acoustic["opset"],
        )
        for name in RUNTIME_FILES:
            if not (stage / name).is_file():
                raise FileNotFoundError(f"Export did not produce {name}")
        output.mkdir(parents=True, exist_ok=True)
        for name in RUNTIME_FILES:
            shutil.copy2(stage / name, output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
