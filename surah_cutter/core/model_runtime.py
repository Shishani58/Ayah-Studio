"""Isolated acoustic runtimes. The GUI never imports torch or NeMo."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np

ADAPTER_VERSION = "known-text-v1"
_INSTALL_LOCK = threading.Lock()
_WORKERS: dict[str, "RuntimeEmitter"] = {}
_PACKAGES = {
    "model-runtime": ["torch==2.8.0", "torchaudio==2.8.0", "transformers==4.57.3",
                      "uroman==1.3.1.1", "quran-transcript==0.6.4", "onnxruntime==1.23.2"],
    "nemo-runtime": ["torch==2.8.0", "torchaudio==2.8.0", "nemo_toolkit[asr]==2.5.0",
                     "numpy==1.26.4", "pyarrow==19.0.1", "datasets==3.6.0",
                     "pandas==2.2.3", "peft==0.17.1", "accelerate==1.10.1",
                     "pytorch-lightning==2.5.3", "torchmetrics==1.8.1",
                     "scipy==1.15.3", "numba==0.61.2"],
}


def environment(model_id: str) -> Path:
    from surah_cutter.core import model_store

    return model_store.ROOT.parent / ("nemo-runtime" if model_id.startswith("fastconformer") else "model-runtime")


def ensure_runtime(model_id: str, cancel: threading.Event | None = None) -> Path:
    """Provision a pinned Python 3.11 environment independently of app Python."""
    folder = environment(model_id)
    packages = _PACKAGES[folder.name]
    marker = folder / "ayah-runtime.json"
    python = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    fingerprint = hashlib.sha256(json.dumps(packages).encode()).hexdigest()
    with _INSTALL_LOCK:
        if marker.is_file() and python.is_file():
            if json.loads(marker.read_text()).get("requirements") == fingerprint:
                return python
        uv = shutil.which("uv")
        if not uv:
            import uv as uv_package

            uv = uv_package.find_uv_bin()
        folder.parent.mkdir(parents=True, exist_ok=True)
        commands = [[str(uv), "venv", str(folder), "--python", "3.11"],
                    [str(uv), "pip", "install", "--python", str(python), *packages]]
        # Logs are local diagnostics, never returned verbatim to the UI.
        with (folder.parent / f"{folder.name}-install.log").open("a") as log:
            for command in commands:
                if command[1] == "venv" and python.exists():
                    continue
                process = subprocess.Popen(command, stdout=log, stderr=log)
                while True:
                    try:
                        code = process.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if cancel is not None and cancel.is_set():
                            process.terminate()
                            process.wait()
                            raise InterruptedError
                if code:
                    raise ValueError("Не удалось подготовить окружение модели. Проверьте сеть и место на диске.")
        marker.write_text(json.dumps({"requirements": fingerprint, "packages": packages,
                                      "python": "3.11", "adapter": ADAPTER_VERSION}))
    return python


class RuntimeEmitter:
    """Serialized RPC to a persistent local process with one loaded model."""

    def __init__(self, model_id: str, folder: Path):
        self.model_id = model_id
        self.folder = folder
        self._lock = threading.RLock()
        self._temporary = tempfile.TemporaryDirectory(prefix="ayah-runtime-")
        self.work = Path(self._temporary.name)
        python = ensure_runtime(model_id)
        script = Path(__file__).with_name("model_worker.py")
        self._log = (environment(model_id).parent / f"{model_id}-runtime.log").open("a")
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                    "OMP_NUM_THREADS": "4"})
        self.process = subprocess.Popen(
            [str(python), "-u", str(script), model_id, str(folder), str(self.work)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, env=env,
        )
        try:
            metadata = self._receive()
            self.vocab = {k: int(v) for k, v in metadata["vocab"].items()}
            self.blank_id = metadata["blank_id"]
            self.text_kind = metadata["text_kind"]
            self.runtime_versions = metadata["versions"]
        except Exception:
            self.close()
            raise

    def _receive(self) -> dict:
        line = self.process.stdout.readline()
        if not line:
            raise ValueError("Процесс модели завершился. См. локальный журнал модели.")
        response = json.loads(line)
        if "error" in response:
            raise ValueError(response["error"])
        return response

    def request(self, action: str, **fields) -> dict:
        with self._lock:
            self.process.stdin.write(json.dumps({"action": action, **fields}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            return self._receive()

    def emissions(self, pcm, **_kwargs) -> np.ndarray:
        with self._lock:
            np.save(self.work / "audio.npy", np.ascontiguousarray(pcm, dtype=np.float32))
            self.request("emissions")
            return np.load(self.work / "emissions.npy", allow_pickle=False)

    def build_targets(self, words, intro: bool = False):
        response = self.request("targets", words=[w.text for w in words],
                                surah=words[0].surah, intro=intro)
        return response["targets"], [tuple(span) for span in response["spans"]]

    def align_words(self, pcm, words, intro: bool = False):
        with self._lock:
            np.save(self.work / "audio.npy", np.ascontiguousarray(pcm, dtype=np.float32))
            response = self.request("align", words=[w.text for w in words], intro=intro)
            rows = response["rows"]
            if len(rows) != len(words):
                raise ValueError("Неполное сопоставление канонических слов.")
            return [(w.index, int(row[0]), float(row[1])) for w, row in zip(words, rows)]

    def probe(self):
        if self.text_kind == "whisper":
            return self.request("probe")
        result = self.emissions(np.zeros(16000, dtype=np.float32))
        if result.ndim != 2 or result.shape[1] != len(self.vocab) or not np.isfinite(result).all():
            raise ValueError("Несовместимый выход модели.")

    def close(self):
        if getattr(self, "process", None) is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        self._log.close()
        self._temporary.cleanup()


def load_runtime(model_id: str, folder: Path) -> RuntimeEmitter:
    key = f"{model_id}:{folder}"
    cached = _WORKERS.get(key)
    if cached is not None and cached.process.poll() is None:
        return cached
    if cached is not None:
        cached.close()
    emitter = RuntimeEmitter(model_id, folder)
    _WORKERS[key] = emitter
    return emitter


def release_runtimes(keep=None):
    for key, emitter in list(_WORKERS.items()):
        if emitter is not keep:
            emitter.close()
            del _WORKERS[key]


atexit.register(release_runtimes)
