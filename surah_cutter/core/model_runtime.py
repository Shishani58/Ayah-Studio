"""Isolated acoustic runtimes. The GUI never imports torch or NeMo."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache

import numpy as np

ADAPTER_VERSION = "known-text-v2"
RPC_TIMEOUT_S = 300.0
INFERENCE_TIMEOUT_S = 1800.0
INSTALL_TIMEOUT_S = 1800.0
_INSTALL_LOCK = threading.Lock()
_WORKERS: dict[str, "RuntimeEmitter"] = {}
_PACKAGES = {
    "model-runtime": ["torch==2.8.0", "torchaudio==2.8.0", "transformers==4.57.3",
                      "uroman==1.3.1.1", "quran-transcript==0.6.4", "onnxruntime==1.23.2"],
    "nemo-runtime": ["torch==2.8.0", "torchaudio==2.8.0", "nemo_toolkit[asr]==2.5.0",
                     "numpy==1.26.4", "pyarrow==19.0.1", "datasets==3.6.0",
                     "pandas==2.2.3", "peft==0.17.1", "accelerate==1.10.1",
                     "pytorch-lightning==2.5.3", "torchmetrics==1.8.1",
                     "scipy==1.13.1", "numba==0.61.2"],
}


@lru_cache(maxsize=1)
def _text_targets():
    from surah_cutter.data.quran_text import DEFAULT_DB_PATH

    path = DEFAULT_DB_PATH.with_name("model_text_targets.json")
    data = json.loads(path.read_text())
    if hashlib.sha256(DEFAULT_DB_PATH.read_bytes()).hexdigest() != data["canonical_sha256"]:
        raise ValueError("Словарь модели не соответствует каноническому Корану.")
    return data["surahs"]


def inference_words(words, model_id):
    if model_id == "mms-fa":
        return [w.text for w in words]
    table = _text_targets()[str(words[0].surah)]
    result = []
    for word in words:
        if table[word.index][0] != word.text:
            raise ValueError("Порядок слов модели не соответствует каноническому тексту.")
        text = table[word.index][1 if model_id == "muaalem" else 2]
        if ((model_id.startswith("whisper-") and model_id not in
             ("whisper-quran", "whisper-tarteel-tiny", "whisper-tarteel-base"))
                or model_id == "fastconformer-arabic"):
            import unicodedata

            text = "".join(c for c in text if unicodedata.category(c) not in ("Mn", "Me"))
        result.append(text)
    return result


def environment(model_id: str) -> Path:
    from surah_cutter.core import model_store

    return model_store.ROOT.parent / ("nemo-runtime" if model_id.startswith("fastconformer") else "model-runtime")


def ensure_runtime(model_id: str, cancel: threading.Event | None = None) -> Path:
    """Provision a pinned Python 3.11 environment independently of app Python."""
    folder = environment(model_id)
    packages = _PACKAGES[folder.name]
    marker = folder / "ayah-runtime.json"
    python = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    lock = Path(__file__).with_name("runtime_locks") / f"{folder.name}.txt"
    fingerprint = hashlib.sha256(lock.read_bytes()).hexdigest()
    with _INSTALL_LOCK:
        if marker.is_file() and python.is_file():
            try:
                if json.loads(marker.read_text()).get("requirements") == fingerprint:
                    return python
            except (ValueError, OSError):
                pass  # An incomplete receipt cannot certify an environment.
        bundled_uv = Path(__file__).resolve().parents[2] / "uv-bin" / ("uv.exe" if os.name == "nt" else "uv")
        uv = str(bundled_uv) if bundled_uv.is_file() else shutil.which("uv")
        if not uv:
            import uv as uv_package

            uv = uv_package.find_uv_bin()
        folder.parent.mkdir(parents=True, exist_ok=True)
        commands = [[str(uv), "venv", str(folder), "--python", "3.11"],
                    [str(uv), "pip", "install", "--python", str(python),
                     "--require-hashes", "-r", str(lock)]]
        # Logs are local diagnostics, never returned verbatim to the UI.
        with (folder.parent / f"{folder.name}-install.log").open("a") as log:
            for command in commands:
                if command[1] == "venv" and python.exists():
                    continue
                from time import monotonic

                if cancel is not None and cancel.is_set():
                    raise InterruptedError
                process = subprocess.Popen(command, stdout=log, stderr=log)
                deadline = monotonic() + INSTALL_TIMEOUT_S
                try:
                    while True:
                        try:
                            code = process.wait(timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            if cancel is not None and cancel.is_set():
                                raise InterruptedError
                            if monotonic() >= deadline:
                                raise ValueError("Подготовка модели превысила время ожидания.")
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
                if code:
                    raise ValueError("Не удалось подготовить окружение модели. Проверьте сеть и место на диске.")
        installed = subprocess.check_output(
            [str(python), "-c", "import importlib.metadata,json; print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()},sort_keys=True))"],
            text=True, timeout=RPC_TIMEOUT_S,
        )
        marker.write_text(json.dumps({"requirements": fingerprint, "packages": packages,
                                      "versions": json.loads(installed),
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
        for name in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(name, None)
        if "LD_LIBRARY_PATH_ORIG" in env:
            env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH_ORIG"]
        elif getattr(sys, "frozen", False):
            env.pop("LD_LIBRARY_PATH", None)
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
            self._loaded_runtime_identity = json.dumps(metadata["runtime_identity"], sort_keys=True, allow_nan=False)
        except BaseException:
            self.close()
            raise

    @property
    def runtime_identity(self):
        """A fresh copy of the immutable identity returned by the loaded worker."""
        return json.loads(self._loaded_runtime_identity)

    @contextmanager
    def _deadline(self, timeout: float):
        """Bound pipe writes and reads, and retire the watchdog before reuse."""
        from surah_cutter.core.cancellation import interruptible, check_cancelled

        expired = threading.Event()

        def abort():
            try:
                self.process.kill()
            except ProcessLookupError:
                pass

        def timed_out():
            expired.set()
            abort()

        timer = threading.Timer(timeout, timed_out)
        timer.daemon = True
        timer.start()
        try:
            with interruptible(abort):
                try:
                    yield
                finally:
                    timer.cancel()
                    timer.join()
                    if expired.is_set():
                        check_cancelled()
                        raise ValueError("Модель превысила время ожидания. Повторите расчёт.")
        finally:
            timer.cancel()
            timer.join()

    def _receive(self, timeout: float | None = None) -> dict:
        with self._deadline(RPC_TIMEOUT_S if timeout is None else timeout):
            return self._read_response()

    def _read_response(self) -> dict:
        from surah_cutter.core.progress import advance

        while True:
            line = self.process.stdout.readline()
            if not line:
                raise ValueError("Процесс модели завершился. См. локальный журнал модели.")
            response = json.loads(line)
            if "progress" in response:
                progress = response["progress"]
                advance("acoustic", progress["done"], progress["total"])
                continue
            if "error" in response:
                raise ValueError(response["error"])
            return response

    def request(self, action: str, **fields) -> dict:
        from surah_cutter.core.progress import current_stage

        if action in ("align", "emissions") and current_stage() == "acoustic":
            fields["report_progress"] = True
        with self._lock:
            timeout = INFERENCE_TIMEOUT_S if action in ("align", "emissions") else RPC_TIMEOUT_S
            with self._deadline(timeout):
                try:
                    self.process.stdin.write(json.dumps({"action": action, **fields}, ensure_ascii=False) + "\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    raise ValueError("Процесс модели завершился. Повторите расчёт.") from exc
                return self._read_response()

    def emissions(self, pcm, **_kwargs) -> np.ndarray:
        with self._lock:
            np.save(self.work / "audio.npy", np.ascontiguousarray(pcm, dtype=np.float32))
            self.request("emissions")
            return np.load(self.work / "emissions.npy", allow_pickle=False)

    def build_targets(self, words, intro: bool = False):
        response = self.request("targets", words=[w.text for w in words],
                                surah=words[0].surah, intro=intro,
                                renderings=inference_words(words, self.model_id),
                                ayahs=[w.ayah for w in words])
        return response["targets"], [tuple(span) for span in response["spans"]]

    def align_words(self, pcm, words, intro: bool = False, strict_words: bool = True):
        with self._lock:
            np.save(self.work / "audio.npy", np.ascontiguousarray(pcm, dtype=np.float32))
            response = self.request("align", words=[w.text for w in words], intro=intro,
                                    renderings=inference_words(words, self.model_id), strict_words=strict_words)
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

    def recognize_local(self, pcm, *, total_samples):
        if (self.model_id not in ("whisper-tarteel-tiny", "whisper-tarteel-base") or np.ndim(pcm) != 1
                or not 4800 <= len(pcm) <= 96000 or len(pcm) >= total_samples):
            raise ValueError("FAST Whisper требует локальный участок до 6 секунд.")
        with self._lock:
            np.save(self.work / "audio.npy", np.ascontiguousarray(pcm, dtype=np.float32))
            return self.request("recognize_local", total_samples=total_samples)

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
        if getattr(self, "_log", None) is not None:
            self._log.close()
        if getattr(self, "_temporary", None) is not None:
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
