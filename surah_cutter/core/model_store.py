"""Curated, revision-pinned model downloads; no executable Hub code is loaded."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path.home() / ".surah-cutter" / "models"
_LOCK = threading.RLock()
RUNTIME_LOCK = threading.RLock()
_JOBS: dict[str, dict] = {}
_CANCEL: dict[str, threading.Event] = {}
ACTIVE_ID = "quran-phonetics"
_CATALOG_PATH = Path(__file__).with_name("model_catalog.json")


def catalog() -> list[dict]:
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))


def available_catalog() -> list[dict]:
    """User shortlist; retain archived specs for reproducible benchmarks."""
    return [item for item in catalog() if not item.get("hidden", False)]


def spec(model_id: str) -> dict:
    for item in catalog():
        if item["id"] == model_id:
            return item
    raise ValueError("Модель отсутствует в каталоге.")


def installed_path(item: dict) -> Path:
    return ROOT / item["id"] / item["revision"]


def is_installed(item: dict) -> bool:
    folder = installed_path(item)
    return (folder / "installation.json").is_file() and all(
        (folder / f["path"]).is_file() and (folder / f["path"]).stat().st_size == f["size"]
        for f in item["files"]
    )


def runtime_path(model_id: str) -> Path | None:
    from surah_cutter.core import align

    if model_id == "quran-phonetics":
        path = Path(align._DEFAULT_ONNX_MODEL)
        return path if path.is_file() else None
    if model_id == "quran-phonetics-fp32":
        path = Path(align._DEFAULT_ONNX_MODEL).with_name("wav2vec2-quran-phonetics.fp32.onnx")
        return path if path.is_file() else None
    item = spec(model_id)
    if model_id == "arabic-ctc" and is_installed(item):
        return installed_path(item) / "wav2vec2-ar-ctc-aligner/model.int8.onnx"
    if is_installed(item):
        if model_id == "muaalem":
            return installed_path(item) / "muaalem/model.fp32.onnx"
        if model_id == "mms-fa":
            return installed_path(item) / "model-fp16.safetensors"
        if model_id.startswith("fastconformer"):
            return installed_path(item) / next(f["path"] for f in item["files"] if f["path"].endswith(".nemo"))
        if model_id.startswith("whisper-") or model_id == "mms-arabic":
            return installed_path(item) / "model.safetensors"
    return None


def active_kind() -> str:
    if ACTIVE_ID in ("quran-phonetics", "quran-phonetics-fp32"):
        return "phonetic"
    return "whisper" if ACTIVE_ID.startswith("whisper-") else "arabic"


def lag_ms() -> int:
    from surah_cutter.core import align

    return align.ALIGN_LAG_MS if active_kind() == "phonetic" else 0


def source() -> str:
    if ACTIVE_ID in ("quran-phonetics", "quran-phonetics-fp32"):
        revision = "local-export"
    else:
        revision = spec(ACTIVE_ID)["revision"]
    if ACTIVE_ID in ("quran-phonetics", "quran-phonetics-fp32", "arabic-ctc"):
        return f"onnx-ctc:{ACTIVE_ID}:revision={revision}:lag={lag_ms()}"
    from surah_cutter.core.model_runtime import ADAPTER_VERSION, _WORKERS

    versions = next((e.runtime_versions for e in _WORKERS.values() if e.model_id == ACTIVE_ID), {})
    method = "attention-dtw" if active_kind() == "whisper" else "ctc"
    text_cache = Path(__file__).resolve().parents[2] / "data/model_text_targets.json"
    text_sha = hashlib.sha256(text_cache.read_bytes()).hexdigest() if text_cache.is_file() else "missing"
    flags = "hafs:madd=4:waqf=ayah" if ACTIVE_ID == "muaalem" else "window=30:context=2:batch=1"
    if active_kind() == "whisper":
        flags = "window=30:commit-before=24:context=1word:teacher-forcing:language=ar"
    if ACTIVE_ID in ("fastconformer-arabic", "mms-arabic") or (active_kind() == "whisper" and ACTIVE_ID != "whisper-quran"):
        flags += ":target=imlaey-plain"
    if ACTIVE_ID.startswith("fastconformer"):
        flags += ":bpe=exact-spelling-variants"
    return (f"{method}:{ACTIVE_ID}:revision={revision}:adapter={ADAPTER_VERSION}:lag=0:"
            f"text={text_sha}:flags={flags}:"
            f"runtime={json.dumps(versions, sort_keys=True, separators=(',', ':'))}")


def load_emitter(model_id: str):
    """Load the selected weights, never substitute another acoustic model."""
    from surah_cutter.core import align, emissions

    path = runtime_path(model_id)
    if path is None:
        raise ValueError("Сначала скачайте модель.")
    if model_id in ("quran-phonetics", "quran-phonetics-fp32", "arabic-ctc"):
        vocab = str(align._DEFAULT_ONNX_MODEL) + ".vocab.json" if model_id == "quran-phonetics-fp32" else None
        return emissions.load_emitter(path, vocab_path=vocab)
    from surah_cutter.core.model_runtime import load_runtime

    return load_runtime(model_id, installed_path(spec(model_id)))


def listing() -> dict:
    items = []
    with _LOCK:
        for item in available_catalog():
            installed = is_installed(item)
            ready = runtime_path(item["id"]) is not None
            job = dict(_JOBS.get(item["id"], {}))
            items.append({k: v for k, v in item.items() if k != "files"} | {
                "size": sum(f["size"] for f in item["files"]),
                "runtime_size": (runtime_path(item["id"]).stat().st_size
                                 if item["id"] in ("quran-phonetics", "arabic-ctc")
                                 else sum(f["size"] for f in item["files"])) if ready else None,
                "runtime_version": "local-export" if item["id"] == "quran-phonetics" and ready else item["revision"][:8],
                "installed": installed, "ready": ready,
                "active": item["id"] == ACTIVE_ID, "job": job,
            })
    return {"active_id": ACTIVE_ID, "models": items}


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise ValueError("Загрузка допускается только по HTTPS.")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def _open_download(opener, url: str, headers: dict):
    """Retry expired CDN redirects without relaxing hash or HTTPS checks."""
    for attempt in range(3):
        try:
            fresh_url = url if attempt == 0 else f"{url}{'&' if '?' in url else '?'}download=true&request={uuid.uuid4().hex}"
            return opener.open(urllib.request.Request(fresh_url, headers=headers), timeout=30)
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 408, 429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise


def _download_ranges(path: Path, url: str, size: int, headers: dict,
                     cancel: threading.Event, on_chunk) -> None:
    """Bounded parallel byte ranges; exact Content-Range and final SHA required."""
    block = 8 * 1024 * 1024
    prefix = 0
    if path.exists():
        actual_size = path.stat().st_size
        if actual_size < size:
            prefix = actual_size // block * block
        elif actual_size == size and hasattr(os, "SEEK_HOLE"):
            with path.open("rb") as existing:
                try:
                    hole = os.lseek(existing.fileno(), 0, os.SEEK_HOLE)
                    prefix = size if hole == size else hole // block * block
                except OSError:
                    pass
    with path.open("r+b" if path.exists() else "wb") as output:
        output.truncate(size)
    on_chunk(prefix)
    abort = threading.Event()
    failures = []
    failure_lock = threading.Lock()

    def transfer(bounds):
        start, end = bounds
        cursor = start
        for attempt in range(4):
            if cancel.is_set() or abort.is_set():
                raise InterruptedError
            if cursor > end:
                return
            request_headers = headers | {"Range": f"bytes={cursor}-{end}"}
            fresh_url = f"{url}{'&' if '?' in url else '?'}download=true&request={uuid.uuid4().hex}"
            try:
                with _open_download(urllib.request.build_opener(_Redirect()), fresh_url,
                                    request_headers) as response, path.open("r+b") as output:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {cursor}-{end}/{size}":
                        raise OSError("Сервер не подтвердил запрошенную часть модели.")
                    output.seek(cursor)
                    while cursor <= end:
                        if cancel.is_set() or abort.is_set():
                            raise InterruptedError
                        chunk = response.read(min(1024 * 1024, end - cursor + 1))
                        if not chunk:
                            raise OSError("Неполная часть модели.")
                        output.write(chunk)
                        cursor += len(chunk)
                        on_chunk(len(chunk))
                    return
            except (OSError, urllib.error.URLError):
                if attempt == 3:
                    raise

    def guarded(bounds):
        try:
            return transfer(bounds)
        except BaseException as exc:
            with failure_lock:
                if not abort.is_set():
                    failures.append(exc)
                    abort.set()
            raise

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(guarded, (start, min(size - 1, start + block - 1)))
                   for start in range(prefix, size, block)]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException:
                pass
    if failures:
        raise failures[0]


def _progress(model_id: str, **fields) -> None:
    with _LOCK:
        _JOBS[model_id].update(fields)


def start_download(model_id: str, *, token: str = "") -> dict:
    item = spec(model_id)
    if item.get("hidden", False):
        raise ValueError("Модель исключена из каталога.")
    if item["gated"] and not token.strip():
        raise ValueError("После одобрения доступа автором введите токен Hugging Face.")
    if token and (not token.startswith("hf_") or any(c.isspace() for c in token)):
        raise ValueError("Введите корректный токен Hugging Face.")
    with _LOCK:
        if is_installed(item):
            return {"state": "installed"}
        if any(job.get("state") == "downloading" for job in _JOBS.values()):
            raise ValueError("Дождитесь завершения текущей загрузки или отмените её.")
        _JOBS[model_id] = {"state": "downloading", "received": 0,
                           "total": sum(f["size"] for f in item["files"]), "file": ""}
        cancel = _CANCEL[model_id] = threading.Event()
        threading.Thread(target=_download, args=(item, token, cancel), daemon=True).start()
        return dict(_JOBS[model_id])


def cancel_download(model_id: str) -> None:
    spec(model_id)
    with _LOCK:
        if model_id in _CANCEL:
            _CANCEL[model_id].set()


def remove_download(model_id: str) -> None:
    """Move only a managed download to recoverable storage; keep bundled weights."""
    item = spec(model_id)
    with _LOCK:
        if model_id == ACTIVE_ID:
            raise ValueError("Сначала выберите другую модель.")
        if _JOBS.get(model_id, {}).get("state") == "downloading":
            raise ValueError("Дождитесь завершения загрузки или отмените её.")
        folder = installed_path(item)
        if not folder.exists():
            raise ValueError("Скачанные файлы модели не найдены.")
        trash = ROOT / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        os.replace(folder, trash / f"{model_id}-{item['revision']}-{uuid.uuid4().hex}")
        _JOBS.pop(model_id, None)


def _download(item: dict, token: str, cancel: threading.Event) -> None:
    model_id = item["id"]
    destination = installed_path(item)
    temporary = destination.with_name(destination.name + ".partial")
    received = 0
    keep_partial = False

    def on_chunk(count):
        nonlocal received
        with _LOCK:
            received += count
            _progress(model_id, received=received)
    try:
        temporary.mkdir(parents=True, exist_ok=True)
        opener = urllib.request.build_opener(_Redirect())
        receipts = []
        for f in item["files"]:
            if cancel.is_set():
                raise InterruptedError
            _progress(model_id, file=f["path"])
            url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{f['path']}"
            headers = {"User-Agent": "AyahStudio/0.1"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            path = temporary / f["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            git_hash = hashlib.sha1(f"blob {f['size']}\0".encode())
            size = 0
            if f["size"] > 64 * 1024 * 1024:
                _download_ranges(path, url, f["size"], headers, cancel, on_chunk)
                with path.open("rb") as downloaded:
                    while chunk := downloaded.read(1024 * 1024):
                        if cancel.is_set():
                            raise InterruptedError
                        size += len(chunk)
                        digest.update(chunk)
                        git_hash.update(chunk)
            else:
                with _open_download(opener, url, headers) as response, path.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        if cancel.is_set():
                            raise InterruptedError
                        size += len(chunk)
                        if size > f["size"]:
                            raise ValueError("Размер файла отличается от каталога.")
                        output.write(chunk)
                        digest.update(chunk)
                        git_hash.update(chunk)
                        on_chunk(len(chunk))
            if size != f["size"]:
                raise ValueError("Файл загружен не полностью. Повторите загрузку.")
            if f["sha256"] and digest.hexdigest() != f["sha256"]:
                raise ValueError("Контрольная сумма файла отличается от каталога.")
            if not f["sha256"] and git_hash.hexdigest() != f["git_oid"]:
                raise ValueError("Контрольная сумма файла отличается от каталога.")
            receipts.append({"path": f["path"], "sha256": digest.hexdigest()})
        if cancel.is_set():
            raise InterruptedError
        if model_id == "arabic-ctc":
            path = temporary / "wav2vec2-ar-ctc-aligner/model.int8.onnx"
            data = json.loads(path.with_name("vocab.json").read_text())
            path.with_suffix(".onnx.vocab.json").write_text(json.dumps(
                {t: i for i, t in enumerate(data["tokens"])}, ensure_ascii=False))
            path.with_suffix(".onnx.meta.json").write_text(json.dumps({"do_normalize": True}))
        (temporary / "installation.json").write_text(json.dumps({
            "revision": item["revision"], "license": item["license"],
            "repo": item["repo"], "license_url": item.get("license_url"),
            "download_authorized": True, "files": receipts,
        }))
        if destination.exists():
            # Preserve a prior installation; only publish an entire new revision.
            raise ValueError("Папка версии уже существует. Проверьте её целостность.")
        os.replace(temporary, destination)
        if item.get("runtime_family"):
            from surah_cutter.core.model_runtime import ensure_runtime

            _progress(model_id, file="Подготовка модели")
            ensure_runtime(model_id, cancel)
        _progress(model_id, state="installed", file="")
    except InterruptedError:
        _progress(model_id, state="cancelled", error="Загрузка отменена.")
    except urllib.error.HTTPError as exc:
        keep_partial = True
        message = ("Доступ не предоставлен. Запросите доступ у автора и проверьте токен."
                   if exc.code in (401, 403) else f"Сервер загрузки вернул ошибку {exc.code}. Повторите попытку.")
        _progress(model_id, state="error", error=message)
    except (OSError, urllib.error.URLError) as exc:
        keep_partial = True
        logging.getLogger(__name__).warning("Model download %s: %s", model_id, exc)
        _progress(model_id, state="error", error="Не удалось загрузить и проверить файлы. Проверьте сеть и место на диске.")
    except ValueError:
        _progress(model_id, state="error", error="Не удалось загрузить и проверить файлы. Проверьте сеть и место на диске.")
    finally:
        token = ""
        if not keep_partial:
            shutil.rmtree(temporary, ignore_errors=True)
