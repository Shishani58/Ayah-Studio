"""FastAPI app for the web GUI: JSON API + SSE progress + audio/static serving.

All heavy work delegates to :mod:`surah_cutter.core` / :mod:`surah_cutter.data`;
this layer only translates HTTP ↔ project state (:mod:`surah_cutter.webgui.state`).
Generation runs in a worker thread and streams per-sura progress over SSE.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from surah_cutter.data import quran_text
from surah_cutter.core import model_store
from surah_cutter.webgui import projects, settings
from surah_cutter.webgui.state import (
    DB_CODE,
    DB_VERSION,
    STATE,
    classify_status,
    discover_audio,
    waveform_events,
    waveform_peaks,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
#: Bundled Qur'an (QPC Hafs Uthmani) font, served at /fonts for the review text.
_FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_WORD_LOCK = threading.Lock()
_GEN_MODES = {"auto", "full", "fast", "hybrid", "tiled", "words"}


def _model_runtime(handler):
    """Hold model selection fixed for synchronous inference and its writes."""
    @wraps(handler)
    def wrapped(*args, **kwargs):
        if not model_store.RUNTIME_LOCK.acquire(blocking=False):
            raise HTTPException(409, "Расчёт модели уже выполняется. Дождитесь завершения.")
        try:
            if handler.__name__ in ("recheck_sura", "get_fast_log") and model_store.active_kind() != "phonetic":
                raise HTTPException(400, "Проверка фонетических начал доступна только для Quran Phonetics.")
            return handler(*args, **kwargs)
        finally:
            model_store.RUNTIME_LOCK.release()
    return wrapped


# --- generation job (one at a time) ---------------------------------------


class _GenJob:
    """A running generation pass; streams events to any number of SSE clients."""

    def __init__(self, db_path: Path, audio_map: dict[int, Path]) -> None:
        self.db_path = db_path.resolve()
        self.audio_map = MappingProxyType({s: p.resolve() for s, p in audio_map.items()})
        self.q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.running = False
        self.cancel = threading.Event()

    def start(self, suras: list[int], mode: str, reciter: str | None) -> None:
        self.running = True
        self.thread = threading.Thread(
            target=self._run, args=(suras, mode, reciter), daemon=True
        )
        self.thread.start()

    def request_cancel(self) -> None:
        """Ask the worker to stop after the current sura finishes."""
        self.cancel.set()

    def _emit(self, event: dict[str, Any]) -> None:
        self.q.put(event)

    def _run(self, suras: list[int], mode: str, reciter: str | None) -> None:
        with model_store.RUNTIME_LOCK:
            self._run_selected(suras, mode, reciter)

    def _run_selected(self, suras: list[int], mode: str, reciter: str | None) -> None:
        from surah_cutter.core.pipeline import process_surah
        from surah_cutter.data.project_db import ProjectDB

        cancelled = False
        completed = 0
        try:
            self._emit({"type": "start", "total": len(suras)})
            db = ProjectDB.open(str(self.db_path), DB_CODE, DB_VERSION)
            try:
                for i, surah in enumerate(suras):
                    # Cancellation is checked between suras (a sura's alignment
                    # is not interruptible mid-flight).
                    if self.cancel.is_set():
                        cancelled = True
                        break
                    self._emit({"type": "sura_start", "surah": surah, "index": i})
                    audio = self.audio_map.get(surah)
                    if audio is None:
                        self._emit(
                            {"type": "sura", "surah": surah, "index": i,
                             "written": 0, "skipped": "no-audio"}
                        )
                        completed = i + 1
                        continue
                    trace: dict[str, Any] = {}
                    log_lines: list[str] = []

                    def log_event(line: str) -> None:
                        text = str(line)
                        log_lines.append(text)
                        self._emit({
                            "type": "sura_log",
                            "surah": surah,
                            "index": i,
                            "line": text,
                        })

                    existing = [(s, a, t) for s, a, t in db.get_timings() if s == surah]
                    complete = {a for _, a, _ in existing} == set(
                        range(1, quran_text.ayah_count(surah) + 1)
                    )
                    if mode == "words" and complete:
                        rows, written = existing, 0
                        log_event("Пословный режим: используются сохранённые границы аятов.")
                    else:
                        boundary_mode = ("full" if model_store.active_kind() != "phonetic" else "fast") if mode == "words" else mode
                        rows = process_surah(
                            str(audio), surah, reciter=reciter, mode=boundary_mode,
                            project_db=db,
                            trace=trace,
                            log_event=log_event,
                        )
                        written = db.write_timings(rows, source="auto")
                        _write_trace_confidences(db, surah, trace)
                    word_result = {}
                    if mode == "words":
                        word_result = _align_sura_words(db, self.db_path, audio, surah, log_event)
                    log_paths = _save_sura_log(
                        surah, mode, trace, log_lines, rows, self.db_path
                    )
                    completed = i + 1
                    event = {
                        "type": "sura", "surah": surah, "index": i,
                        "ayahs": len(rows), "written": written, **word_result,
                    }
                    if log_paths:
                        event["log_paths"] = log_paths
                    if trace and trace.get("mode") != "full":
                        event["fast_trace"] = _fast_trace_summary(trace)
                    self._emit(event)
            finally:
                db.close()
            self._emit(
                {"type": "cancelled", "completed": completed} if cancelled
                else {"type": "done", "completed": completed}
            )
        except Exception as exc:  # surface failures to the UI, don't crash the server
            self._emit({"type": "error", "message": str(exc)})
        finally:
            self.running = False
            self.q.put(None)  # sentinel: close SSE streams


_JOB: _GenJob | None = None
_PROJECT_LOCK = threading.RLock()


def create_app() -> FastAPI:
    app = FastAPI(title="Ayah Studio")
    _restore_from_settings()
    selected = settings.load().get("alignment_model", "quran-phonetics")
    valid_ids = {item["id"] for item in model_store.catalog()} | {"quran-phonetics-fp32"}
    model_store.ACTIVE_ID = selected if selected in valid_ids else "quran-phonetics"

    @app.get("/api/models")
    def get_models() -> dict[str, Any]:
        return model_store.listing()

    @app.post("/api/models/{model_id}/download")
    def download_model(model_id: str, payload: dict[str, Any] = Body(default={})) -> dict:
        try:
            token = payload.get("token", "")
            if not isinstance(token, str):
                raise ValueError("Введите токен текстом.")
            return model_store.start_download(model_id, token=token)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/models/{model_id}/cancel")
    def cancel_model_download(model_id: str) -> dict:
        try:
            model_store.cancel_download(model_id)
            return {"ok": True}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/models/{model_id}/select")
    @_model_runtime
    def select_model(model_id: str) -> dict:
        with _PROJECT_LOCK:
            if (_JOB is not None and _JOB.running) or _WORD_LOCK.locked():
                raise HTTPException(409, "Дождитесь завершения расчёта перед сменой модели.")
            try:
                path = model_store.runtime_path(model_id)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            if path is None:
                raise HTTPException(400, "Сначала скачайте модель.")
            try:
                import numpy as np
                from surah_cutter.core import align

                emitter = model_store.load_emitter(model_id)
                if hasattr(emitter, "probe"):
                    emitter.probe()
                else:
                    probe = emitter.emissions(np.zeros(16000, dtype=np.float32))
                    if probe.ndim != 2 or probe.shape[1] != len(emitter.vocab) or not np.isfinite(probe).all():
                        raise ValueError("Несовместимый выход модели.")
                fields = {"alignment_model": model_id}
                if model_id not in ("quran-phonetics", "quran-phonetics-fp32"):
                    fields["mode"] = "full"
                settings.update(strict=True, **fields)
            except Exception as exc:
                raise HTTPException(400, "Не удалось проверить модель или сохранить выбор. Текущая модель сохранена.") from exc
            model_store.ACTIVE_ID = model_id
            bundle = align._OnnxBundle(emitter)
            if model_id == "arabic-ctc":
                bundle.text_kind = "arabic"
                bundle.lag_ms = 0
            align._ONNX_CACHE.clear()
            align._ONNX_CACHE[str(path)] = bundle
            from surah_cutter.core.model_runtime import release_runtimes

            release_runtimes(keep=emitter)
            return {**model_store.listing(), "mode": settings.load().get("mode", "auto")}

    @app.post("/api/models/{model_id}/remove")
    @_model_runtime
    def remove_model(model_id: str) -> dict:
        with _PROJECT_LOCK:
            if (_JOB is not None and _JOB.running) or _WORD_LOCK.locked():
                raise HTTPException(409, "Дождитесь завершения расчёта.")
            try:
                model_store.remove_download(model_id)
            except (ValueError, OSError) as exc:
                raise HTTPException(400, str(exc)) from exc
            return {"ok": True}

    # -- project ----------------------------------------------------------
    @app.get("/api/project")
    def get_project() -> dict[str, Any]:
        cfg = settings.load()
        managed = next((p for p in projects.list_projects()
                        if STATE.db_path and Path(p["db_path"]).resolve() == STATE.db_path.resolve()), {})
        return {
            "audio_dir": str(STATE.audio_dir) if STATE.audio_dir else None,
            "db_path": str(STATE.db_path) if STATE.db_path else None,
            "audio_count": len(STATE.audio_map),
            "suras_with_audio": sorted(STATE.audio_map),
            "mode": cfg.get("mode", "auto"),
            "project_id": managed.get("id"),
            "reciter": managed.get("reciter", STATE.audio_dir.name if STATE.audio_dir else ""),
            "recording": managed.get("recording", "Внешний проект"),
            "time_step": cfg.get("time_step", 10),
            "arabic_size": cfg.get("arabic_size", 30),
            "theme": cfg.get("theme", "dark"),
            "alignment_model": model_store.ACTIVE_ID,
        }

    @app.post("/api/project/open")
    def open_project(payload: dict[str, str] = Body(...)) -> dict[str, Any]:
        with _PROJECT_LOCK:
            if _JOB is not None and _JOB.running:
                raise HTTPException(409, "generation already running")
            audio_dir = payload.get("audio_dir")
            db_path = payload.get("db_path")
            new_audio_dir = STATE.audio_dir
            new_audio_map = STATE.audio_map
            new_db_path = STATE.db_path
            if audio_dir:
                new_audio_dir = Path(audio_dir)
                if not new_audio_dir.is_dir():
                    raise HTTPException(400, f"not a folder: {audio_dir}")
                new_audio_map = discover_audio(new_audio_dir)
            if db_path:
                from surah_cutter.data.project_db import ProjectDB

                ProjectDB.open(db_path, DB_CODE, DB_VERSION).close()  # ensure schema
                new_db_path = Path(db_path)
            # Persist before publishing so a failed save leaves the selection intact.
            try:
                settings.update(
                    strict=True,
                    audio_dir=str(new_audio_dir) if new_audio_dir else None,
                    db_path=str(new_db_path) if new_db_path else None,
                )
            except OSError as exc:
                raise HTTPException(500, "Не удалось сохранить проект. Проверьте доступ к папке настроек.") from exc
            # Publish the complete selection only after every input is ready.
            STATE.audio_dir = new_audio_dir
            STATE.audio_map = new_audio_map
            STATE.db_path = new_db_path
            return get_project()

    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return projects.list_projects()

    @app.post("/api/projects/create")
    def create_project(payload: dict[str, str] = Body(...)) -> dict[str, Any]:
        with _PROJECT_LOCK:
            if _JOB is not None and _JOB.running:
                raise HTTPException(409, "Дождитесь завершения генерации перед импортом проекта.")
            try:
                project = projects.create(payload.get("reciter", ""),
                                          payload.get("recording", ""),
                                          payload.get("audio_dir", ""),
                                          payload.get("db_path", ""))
            except (OSError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                raise HTTPException(400, "Не удалось импортировать БД. Выберите SQLite .db с разметкой.") from exc
            return open_project(project)

    @app.post("/api/projects/activate")
    def activate_project(payload: dict[str, str] = Body(...)) -> dict[str, Any]:
        try:
            project = projects.get(payload.get("id", ""))
        except (OSError, ValueError) as exc:
            raise HTTPException(404, "Проект не найден. Обновите список проектов.") from exc
        return open_project(project)

    @app.post("/api/settings")
    def set_settings(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        allowed = {k: v for k, v in payload.items() if k in ("mode", "time_step", "arabic_size", "theme")}
        if "theme" in allowed and allowed["theme"] not in ("light", "dark"):
            raise HTTPException(400, "Выберите светлую или тёмную тему.")
        if "mode" in allowed and allowed["mode"] not in _GEN_MODES:
            raise HTTPException(400, f"unknown mode: {allowed['mode']}")
        if "mode" in allowed and model_store.active_kind() != "phonetic" and allowed["mode"] not in ("full", "tiled", "words"):
            raise HTTPException(400, "Арабская модель поддерживает полный, точный и пословный режимы.")
        if "time_step" in allowed and allowed["time_step"] not in (10, 50, 100):
            raise HTTPException(400, "Выберите шаг времени: 10, 50 или 100 мс.")
        if "arabic_size" in allowed and allowed["arabic_size"] not in (26, 30, 36, 42):
            raise HTTPException(400, "Выберите размер текста из списка.")
        if payload.get("project_id"):
            with _PROJECT_LOCK:
                if payload["project_id"] != get_project()["project_id"]:
                    raise HTTPException(409, "Проект изменился. Откройте настройки заново.")
                try:
                    projects.rename(payload["project_id"], payload.get("reciter", ""),
                                    payload.get("recording", ""))
                except (OSError, ValueError) as exc:
                    raise HTTPException(400, str(exc)) from exc
        try:
            return settings.update(strict=True, **allowed)
        except OSError as exc:
            raise HTTPException(500, "Не удалось записать настройки. Проверьте доступ к папке настроек.") from exc

    # -- suras + statuses -------------------------------------------------
    @app.get("/api/suras")
    def list_suras() -> list[dict[str, Any]]:
        counts = quran_text.surah_ayah_counts()
        reviews = _reviews_by_sura()
        complete = _processed_suras()
        word_progress = _word_progress()
        out = []
        for s in range(1, 115):
            ac = counts.get(s, 0)
            out.append({
                "surah": s,
                "ayah_count": ac,
                "has_audio": s in STATE.audio_map,
                "words_complete": word_progress.get(s, {}).get("complete", False),
                "status": (
                    "partial" if reviews.get(s) and s not in complete
                    else classify_status(reviews.get(s), ac)
                ),
            })
        return out

    # -- generation -------------------------------------------------------
    @app.post("/api/generate")
    def generate(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        global _JOB
        with _PROJECT_LOCK:
            if STATE.db_path is None:
                raise HTTPException(400, "open a project DB first")
            if _JOB is not None and _JOB.running:
                raise HTTPException(409, "generation already running")
            suras = payload.get("suras") or sorted(STATE.audio_map)
            suras = [s for s in suras if s in STATE.audio_map]
            mode = payload.get("mode", "auto")
            if mode not in _GEN_MODES:
                raise HTTPException(400, f"unknown mode: {mode}")
            if model_store.active_kind() != "phonetic" and mode not in ("full", "tiled", "words"):
                raise HTTPException(400, "Для арабской модели выберите полный, точный или пословный режим.")
            if payload.get("only_missing"):
                done = ({s for s, p in _word_progress().items() if p["complete"]}
                        if mode == "words" else _processed_suras())
                suras = [s for s in suras if s not in done]
            if not suras:
                raise HTTPException(400, "no suras to process (all done or no audio)")
            reciter = payload.get("reciter")
            settings.update(mode=mode)  # remember last-used mode
            _JOB = _GenJob(STATE.db_path, STATE.audio_map)
            _JOB.start(suras, mode, reciter)
            return {"started": True, "suras": suras, "mode": mode}

    @app.get("/api/generate/stream")
    def generate_stream() -> StreamingResponse:
        job = _JOB

        def gen():
            if job is None:
                yield _sse({"type": "idle"})
                return
            while True:
                event = job.q.get()
                if event is None:
                    break
                yield _sse(event)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/generate/cancel")
    def cancel_generation() -> dict[str, Any]:
        if _JOB is not None and _JOB.running:
            _JOB.request_cancel()
            return {"cancelling": True}
        return {"cancelling": False}

    # -- one sura: ayahs + timings + text ---------------------------------
    @app.get("/api/sura/{surah}")
    def get_sura(surah: int) -> dict[str, Any]:
        _check_surah(surah)
        counts = quran_text.surah_ayah_counts()
        ac = counts.get(surah, 0)
        timings, reviews = _sura_timings_reviews(surah)
        ayahs = []
        for a in range(1, ac + 1):
            rv = reviews.get(a, {})
            ayahs.append({
                "ayah": a,
                "time": timings.get(a),
                "status": rv.get("status"),
                "confidence": rv.get("confidence"),
                "text": quran_text.ayah_text(surah, a),
            })
        return {
            "surah": surah,
            "ayah_count": ac,
            "word_progress": _word_progress(surah=surah).get(surah),
            "has_audio": surah in STATE.audio_map,
            "ayahs": ayahs,
        }

    @app.get("/api/sura/{surah}/waveform")
    def get_waveform(surah: int) -> dict[str, Any]:
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        peaks, duration_ms = waveform_peaks(str(audio))
        return {"surah": surah, "peaks": peaks, "duration_ms": duration_ms}

    @app.get("/api/sura/{surah}/waveform/stream")
    def stream_waveform(surah: int) -> StreamingResponse:
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")

        async def events():
            source = waveform_events(str(audio))
            try:
                async for event in source:
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            finally:
                await source.aclose()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/sura/{surah}/recheck")
    @_model_runtime
    def recheck_sura(surah: int) -> dict[str, Any]:
        """Audit this sura's stored timings against the audio (read-only).

        Greedy-decodes the audio at every stored ayah onset and compares what is
        heard to that ayah's expected opening, flagging timecodes that look
        drifted / off-by-one. Writes nothing — it only reports, so it never
        clobbers manual work; the user fixes flagged ayahs by ear.
        """
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        timings, _reviews = _sura_timings_reviews(surah)
        if not timings:
            raise HTTPException(400, "у суры нет тайм-кодов — сначала рассчитайте её")
        from surah_cutter.core import decode, recheck

        words = quran_text.surah_words(surah)
        pcm = decode.decode(str(audio))
        results = recheck.recheck_surah(
            pcm, decode.DEFAULT_SR, words, timings, surah=surah
        )
        return {
            "surah": surah,
            "results": results,
            "summary": recheck.summarize(results),
        }

    @app.post("/api/sura/{surah}/draft-pauses")
    def draft_pauses(surah: int) -> dict[str, Any]:
        """Write a rough pause/onset draft for this sura.

        This is deliberately approximate: it creates usable markers from the
        waveform before the heavier correction pass. Existing ``edited`` and
        ``verified`` rows are treated as anchors and are not overwritten.
        """
        _check_surah(surah)
        with _PROJECT_LOCK:
            _require_db()
            db_path = STATE.db_path
            audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        from surah_cutter.core import decode, draft
        from surah_cutter.data.project_db import ProjectDB

        timings, reviews = _sura_timings_reviews(surah, db_path)
        anchors = {
            ayah: timings[ayah]
            for ayah, review in reviews.items()
            if review.get("status") in ("edited", "verified") and ayah in timings
        }
        pcm = decode.decode(str(audio))
        result = draft.draft_from_pauses(
            pcm,
            decode.DEFAULT_SR,
            surah,
            anchors=anchors,
        )
        with ProjectDB.open(str(db_path), DB_CODE, DB_VERSION) as db:
            written = db.write_timings(result.rows, source="pause-draft")
        return {
            "ok": True,
            "surah": surah,
            "written": written,
            "ayahs": len(result.rows),
            "pauses": result.pause_count,
            "pause_backed": result.pause_backed,
            "interpolated": result.interpolated,
            "anchored": result.anchored,
        }

    @app.post("/api/sura/{surah}/correct-draft")
    @_model_runtime
    def correct_draft(
        surah: int,
        payload: dict[str, Any] = Body(default={}),
    ) -> dict[str, Any]:
        """Correct current draft timings with the exact alignment pipeline.

        The write path still preserves ``edited``/``verified`` rows. In practice
        those rows are the known anchors; if an auto pause drifted in the middle,
        the correction pass recomputes the surrounding auto timings from the
        audio/text model instead of trusting the pause sequence.
        """
        _check_surah(surah)
        with _PROJECT_LOCK:
            _require_db()
            db_path = STATE.db_path
            audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        mode = payload.get("mode", "tiled")
        if mode not in ("auto", "full", "fast", "tiled"):
            raise HTTPException(400, f"unknown correction mode: {mode}")
        from surah_cutter.core.pipeline import process_surah
        from surah_cutter.data.project_db import ProjectDB

        trace: dict[str, Any] = {}
        log_lines: list[str] = []
        with ProjectDB.open(str(db_path), DB_CODE, DB_VERSION) as db:
            rows = process_surah(
                str(audio), surah, mode=mode, project_db=db,
                trace=trace, log_event=log_lines.append,
            )
            written = db.write_timings(rows, source=f"correct-{mode}")
            _write_trace_confidences(db, surah, trace)
        _save_sura_log(surah, f"correct-{mode}", trace, log_lines, rows, db_path)
        return {
            "ok": True,
            "surah": surah,
            "mode": mode,
            "written": written,
            "ayahs": len(rows),
        }

    @app.get("/api/sura/{surah}/fast-phase1")
    def get_fast_phase1(surah: int) -> dict[str, Any]:
        """Return fast-mode phase-1 pause marks for visual debugging."""
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        from surah_cutter.core import candidates, decode

        pcm = decode.decode(str(audio))
        pauses = candidates.detect_pauses(pcm, decode.DEFAULT_SR)
        return {
            "surah": surah,
            "pauses": [
                {
                    "start_ms": round(p.start_ms, 1),
                    "end_ms": round(p.end_ms, 1),
                    "center_ms": round(p.center_ms, 1),
                    "onset_ms": round(p.center_ms, 1),
                    "depth_ms": round(p.depth_ms, 1),
                }
                for p in pauses
            ],
        }

    @app.get("/api/sura/{surah}/fast-combined-phase1")
    def get_fast_combined_phase1(surah: int) -> dict[str, Any]:
        """Return the combined pause marks used by the fast algorithm."""
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        from surah_cutter.core import candidates, decode

        pcm = decode.decode(str(audio))
        pauses = candidates.detect_combined_pauses(pcm, decode.DEFAULT_SR, surah=surah)
        return {
            "surah": surah,
            "pauses": [
                {
                    "start_ms": round(p.start_ms, 1),
                    "end_ms": round(p.end_ms, 1),
                    "center_ms": round(p.center_ms, 1),
                    "depth_ms": round(p.depth_ms, 1),
                }
                for p in pauses
            ],
        }

    @app.get("/api/sura/{surah}/pause-pipeline")
    def get_pause_pipeline(surah: int) -> dict[str, Any]:
        """Return energy→DSP→local-VAD pause decisions for debugging phases."""
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        from surah_cutter.core import candidates, decode

        pcm = decode.decode(str(audio))
        decisions = candidates.pause_pipeline_decisions(pcm, decode.DEFAULT_SR, surah=surah)
        return {
            "surah": surah,
            "decisions": [
                {
                    "start_ms": round(d.pause.start_ms, 1),
                    "end_ms": round(d.pause.end_ms, 1),
                    "center_ms": round(d.pause.center_ms, 1),
                    "depth_ms": round(d.pause.depth_ms, 1),
                    "accepted": d.accepted,
                    "dsp_ok": d.dsp_ok,
                    "vad_ok": d.vad_ok,
                    "reason": d.reason,
                    "flatness": round(d.flatness, 3),
                    "zcr": round(d.zcr, 3),
                    "rms_ratio": round(d.rms_ratio, 3),
                }
                for d in decisions
            ],
        }

    @app.get("/api/sura/{surah}/logs")
    def list_sura_logs(surah: int) -> dict[str, Any]:
        _check_surah(surah)
        folder = _sura_log_folder()
        entries: dict[str, dict[str, Any]] = {}
        for path in folder.glob(f"sura_{surah:03d}_*"):
            if path.suffix not in {".log", ".json"} or path.is_symlink() or not path.is_file():
                continue
            if not re.fullmatch(rf"sura_{surah:03d}_[A-Za-z0-9_-]+", path.stem):
                continue
            entry = entries.setdefault(path.stem, {
                "id": path.stem, "mode": path.stem.split("_", 2)[2], "modified_ms": 0,
            })
            entry["modified_ms"] = max(entry["modified_ms"], int(path.stat().st_mtime * 1000))
        return {"logs": sorted(entries.values(), key=lambda item: item["modified_ms"], reverse=True)}

    @app.get("/api/sura/{surah}/logs/{log_id}")
    def get_sura_log(surah: int, log_id: str) -> dict[str, Any]:
        _check_surah(surah)
        if not re.fullmatch(rf"sura_{surah:03d}_[A-Za-z0-9_-]+", log_id):
            raise HTTPException(404, "Лог не найден")
        folder = _sura_log_folder()
        result = {"id": log_id, "log": None, "json": None}
        try:
            for suffix, key in ((".log", "log"), (".json", "json")):
                path = folder / f"{log_id}{suffix}"
                if path.is_file() and not path.is_symlink():
                    result[key] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise HTTPException(500, "Не удалось прочитать лог") from exc
        if result["log"] is None and result["json"] is None:
            raise HTTPException(404, "Лог не найден")
        return result

    @app.get("/api/sura/{surah}/fast-log")
    @_model_runtime
    def get_fast_log(surah: int) -> dict[str, Any]:
        """Run fast alignment in debug mode and return a readable phase log."""
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        from surah_cutter.core import align_fast, candidates, decode

        words = quran_text.surah_words(surah)
        pcm = decode.decode(str(audio))
        decisions = candidates.pause_pipeline_decisions(pcm, decode.DEFAULT_SR, surah=surah)
        trace: dict[str, Any] = {}
        rows = align_fast.align_surah_fast(
            pcm, decode.DEFAULT_SR, words, surah=surah, trace=trace
        )
        return {
            "surah": surah,
            "log": _format_fast_log(surah, decisions, trace, rows),
        }

    @app.get("/api/sura/{surah}/ten-phase")
    def get_ten_phase(
        surah: int,
        start_ms: float = 0.0,
        end_ms: float | None = None,
        threshold: float = 0.5,
        merge_gap_ms: float = 180.0,
        min_pause_ms: float = 300.0,
    ) -> dict[str, Any]:
        """Return TEN VAD pause marks for the requested visible time range."""
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        from surah_cutter.core import candidates, decode

        total_ms = decode.duration_ms(str(audio))
        start = max(0.0, start_ms)
        end = min(total_ms, end_ms if end_ms is not None else start + 120_000.0)
        if end <= start:
            raise HTTPException(400, "empty TEN range")
        # Keep the UI responsive on long suras; frontend requests the visible
        # viewport and reloads as the user scrolls/zooms.
        if end - start > 180_000.0:
            end = start + 180_000.0
        threshold = min(0.95, max(0.05, threshold))
        merge_gap_ms = min(800.0, max(0.0, merge_gap_ms))
        min_pause_ms = min(2000.0, max(80.0, min_pause_ms))
        try:
            pcm = decode.decode_range(str(audio), start_ms=start, end_ms=end)
            pauses = candidates.detect_ten_pauses(
                pcm,
                decode.DEFAULT_SR,
                threshold=threshold,
                merge_gap_ms=merge_gap_ms,
                min_pause_ms=min_pause_ms,
            )
        except RuntimeError as exc:
            raise HTTPException(501, str(exc)) from exc
        return {
            "surah": surah,
            "start_ms": round(start, 1),
            "end_ms": round(end, 1),
            "threshold": threshold,
            "merge_gap_ms": merge_gap_ms,
            "min_pause_ms": min_pause_ms,
            "pauses": [
                {
                    "start_ms": round(p.start_ms + start, 1),
                    "end_ms": round(p.end_ms + start, 1),
                    "center_ms": round(p.center_ms + start, 1),
                    "depth_ms": round(p.depth_ms, 1),
                }
                for p in pauses
            ],
        }

    @app.get("/api/sura/{surah}/vad-phase")
    def get_vad_phase(
        surah: int,
        start_ms: float = 0.0,
        end_ms: float | None = None,
        threshold: float = 0.5,
        merge_gap_ms: float = 180.0,
        min_pause_ms: float = 300.0,
    ) -> dict[str, Any]:
        """Return Silero VAD pause marks for the requested visible time range."""
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        from surah_cutter.core import candidates, decode

        total_ms = decode.duration_ms(str(audio))
        start = max(0.0, start_ms)
        end = min(total_ms, end_ms if end_ms is not None else start + 120_000.0)
        if end <= start:
            raise HTTPException(400, "empty VAD range")
        if end - start > 180_000.0:
            end = start + 180_000.0
        threshold = min(0.95, max(0.05, threshold))
        merge_gap_ms = min(800.0, max(0.0, merge_gap_ms))
        min_pause_ms = min(2000.0, max(80.0, min_pause_ms))
        try:
            pcm = decode.decode_range(str(audio), start_ms=start, end_ms=end)
            pauses = candidates.detect_silero_pauses(
                pcm,
                decode.DEFAULT_SR,
                threshold=threshold,
                merge_gap_ms=merge_gap_ms,
                min_pause_ms=min_pause_ms,
            )
        except RuntimeError as exc:
            raise HTTPException(501, str(exc)) from exc
        return {
            "surah": surah,
            "start_ms": round(start, 1),
            "end_ms": round(end, 1),
            "threshold": threshold,
            "merge_gap_ms": merge_gap_ms,
            "min_pause_ms": min_pause_ms,
            "pauses": [
                {
                    "start_ms": round(p.start_ms + start, 1),
                    "end_ms": round(p.end_ms + start, 1),
                    "center_ms": round(p.center_ms + start, 1),
                    "depth_ms": round(p.depth_ms, 1),
                }
                for p in pauses
            ],
        }

    @app.get("/api/audio/{surah}")
    def get_audio(surah: int) -> FileResponse:
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        return FileResponse(str(audio))

    # -- local word review ------------------------------------------------
    @app.get("/api/sura/{surah}/words")
    def get_sura_words(surah: int) -> dict[str, Any]:
        from surah_cutter.core.decode import duration_ms
        from surah_cutter.data.project_db import ProjectDB

        _check_surah(surah)
        _require_db()
        with _PROJECT_LOCK:
            db_path = STATE.db_path.resolve()
            audio = STATE.audio_for(surah)
            if audio is None:
                raise HTTPException(404, "Нет аудио суры.")
            timings, _ = _sura_timings_reviews(surah, db_path)
            with ProjectDB.open(str(db_path), DB_CODE, DB_VERSION) as db:
                cursor = db._conn.execute(
                    "SELECT word_index, time, confidence, status, ayah_start, ayah_end "
                    "FROM word_timings WHERE sura=?", (surah,)
                )
                keys = [col[0] for col in cursor.description]
                saved = {row[0]: dict(zip(keys, row)) for row in cursor}
        duration = duration_ms(str(audio))
        words, numbers = [], {}
        for word in quran_text.surah_words(surah):
            numbers[word.ayah] = numbers.get(word.ayah, 0) + 1
            start = timings.get(word.ayah)
            end = timings.get(word.ayah + 1,
                duration if word.ayah == quran_text.ayah_count(surah) else None)
            row = saved.get(word.index, {})
            words.append({**row, "word_index": word.index, "text": word.text,
                "ayah": word.ayah, "number": numbers[word.ayah],
                "start_ms": start, "end_ms": end,
                "stale": bool(row and (row["ayah_start"], row["ayah_end"]) != (start, end))})
        return {"surah": surah, "db_path": str(db_path), "words": words,
                "word_progress": _word_progress(db_path, surah).get(surah)}

    @app.get("/api/sura/{surah}/ayah/{ayah}/words")
    def get_words(surah: int, ayah: int) -> dict[str, Any]:
        db_path, audio, words, start, end = _word_context(surah, ayah)
        return _word_response(db_path, surah, ayah, words, start, end)

    @app.post("/api/sura/{surah}/ayah/{ayah}/words/align")
    @_model_runtime
    def generate_words(surah: int, ayah: int,
                       payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        from surah_cutter.core import decode, word_align
        from surah_cutter.data.project_db import ProjectDB

        if not _WORD_LOCK.acquire(blocking=False):
            raise HTTPException(409, "Пословный расчёт уже выполняется.")
        try:
            db_path, audio, words, start, end = _word_context(surah, ayah, payload)
            pcm = decode.decode_range(str(audio), start_ms=start, end_ms=end)
            if abs(len(pcm) * 1000 / decode.DEFAULT_SR - (end - start)) > 100:
                raise HTTPException(400, "Границы аята выходят за аудио. Уточните метки.")
            rows = word_align.align_ayah_words(pcm, decode.DEFAULT_SR, words)
            with _PROJECT_LOCK:
                current = _word_context(surah, ayah, payload)
                if current[:2] != (db_path, audio) or current[3:] != (start, end):
                    raise HTTPException(409, "Проект или границы аята изменились. Повторите расчёт.")
                with ProjectDB.open(str(db_path), DB_CODE, DB_VERSION) as db:
                    db.write_word_timings(surah, ayah,
                        [(i, start + t, c) for i, t, c in rows], start, end,
                        model_store.source())
            return _word_response(db_path, surah, ayah, words, start, end)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            _WORD_LOCK.release()

    @app.post("/api/sura/{surah}/ayah/{ayah}/words/edit")
    def edit_word(surah: int, ayah: int,
                  payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        from surah_cutter.data.project_db import ProjectDB

        with _PROJECT_LOCK:
            db_path, audio, words, start, end = _word_context(surah, ayah, payload)
            index = payload.get("word_index")
            if type(index) is not int or index not in {w.index for w in words}:
                raise HTTPException(400, "Выберите существующее слово аята.")
            try:
                with ProjectDB.open(str(db_path), DB_CODE, DB_VERSION) as db:
                    if payload.get("action") == "verify":
                        db.verify_word(surah, ayah, index, start, end)
                    else:
                        db.set_word_time(surah, ayah, index, payload.get("time"), start, end)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return _word_response(db_path, surah, ayah, words, start, end)

    # -- edits ------------------------------------------------------------
    @app.post("/api/timing")
    def set_timing(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        surah, ayah, time = payload["surah"], payload["ayah"], payload["time"]
        if any(type(v) is not int for v in (surah, ayah, time)):
            raise HTTPException(400, "Введите номер суры, аята и время целыми числами.")
        _check_surah(surah)
        _require_db()
        if time < 0 or not 1 <= ayah <= quran_text.surah_ayah_counts()[surah]:
            raise HTTPException(400, "Введите неотрицательное время и существующий номер аята.")
        from surah_cutter.data.project_db import ProjectDB

        with _PROJECT_LOCK:
            if payload.get("db_path") and Path(payload["db_path"]).resolve() != STATE.db_path.resolve():
                raise HTTPException(409, "Проект изменился. Откройте суру заново перед правкой.")
            with ProjectDB.open(str(STATE.db_path), DB_CODE, DB_VERSION) as db:
                previous = db._conn.execute(
                    "SELECT MAX(time) FROM timings WHERE sura=? AND ayah<?", (surah, ayah)
                ).fetchone()[0]
                following = db._conn.execute(
                    "SELECT MIN(time) FROM timings WHERE sura=? AND ayah>?", (surah, ayah)
                ).fetchone()[0]
                if ((previous is not None and time <= previous)
                        or (following is not None and time >= following)):
                    raise HTTPException(400, "Начало аята должно находиться между соседними метками.")
                db.set_time(surah, ayah, time)
        return {"ok": True, "surah": surah, "ayah": ayah, "time": int(time)}

    @app.post("/api/verify")
    def set_verify(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        surah, ayah = int(payload["surah"]), int(payload["ayah"])
        status = payload.get("status", "verified")
        _check_surah(surah)
        _require_db()
        from surah_cutter.data.project_db import ProjectDB

        with ProjectDB.open(str(STATE.db_path), DB_CODE, DB_VERSION) as db:
            db.set_status(surah, ayah, status)
        return {"ok": True, "surah": surah, "ayah": ayah, "status": status}

    @app.post("/api/export")
    def export(payload: dict[str, str] = Body(...)) -> dict[str, Any]:
        _require_db()
        dst = payload.get("dst")
        if not dst:
            raise HTTPException(400, "missing 'dst'")
        from surah_cutter.data.export_db import ExportError, export_db

        try:
            out = export_db(str(STATE.db_path), dst)
        except ExportError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "path": out}

    # -- bundled fonts ----------------------------------------------------
    @app.get("/fonts/{name}")
    def get_font(name: str) -> FileResponse:
        fp = (_FONTS_DIR / name).resolve()
        if _FONTS_DIR.resolve() not in fp.parents or not fp.is_file():
            raise HTTPException(404, f"no font {name}")
        return FileResponse(str(fp))

    # -- static (SPA) -----------------------------------------------------
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    return app


# --- helpers --------------------------------------------------------------


def _align_sura_words(db, db_path, audio, surah, log_event):
    """Align all ayah clips, preserving saved ayah marks and manual word edits."""
    from surah_cutter.core import decode, word_align

    pcm = decode.decode(str(audio))
    duration = round(len(pcm) * 1000 / decode.DEFAULT_SR)
    by_ayah = {}
    for word in quran_text.surah_words(surah):
        by_ayah.setdefault(word.ayah, []).append(word)
    written = failures = 0
    with _WORD_LOCK:
        for ayah, words in by_ayah.items():
            timings, _ = _sura_timings_reviews(surah, db_path)
            start = timings.get(ayah)
            end = timings.get(ayah + 1, duration if ayah == quran_text.ayah_count(surah) else None)
            try:
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= duration):
                    raise ValueError("Сначала уточните границы аята.")
                clip = pcm[start * decode.DEFAULT_SR // 1000:end * decode.DEFAULT_SR // 1000]
                rows = word_align.align_ayah_words(clip, decode.DEFAULT_SR, words)
                with _PROJECT_LOCK:
                    current, _ = _sura_timings_reviews(surah, db_path)
                    if (current.get(ayah), current.get(ayah + 1, duration)) != (start, end):
                        raise ValueError("Границы аята изменились во время расчёта.")
                    written += db.write_word_timings(surah, ayah,
                        [(i, start + t, c) for i, t, c in rows], start, end,
                        model_store.source())
                log_event(f"Аят {ayah}: слов {len(words)}, метки сохранены.")
            except ValueError as exc:
                failures += 1
                log_event(f"Аят {ayah}: требуется проверка — {exc}")
    return {"words_written": written, "word_failures": failures}


def _word_progress(db_path: Path | None = None, surah: int | None = None):
    """Count canonical, current word marks without decoding audio for the dashboard."""
    import sqlite3

    db_path = db_path if db_path is not None else STATE.db_path
    saved, timings = {}, {}
    if db_path and db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            where, args = (" WHERE sura=?", (surah,)) if surah else ("", ())
            for s, a, t in conn.execute("SELECT sura, ayah, time FROM timings" + where, args):
                timings.setdefault(s, {})[a] = t
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='word_timings'").fetchone():
                for row in conn.execute(
                    "SELECT sura, ayah, word_index, time, status, ayah_start, ayah_end "
                    "FROM word_timings" + where, args
                ):
                    saved.setdefault(row[0], {})[row[2]] = row
    result = {}
    for s in ([surah] if surah else saved):
        words = quran_text.surah_words(s)
        valid = []
        marks = timings.get(s, {})
        for word in words:
            row = saved.get(s, {}).get(word.index)
            if row is None:
                continue
            _, a, _, time, status, start, end = row
            if (a == word.ayah and start == marks.get(a) and start <= time < end
                    and (end == marks.get(a + 1) or a == quran_text.ayah_count(s))):
                valid.append(status)
        result[s] = {"total": len(words), "marked": len(valid),
                     "edited": valid.count("edited"), "verified": valid.count("verified"),
                     "complete": len(valid) == len(words)}
    return result


def _word_context(surah: int, ayah: int, payload: dict | None = None):
    """Require exact adjacent ayah boundaries, and snapshot project identity."""
    _check_surah(surah)
    _require_db()
    count = quran_text.ayah_count(surah)
    if not 1 <= ayah <= count:
        raise HTTPException(400, "Нет такого аята.")
    with _PROJECT_LOCK:
        db_path = STATE.db_path.resolve()
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, "Нет аудио суры.")
        audio = audio.resolve()
        if payload is not None:
            if payload.get("db_path") != str(db_path):
                raise HTTPException(409, "Проект изменился. Откройте пословный режим заново.")
            if _JOB and _JOB.running:
                raise HTTPException(409, "Дождитесь завершения расчёта суры.")
        timings, _ = _sura_timings_reviews(surah, db_path)
    start = timings.get(ayah)
    end = timings.get(ayah + 1)
    if ayah == count:
        from surah_cutter.core.decode import duration_ms
        end = duration_ms(str(audio))
    if (type(start) is not int or type(end) is not int or start < 0 or end <= start):
        raise HTTPException(400, "Сначала разметьте начало этого и следующего аята.")
    if payload is not None and (payload.get("start_ms"), payload.get("end_ms")) != (start, end):
        raise HTTPException(409, "Границы аята изменились. Откройте пословный режим заново.")
    words = [w for w in quran_text.surah_words(surah) if w.ayah == ayah]
    return db_path, audio, words, start, end


def _word_response(db_path, surah, ayah, words, start, end):
    from surah_cutter.data.project_db import ProjectDB

    with ProjectDB.open(str(db_path), DB_CODE, DB_VERSION) as db:
        saved = {r["word_index"]: r for r in db.get_word_timings(surah, ayah)}
    result = []
    for word in words:
        row = saved.get(word.index, {})
        result.append({**row, "word_index": word.index, "text": word.text,
            "stale": bool(row and (row["ayah_start"], row["ayah_end"]) != (start, end))})
    return {"surah": surah, "ayah": ayah, "db_path": str(db_path),
            "start_ms": start, "end_ms": end, "words": result,
            "word_progress": _word_progress(db_path, surah).get(surah)}


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _fast_trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    """Compact fast-align diagnostics for generation progress UI."""
    ayahs = trace.get("ayahs") or []
    repaired = sum(1 for a in ayahs if a.get("repaired"))
    reliable = sum(1 for a in ayahs if a.get("reliable"))
    tempo = trace.get("tempo_forecast") or {}
    unresolved = {
        event.get("ayah")
        for event in trace.get("pause_sequence_events") or []
        if event.get("action") == "unresolved_current"
    }
    return {
        "coverage": trace.get("coverage"),
        "n_pause": trace.get("n_pause"),
        "n_wasl": trace.get("n_wasl"),
        "n_refine_skipped": trace.get("n_refine_skipped"),
        "reliable": reliable,
        "repaired": repaired,
        "ayahs": len(ayahs),
        "drift_detected": trace.get("drift_detected"),
        "mean_conf": trace.get("mean_conf"),
        "verify_sim": trace.get("verify_sim"),
        "tempo_rate": tempo.get("rate_units_per_ms"),
        "tempo_confidence": tempo.get("confidence"),
        "tempo_observations": tempo.get("observation_count", 0),
        "unresolved": len(unresolved),
        "atomic_wasl_reads": sum(
            1
            for read in trace.get("model_reads") or []
            if read.get("phase") in {
                "wasl-run",
                "wasl-tiled-fallback",
                "wasl-full-fallback",
            }
            and read.get("atomic")
        ),
    }


def _write_trace_confidences(db: Any, surah: int, trace: dict[str, Any]) -> None:
    """Persist per-ayah alignment confidence for UI highlighting."""
    ayahs = trace.get("ayahs") or []
    confidences: dict[int, float | None] = {}
    for item in ayahs:
        ayah = item.get("ayah")
        if ayah is None:
            continue
        conf = item.get("conf")
        confidences[int(ayah)] = float(conf) if conf is not None else None
    if confidences:
        db.update_auto_confidences(surah, confidences)


def _sura_log_folder() -> Path:
    if STATE.db_path is None:
        raise HTTPException(400, "Сначала откройте проект")
    db_path = STATE.db_path.resolve()
    folder = db_path.with_name(f"{db_path.stem}.logs")
    if folder.is_symlink():
        raise HTTPException(400, "Папка логов не должна быть ссылкой")
    return folder


def _save_sura_log(
    surah: int,
    mode: str,
    trace: dict[str, Any],
    log_lines: list[str],
    rows: list[tuple[int, int, int]],
    db_path: Path,
) -> dict[str, str] | None:
    """Persist per-sura realtime logs next to the project DB."""
    if not log_lines:
        return None
    log_dir = db_path.with_name(f"{db_path.stem}.logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    base = f"sura_{surah:03d}_{mode}"
    log_path = log_dir / f"{base}.log"
    json_path = log_dir / f"{base}.json"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "surah": surah,
                "mode": mode,
                "log": log_lines,
                "rows": rows,
                "trace": trace,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"log": str(log_path), "json": str(json_path)}


def _check_surah(surah: int) -> None:
    if not 1 <= surah <= 114:
        raise HTTPException(404, f"sura out of range: {surah}")


def _require_db() -> None:
    if STATE.db_path is None:
        raise HTTPException(400, "open a project DB first")


def _restore_from_settings() -> None:
    """Re-open the last project on startup (if its paths still exist)."""
    cfg = settings.load()
    audio_dir = cfg.get("audio_dir")
    db_path = cfg.get("db_path")
    if audio_dir and Path(audio_dir).is_dir():
        STATE.set_audio_dir(audio_dir)
    if db_path and Path(db_path).exists():
        STATE.set_db_path(db_path)


def _processed_suras() -> set[int]:
    """Suras containing exactly the canonical set of ayahs."""
    if STATE.db_path is None or not Path(STATE.db_path).exists():
        return set()
    import sqlite3

    conn = sqlite3.connect(str(STATE.db_path))
    try:
        rows = conn.execute("SELECT sura, ayah FROM timings").fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    ayahs: dict[int, list[int]] = {}
    for surah, ayah in rows:
        ayahs.setdefault(surah, []).append(ayah)
    return {
        surah for surah, count in quran_text.surah_ayah_counts().items()
        if len(ayahs.get(surah, [])) == count
        and set(ayahs[surah]) == set(range(1, count + 1))
    }


def _reviews_by_sura() -> dict[int, list[str]]:
    """Map sura → list of per-ayah review statuses from the open DB (or empty)."""
    if STATE.db_path is None or not Path(STATE.db_path).exists():
        return {}
    import sqlite3

    out: dict[int, list[str]] = {}
    conn = sqlite3.connect(str(STATE.db_path))
    try:
        rows = conn.execute(
            "SELECT t.sura, COALESCE(r.status, 'auto') FROM timings t "
            "LEFT JOIN review_meta r ON r.sura = t.sura AND r.ayah = t.ayah"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    for sura, status in rows:
        out.setdefault(sura, []).append(status)
    return out


def _sura_timings_reviews(
    surah: int, db_path: Path | None = None,
) -> tuple[dict[int, int], dict[int, dict]]:
    """Return (ayah→time, ayah→review-meta) for one sura from the open DB."""
    timings: dict[int, int] = {}
    reviews: dict[int, dict] = {}
    db_path = db_path if db_path is not None else STATE.db_path
    if db_path is None or not db_path.exists():
        return timings, reviews
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        for ayah, t in conn.execute(
            "SELECT ayah, time FROM timings WHERE sura = ?", (surah,)
        ):
            timings[ayah] = t
        try:
            for ayah, status, conf in conn.execute(
                "SELECT ayah, status, confidence FROM review_meta WHERE sura = ?",
                (surah,),
            ):
                reviews[ayah] = {"status": status, "confidence": conf}
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()
    return timings, reviews


def _format_fast_log(
    surah: int,
    decisions: list[Any],
    trace: dict[str, Any],
    rows: list[tuple[int, int, float]],
) -> str:
    total = len(decisions)
    accepted = sum(1 for d in decisions if d.accepted)
    rejected = total - accepted
    lines = [
        f"Сура {surah}: fast debug log",
        "",
        f"1. Паузы: energy нашёл {total}; DSP/VAD принял {accepted}; отклонил {rejected}.",
    ]
    budget = trace.get("pause_budget") or {}
    if budget:
        lines.append(
            f"   бюджет: границ аятов {budget.get('ayah_boundaries')}, "
            f"waqf {budget.get('waqf_marks')}, "
            f"ожидаемых целей ~{budget.get('expected_pause_targets')}, "
            f"trusted {budget.get('trusted_pauses')}, weak {budget.get('weak_pauses')}, "
            f"active {budget.get('active_pauses')}, "
            f"activated_low_conf {budget.get('activated_low_conf_pauses')}, "
            f"использовано {budget.get('used_pauses')}, "
            f"прочитано пауз {budget.get('decoded_pauses')}, "
            f"micro {budget.get('micro_pauses')}, ratio={budget.get('pause_target_ratio')}"
        )
    tempo = trace.get("tempo_forecast") or {}
    if tempo:
        lines.append(
            f"   local tempo: rate={tempo.get('rate_units_per_ms')} "
            f"conservative={tempo.get('conservative_rate')} "
            f"spread={tempo.get('spread_units_per_ms')} "
            f"confidence={tempo.get('confidence')} "
            f"observations={tempo.get('observation_count')} "
            f"pause={tempo.get('pause_median_ms')}±{tempo.get('pause_iqr_ms')}ms"
        )
    text_plans = trace.get("text_disambiguation") or []
    unresolved_plans = [plan for plan in text_plans if plan.get("reason") == "unresolved"]
    if unresolved_plans:
        lines.append(
            "   unresolved text → atomic wasl: "
            + ", ".join(str(plan.get("ayah")) for plan in unresolved_plans)
        )
    for i, d in enumerate(decisions, 1):
        status = "OK" if d.accepted else "BAD"
        vad = "—" if d.vad_ok is None else ("ok" if d.vad_ok else "bad")
        lines.append(
            f"   {i:02d}. {status} {d.pause.start_ms:.0f}-{d.pause.end_ms:.0f} "
            f"mid={d.pause.center_ms:.0f} depth={d.pause.depth_ms:.0f} "
            f"reason={d.reason} dsp={d.dsp_ok} vad={vad} "
            f"flat={d.flatness:.3f} zcr={d.zcr:.3f} rms={d.rms_ratio:.3f}"
        )

    lines += [
        "",
        "2. Последовательный анализ пауз:",
    ]
    events = trace.get("pause_sequence_events") or []
    shown = 0
    extra_count = sum(1 for e in events if e.get("action") == "extra")
    for e in events:
        if e.get("action") in ("extra", "early_pool"):
            continue
        shown += 1
        pause = e.get("pause_idx", "—")
        ayah = e.get("ayah", "—")
        if e.get("action") == "future_gap":
            lines.append(
                f"   pause {pause}: слышен аят {e.get('heard_ayah')} вместо {ayah}; "
                f"heard_sim={e.get('heard_sim')} current_sim={e.get('current_sim')} "
                f"tempo={e.get('tempo_score')}"
            )
        elif e.get("action") == "unresolved_current":
            lines.append(
                f"   pause {pause}: unresolved ayah={ayah}; "
                f"лучше слышен {e.get('observed_ayah')} "
                f"direction={e.get('direction')} sim={e.get('observed_sim')} "
                f"best_current={e.get('best_sim')}"
            )
        elif e.get("action") in ("confirmed", "recovered", "future_anchor"):
            lines.append(
                f"   pause {pause}: {e.get('action')} ayah={ayah} "
                f"sim={e.get('sim')} tempo={e.get('tempo_score')} "
                f"delta={e.get('expected_delta_ms')}ms"
            )
        elif e.get("action") == "wasl":
            lines.append(
                f"   ayah {ayah}: wasl/local-align reason={e.get('reason')} "
                f"best_sim={e.get('best_sim')}"
            )
    if extra_count:
        lines.append(f"   internal/extra pauses skipped: {extra_count}")
    micro = trace.get("micro_pauses") or []
    for item in micro:
        lines.append(
            f"   micro ayah={item.get('ayah')} status={item.get('status')} "
            f"expected={item.get('expected_ms')} center={item.get('center_ms')} "
            f"rms={item.get('rms_ratio')}"
        )
    if not shown and not extra_count:
        lines.append("   нет событий")

    lines += [
        "",
        "3. Нейро-чтение окон:",
    ]
    reads = trace.get("model_reads") or []
    for i, r in enumerate(reads, 1):
        w0, w1 = r.get("window_ms", ["?", "?"])
        ayah = r.get("ayah")
        delta = r.get("delta_ms")
        delta_s = "" if delta is None else f" delta={delta:+.0f}ms"
        words = r.get("words") or ""
        if len(words) > 160:
            words = words[:157] + "..."
        lines.append(
            f"   {i:02d}. {r.get('phase')} ayah={ayah} window={w0}-{w1} "
            f"target={r.get('target', '')} onset={r.get('onset_ms')} "
            f"conf={r.get('conf')}{delta_s} edge={r.get('near_edge', False)}"
        )
        lines.append(f"       words: {words}")

    lines += [
        "",
        "4. Соотношение пауз и аятов:",
    ]
    by_ayah = {t.get("ayah"): t for t in trace.get("ayahs", [])}
    by_match = {m.get("ayah"): m for m in trace.get("pause_matches", [])}
    for ayah, time_ms, conf in rows:
        t = by_ayah.get(ayah, {})
        m = by_match.get(ayah, {})
        src = "pause" if t.get("from_pause") else ("anchor" if ayah == 1 else "wasl/interp")
        lines.append(
            f"   ayah {ayah:03d}: {time_ms}ms conf={conf:.3f} src={src} "
            f"rough={t.get('rough_ms')} onset={t.get('onset_ms')} "
            f"win={t.get('win_s')} reliable={t.get('reliable')} "
            f"tempo={m.get('tempo_score')} span_prev={m.get('prev_span_ratio')} "
            f"span_next={m.get('next_span_ratio')} waqf={m.get('waqf_before')}/{m.get('waqf_after')}"
        )

    lines += [
        "",
        "5. Статистика нейро-чтения:",
        f"   прочитано нейронкой: {trace.get('audio_ms', 0) / 1000.0:.2f}s",
        f"   длительность суры: {trace.get('total_ms', 0) / 1000.0:.2f}s",
        f"   покрытие окнами: {trace.get('coverage')}",
        f"   чтений окон: {len(reads)}",
        f"   refine пропущен по подтверждённым паузам: {trace.get('n_refine_skipped')}",
        f"   verify extra reads: {trace.get('verify_extra_reads')}",
        "",
        f"coverage={trace.get('coverage')} audio_ms_read={round(trace.get('audio_ms', 0))} "
        f"total_ms={round(trace.get('total_ms', 0))} n_pause={trace.get('n_pause')} "
        f"n_wasl={trace.get('n_wasl')}",
    ]
    return "\n".join(lines)


# Convenience for ASGI servers / tests.
app = create_app()
