"""FastAPI app for the web GUI: JSON API + SSE progress + audio/static serving.

All heavy work delegates to :mod:`surah_cutter.core` / :mod:`surah_cutter.data`;
this layer only translates HTTP ↔ project state (:mod:`surah_cutter.webgui.state`).
Generation runs in a worker thread and streams per-sura progress over SSE.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from surah_cutter.data import quran_text
from surah_cutter.webgui import settings
from surah_cutter.webgui.state import (
    DB_CODE,
    DB_VERSION,
    STATE,
    classify_status,
    waveform_peaks,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
#: Bundled Qur'an (QPC Hafs Uthmani) font, served at /fonts for the review text.
_FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_GEN_MODES = {"full", "fast", "hybrid"}


# --- generation job (one at a time) ---------------------------------------


class _GenJob:
    """A running generation pass; streams events to any number of SSE clients."""

    def __init__(self) -> None:
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
        from surah_cutter.core.pipeline import process_surah
        from surah_cutter.data.project_db import ProjectDB

        cancelled = False
        completed = 0
        try:
            self._emit({"type": "start", "total": len(suras)})
            db = ProjectDB.open(str(STATE.db_path), DB_CODE, DB_VERSION)
            try:
                for i, surah in enumerate(suras):
                    # Cancellation is checked between suras (a sura's alignment
                    # is not interruptible mid-flight).
                    if self.cancel.is_set():
                        cancelled = True
                        break
                    self._emit({"type": "sura_start", "surah": surah, "index": i})
                    audio = STATE.audio_for(surah)
                    if audio is None:
                        self._emit(
                            {"type": "sura", "surah": surah, "index": i,
                             "written": 0, "skipped": "no-audio"}
                        )
                        completed = i + 1
                        continue
                    trace: dict[str, Any] = {}
                    rows = process_surah(
                        str(audio), surah, reciter=reciter, mode=mode,
                        project_db=db,
                        trace=trace if mode == "fast" else None,
                    )
                    written = db.write_timings(rows, source="auto")
                    completed = i + 1
                    event = {
                        "type": "sura", "surah": surah, "index": i,
                        "ayahs": len(rows), "written": written,
                    }
                    if trace:
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


def create_app() -> FastAPI:
    app = FastAPI(title="surah-cutter")
    _restore_from_settings()

    # -- project ----------------------------------------------------------
    @app.get("/api/project")
    def get_project() -> dict[str, Any]:
        cfg = settings.load()
        return {
            "audio_dir": str(STATE.audio_dir) if STATE.audio_dir else None,
            "db_path": str(STATE.db_path) if STATE.db_path else None,
            "audio_count": len(STATE.audio_map),
            "suras_with_audio": sorted(STATE.audio_map),
            "mode": cfg.get("mode", "hybrid"),
        }

    @app.post("/api/project/open")
    def open_project(payload: dict[str, str] = Body(...)) -> dict[str, Any]:
        audio_dir = payload.get("audio_dir")
        db_path = payload.get("db_path")
        if audio_dir:
            if not Path(audio_dir).is_dir():
                raise HTTPException(400, f"not a folder: {audio_dir}")
            STATE.set_audio_dir(audio_dir)
        if db_path:
            STATE.set_db_path(db_path)
            from surah_cutter.data.project_db import ProjectDB

            ProjectDB.open(db_path, DB_CODE, DB_VERSION).close()  # ensure schema
        # Remember the project so the app reopens here next launch.
        settings.update(
            audio_dir=str(STATE.audio_dir) if STATE.audio_dir else None,
            db_path=str(STATE.db_path) if STATE.db_path else None,
        )
        return get_project()

    @app.post("/api/settings")
    def set_settings(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        allowed = {k: v for k, v in payload.items() if k in ("mode",)}
        if "mode" in allowed and allowed["mode"] not in _GEN_MODES:
            raise HTTPException(400, f"unknown mode: {allowed['mode']}")
        return settings.update(**allowed)

    # -- suras + statuses -------------------------------------------------
    @app.get("/api/suras")
    def list_suras() -> list[dict[str, Any]]:
        counts = quran_text.surah_ayah_counts()
        reviews = _reviews_by_sura()
        out = []
        for s in range(1, 115):
            ac = counts.get(s, 0)
            out.append({
                "surah": s,
                "ayah_count": ac,
                "has_audio": s in STATE.audio_map,
                "status": classify_status(reviews.get(s), ac),
            })
        return out

    # -- generation -------------------------------------------------------
    @app.post("/api/generate")
    def generate(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        global _JOB
        if STATE.db_path is None:
            raise HTTPException(400, "open a project DB first")
        if _JOB is not None and _JOB.running:
            raise HTTPException(409, "generation already running")
        suras = payload.get("suras") or sorted(STATE.audio_map)
        suras = [s for s in suras if s in STATE.audio_map]
        # Resume support: skip suras already generated (per-sura atomic write →
        # any timings for a sura means it finished). Lets a cancelled/crashed run
        # be continued after restart without redoing completed suras.
        if payload.get("only_missing"):
            done = _processed_suras()
            suras = [s for s in suras if s not in done]
        if not suras:
            raise HTTPException(400, "no suras to process (all done or no audio)")
        mode = payload.get("mode", "hybrid")
        if mode not in _GEN_MODES:
            raise HTTPException(400, f"unknown mode: {mode}")
        reciter = payload.get("reciter")
        settings.update(mode=mode)  # remember last-used mode
        _JOB = _GenJob()
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

    @app.get("/api/audio/{surah}")
    def get_audio(surah: int) -> FileResponse:
        _check_surah(surah)
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        return FileResponse(str(audio))

    # -- edits ------------------------------------------------------------
    @app.post("/api/timing")
    def set_timing(payload: dict[str, int] = Body(...)) -> dict[str, Any]:
        surah, ayah, time = payload["surah"], payload["ayah"], payload["time"]
        _check_surah(surah)
        _require_db()
        from surah_cutter.data.project_db import ProjectDB

        with ProjectDB.open(str(STATE.db_path), DB_CODE, DB_VERSION) as db:
            db.set_time(surah, ayah, int(time))
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


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _fast_trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    """Compact fast-align diagnostics for generation progress UI."""
    ayahs = trace.get("ayahs") or []
    repaired = sum(1 for a in ayahs if a.get("repaired"))
    reliable = sum(1 for a in ayahs if a.get("reliable"))
    return {
        "coverage": trace.get("coverage"),
        "n_pause": trace.get("n_pause"),
        "n_wasl": trace.get("n_wasl"),
        "reliable": reliable,
        "repaired": repaired,
        "ayahs": len(ayahs),
    }


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
    """Suras that already have any timings in the open DB (i.e. finished)."""
    if STATE.db_path is None or not Path(STATE.db_path).exists():
        return set()
    import sqlite3

    conn = sqlite3.connect(str(STATE.db_path))
    try:
        rows = conn.execute("SELECT DISTINCT sura FROM timings").fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _reviews_by_sura() -> dict[int, list[str]]:
    """Map sura → list of per-ayah review statuses from the open DB (or empty)."""
    if STATE.db_path is None or not Path(STATE.db_path).exists():
        return {}
    import sqlite3

    out: dict[int, list[str]] = {}
    conn = sqlite3.connect(str(STATE.db_path))
    try:
        rows = conn.execute("SELECT sura, status FROM review_meta").fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    for sura, status in rows:
        out.setdefault(sura, []).append(status)
    return out


def _sura_timings_reviews(surah: int) -> tuple[dict[int, int], dict[int, dict]]:
    """Return (ayah→time, ayah→review-meta) for one sura from the open DB."""
    timings: dict[int, int] = {}
    reviews: dict[int, dict] = {}
    if STATE.db_path is None or not Path(STATE.db_path).exists():
        return timings, reviews
    import sqlite3

    conn = sqlite3.connect(str(STATE.db_path))
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


# Convenience for ASGI servers / tests.
app = create_app()
