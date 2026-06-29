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
_GEN_MODES = {"auto", "full", "fast", "hybrid", "tiled"}


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

                    rows = process_surah(
                        str(audio), surah, reciter=reciter, mode=mode,
                        project_db=db,
                        trace=trace if mode in ("auto", "fast", "tiled") else None,
                        log_event=log_event,
                    )
                    written = db.write_timings(rows, source="auto")
                    _write_trace_confidences(db, surah, trace)
                    log_paths = _save_sura_log(surah, mode, trace, log_lines, rows)
                    completed = i + 1
                    event = {
                        "type": "sura", "surah": surah, "index": i,
                        "ayahs": len(rows), "written": written,
                    }
                    if log_paths:
                        event["log_paths"] = log_paths
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
            "mode": cfg.get("mode", "auto"),
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
        mode = payload.get("mode", "auto")
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

    @app.get("/api/sura/{surah}/recheck")
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
        _require_db()
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        from surah_cutter.core import decode, draft
        from surah_cutter.data.project_db import ProjectDB

        timings, reviews = _sura_timings_reviews(surah)
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
        with ProjectDB.open(str(STATE.db_path), DB_CODE, DB_VERSION) as db:
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
        _require_db()
        audio = STATE.audio_for(surah)
        if audio is None:
            raise HTTPException(404, f"no audio for sura {surah}")
        mode = payload.get("mode", "tiled")
        if mode not in ("auto", "full", "fast", "tiled"):
            raise HTTPException(400, f"unknown correction mode: {mode}")
        from surah_cutter.core.pipeline import process_surah
        from surah_cutter.data.project_db import ProjectDB

        with ProjectDB.open(str(STATE.db_path), DB_CODE, DB_VERSION) as db:
            rows = process_surah(str(audio), surah, mode=mode, project_db=db)
            written = db.write_timings(rows, source=f"correct-{mode}")
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

    @app.get("/api/sura/{surah}/fast-log")
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
        "n_refine_skipped": trace.get("n_refine_skipped"),
        "reliable": reliable,
        "repaired": repaired,
        "ayahs": len(ayahs),
        "drift_detected": trace.get("drift_detected"),
        "mean_conf": trace.get("mean_conf"),
        "verify_sim": trace.get("verify_sim"),
    }


def _write_trace_confidences(db: Any, surah: int, trace: dict[str, Any]) -> None:
    """Persist per-ayah fast confidence for UI highlighting."""
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


def _save_sura_log(
    surah: int,
    mode: str,
    trace: dict[str, Any],
    log_lines: list[str],
    rows: list[tuple[int, int, int]],
) -> dict[str, str] | None:
    """Persist per-sura realtime logs next to the project DB."""
    if STATE.db_path is None or not log_lines:
        return None
    db_path = Path(STATE.db_path)
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
