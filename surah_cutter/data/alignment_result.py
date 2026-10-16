"""Persist one generated surah with its provenance, confidence and trace."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class AlignmentSaveResult:
    """Committed outcome; retain legacy ``written, paths = result`` usage."""

    written: int
    log_paths: dict[str, str] | None
    outcome: str = "saved"
    diagnostics_error: str | None = None
    persisted_rows: tuple[tuple[int, int, int], ...] = ()
    persisted_review: tuple[dict, ...] = ()
    unresolved_ayahs: tuple[int, ...] = ()

    def __iter__(self):
        return iter((self.written, self.log_paths))

    def __getitem__(self, index):
        return (self.written, self.log_paths)[index]

    def __len__(self):
        return 2


def result_source(trace: dict[str, Any], requested_mode: str) -> str:
    """Keep model identity and the requested/actual algorithm in review_meta."""
    from surah_cutter.core import model_store
    from surah_cutter.data.alignment_identity import trace_run_key

    model = str(trace.get("model_source") or model_store.source())
    actual = str(trace.get("auto_mode") or trace.get("mode") or requested_mode)
    if trace.get("result_contract") == "partial-v1":
        if trace.get("mode") not in (None, "fast"):
            raise ValueError("Partial FAST contract disagrees with actual mode")
        actual = "fast"
    backend = trace.get("backend")
    if backend is None:
        backend = next((part.partition("=")[2] for part in model.split(":")
                        if part.startswith("backend=")), None)
    if backend is None:
        backend = ("onnx" if model.startswith("onnx-ctc:") else
                   "runtime-worker" if model.startswith(("ctc:", "attention-dtw:")) else
                   requested_mode if requested_mode in ("fingerprint", "hybrid") else "unknown")
    align_lag = trace.get("align_lag_ms")
    if align_lag is None:
        align_lag = (trace.get("run_snapshot") or {}).get("lag_ms")
    if align_lag is None:
        align_lag = model_store.lag_ms()
    adjust = trace.get("lag_adjust_ms", 0)
    source = (f"{model}:backend={backend}:requested_mode={requested_mode}"
              f":actual_mode={actual}:align_lag_ms={align_lag}:lag_adjust_ms={adjust}")
    full = trace.get("full_windows")
    if isinstance(full, dict) and full.get("algorithm"):
        source += (f":full_policy={full['algorithm']}:full_window_s={full['window_s']}"
                   f":full_workers={full['workers']}:full_context_s={full['context_s']}"
                   f":full_pause_policy={full['pause_policy']}"
                   f":full_max_window_s={full['max_window_s']}")
        if full.get("review_policy"):
            source += f":full_review_policy={full['review_policy']}"
        if full.get("onset_policy"):
            source += f":full_onset_policy={full['onset_policy']}"
    if trace.get("full_boundary_verifier_source"):
        source += f":full_boundary_verifier={trace['full_boundary_verifier_source']}"
    if trace.get("result_contract") == "partial-v1":
        source += ":fast_result=partial-v1"
    key = trace_run_key(trace)
    return f"{source}:fast_run_key={key}" if key is not None else source


def write_alignment_result(
    db: Any,
    rows: Iterable[Sequence[int]],
    *,
    trace: dict[str, Any],
    requested_mode: str,
    log_lines: list[str] | None = None,
    db_path: str | Path | None = None,
    overwrite_manual: bool = False,
) -> AlignmentSaveResult:
    """Write auto rows and confidences, then save a reproducible JSON trace.

    Human rows stay protected unless overwrite_manual was explicitly chosen.
    The timing DB transaction completes before the diagnostic sidecar is written.
    """
    from surah_cutter.core.cancellation import GenerationCancelled, check_cancelled

    check_cancelled()
    materialized = [tuple(row) for row in rows]
    partial_fast = trace.get("result_contract") == "partial-v1"
    unresolved = tuple(trace.get("unresolved_ayahs", ())) if partial_fast else ()
    if not materialized and not partial_fast:
        return AlignmentSaveResult(0, None)
    # Re-read manual statuses under the writer lock, using the original model
    # proposal rather than a potentially stale pipeline overlay.
    model_rows = [tuple(row) for row in trace.get("generated_rows", materialized)]
    model_keys = {(r[0], r[1]) for r in model_rows}
    output_keys = {(r[0], r[1]) for r in materialized}
    protected_keys = {(r[0], r[1]) for r in trace.get("protected_rows", ())} if partial_fast else set()
    if model_keys | protected_keys != output_keys:
        raise ValueError("Model result and output ayah keys disagree")
    surahs = {row[0] for row in materialized}
    if partial_fast:
        surahs.add(trace.get("surah"))
    if len(surahs) != 1 or None in surahs:
        raise ValueError("Alignment result must contain exactly one surah")
    surah = next(iter(surahs))
    if partial_fast:
        from surah_cutter.data import quran_text
        expected = set(range(1, quran_text.ayah_count(surah) + 1))
        predicted = {r[1] for r in model_rows}
        if (predicted & set(unresolved) or predicted | set(unresolved) != expected
                or len(unresolved) != len(set(unresolved))):
            raise ValueError("Incomplete partial FAST result contract")
    confidences: dict[int, float | None] = {}
    for item in trace.get("ayahs", ()):
        if item.get("ayah") is not None:
            value = item.get("conf")
            confidences[int(item["ayah"])] = float(value) if value is not None else None
    if partial_fast:
        import math
        checks = {c.get("ayah"): c for c in trace.get("ayahs", ())}
        for _, ayah, ms in model_rows:
            c = confidences.get(ayah)
            if (type(c) not in (int, float) or not math.isfinite(c) or not 0 < c <= 1
                    or checks[ayah].get("onset_ms") != ms or not checks[ayah].get("reliable")):
                raise ValueError("Partial FAST row lacks a resolved boundary")
    source = result_source(trace, requested_mode)
    staged: list[tuple[Path, Path]] = []
    log_paths: dict[str, str] | None = None
    history = None
    # Preflight paths before mutating SQLite. All diagnostic content is written
    # after commit so its effective rows come from the locked merge snapshot.
    if db_path is not None:
        folder_path = Path(db_path)
        folder = folder_path.with_name(f"{folder_path.stem}.logs")
        if folder.is_symlink():
            raise ValueError("Log folder must not be a symlink")
        folder.mkdir(parents=True, exist_ok=True)
        safe_mode = requested_mode if requested_mode.isalnum() else "alignment"
        base = f"sura_{surah:03d}_{safe_mode}"
        json_path = folder / f"{base}.json"
        log_path = folder / f"{base}.log"
        log_paths = {"json": str(json_path), "log": str(log_path)}
        if requested_mode in ("fast", "full"):
            from surah_cutter.data.alignment_runs import runs_folder

            history = runs_folder(folder)
            history.mkdir(parents=True, exist_ok=True)
            run_id = f"{base}_{uuid.uuid4().hex}"
            snapshot_path = history / f"{run_id}.json"
            log_paths["run"] = str(snapshot_path)

    persisted: list[dict] = []
    written = db.write_timings(
        model_rows, source=source, confidences=confidences,
        overwrite_manual=overwrite_manual,
        duration_ms=trace.get("duration_ms", trace.get("total_ms")), persisted=persisted,
        **({"replace_fast_sura": surah} if partial_fast else {}),
    )
    persisted_rows = tuple((r["sura"], r["ayah"], r["time"]) for r in persisted)
    if db_path is None:
        return AlignmentSaveResult(written, None, persisted_rows=persisted_rows,
                                   persisted_review=tuple(persisted),
                                   unresolved_ayahs=unresolved)
    error = None
    try:
        payload = {
            "surah": surah, "mode": requested_mode,
            "log": log_lines or [], "rows": persisted_rows, "trace": trace,
            "model_rows": model_rows, "persisted_review": persisted,
            "source": source,
            "overwrite_manual": overwrite_manual, "unresolved_ayahs": unresolved,
        }
        staged.append((_stage_text(json_path, json.dumps(
            payload, ensure_ascii=False, indent=2, default=str,
        ) + "\n"), json_path))
        staged.append((_stage_text(log_path, "\n".join(log_lines or []) +
                                   ("\n" if log_lines else "")), log_path))
        if history is not None:
            if (json_path.is_file() and not json_path.is_symlink()
                    and not any(history.glob(f"{base}_*.json"))):
                try:
                    previous = json.loads(json_path.read_text(encoding="utf-8"))
                    previous["created_ms"] = int(json_path.stat().st_mtime * 1000)
                    previous_path = history / f"{base}_{uuid.uuid4().hex}.json"
                    staged.append((_stage_text(previous_path, json.dumps(
                        previous, ensure_ascii=False, indent=2, default=str,
                    ) + "\n"), previous_path))
                except (ValueError, TypeError):
                    pass  # A broken legacy log must not prevent a new alignment.
            snapshot = {
                "version": 2, "id": run_id, "created_ms": time.time_ns() // 1_000_000,
                "surah": surah, "mode": requested_mode, "source": source,
                "rows": model_rows, "persisted_rows": persisted_rows,
                "persisted_review": persisted,
                "raw_model_result": "generated_rows" in trace,
                "confidences": confidences, "audio_signature": trace.get("audio_signature"),
                # Immutable history must retain evidence and effective settings,
                # not just a hash after the per-mode latest log is overwritten.
                "trace": trace,
            }
            staged.append((_stage_text(snapshot_path, json.dumps(
                snapshot, ensure_ascii=False, indent=2, default=str,
            ) + "\n"), snapshot_path))
        # Publish history before replacing the per-mode latest log.
        for temporary, destination in sorted(staged, key=lambda pair: pair[1].parent.name != "runs"):
            os.replace(temporary, destination)
    except (Exception, GenerationCancelled) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except (OSError, GenerationCancelled) as exc:
                error = error or f"{type(exc).__name__}: {exc}"
    return AlignmentSaveResult(
        written, log_paths, "saved_with_diagnostics_error" if error else "saved",
        error, persisted_rows, tuple(persisted), unresolved,
    )


def _stage_text(path: Path, content: str) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
        return Path(name)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def write_unsaved_diagnostics(
    *,
    db_path: str | Path,
    surah: int,
    requested_mode: str,
    trace: dict[str, Any],
    outcome: str,
    error: BaseException,
    log_lines: list[str] | None = None,
) -> Path | None:
    """Best-effort failed/cancelled trace, never a successful alignment run.

    Called while handling the original exception. Diagnostic failures must not
    replace that exception or create a successful snapshot in ``runs``.
    """
    temporary = None
    try:
        if outcome not in ("failed", "cancelled"):
            return None
        path = Path(db_path)
        folder = path.with_name(f"{path.stem}.logs")
        if folder.is_symlink():
            return None
        folder.mkdir(parents=True, exist_ok=True)
        failures = folder / "failures"
        if failures.is_symlink():
            return None
        failures.mkdir(exist_ok=True)
        mode = requested_mode if requested_mode.isalnum() else "alignment"
        destination = failures / f"sura_{surah:03d}_{mode}_{outcome}_{uuid.uuid4().hex}.json"
        payload = {
            "surah": surah, "mode": requested_mode, "outcome": outcome,
            "saved": False, "error": f"{type(error).__name__}: {error}",
            "trace": trace, "log": log_lines or [],
        }
        temporary = _stage_text(destination, json.dumps(
            payload, ensure_ascii=False, indent=2, default=str,
        ) + "\n")
        os.replace(temporary, destination)
        return destination
    except BaseException:
        # Even cancellation raised by a diagnostic hook must leave the original
        # model/cancellation exception in charge of orchestration.
        return None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException:
                pass
