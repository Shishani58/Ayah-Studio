"""Read immutable alignment snapshots and the previous per-mode JSON logs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def runs_folder(log_folder: Path) -> Path:
    folder = log_folder / "runs"
    if log_folder.is_symlink() or folder.is_symlink():
        raise ValueError("Папка результатов не должна быть ссылкой")
    return folder


def load_run(log_folder: Path, surah: int, run_id: str) -> dict[str, Any]:
    legacy = run_id in ("legacy-fast", "legacy-full")
    if legacy:
        path = log_folder / f"sura_{surah:03d}_{run_id[7:]}.json"
    elif re.fullmatch(rf"sura_{surah:03d}_(fast|full)_[0-9a-f]{{32}}", run_id):
        path = runs_folder(log_folder) / f"{run_id}.json"
    else:
        raise ValueError("Результат не найден")
    if log_folder.is_symlink() or path.is_symlink():
        raise ValueError("Результат не должен быть ссылкой")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("surah") != surah or payload.get("mode") not in ("fast", "full"):
        raise ValueError("Некорректный результат")
    from surah_cutter.data.quran_text import ayah_count

    count = ayah_count(surah)
    scores = payload.get("confidences", {})
    if "confidences" not in payload:
        scores = {str(a["ayah"]): a.get("conf") for a in payload.get("trace", {}).get("ayahs", []) if "ayah" in a}
    rows = []
    seen = set()
    from surah_cutter.data.onset_evidence import onset_evidence

    trace = payload.get("trace", {})
    checks = {c["ayah"]: c for c in trace.get("ayahs", []) if isinstance(c, dict) and "ayah" in c}
    persisted = {r["ayah"]: r for r in payload.get("persisted_review", [])}
    for row in payload.get("rows", []):
        if (not isinstance(row, (list, tuple)) or len(row) != 3
                or any(type(v) is not int for v in row)
                or row[0] != surah or not 1 <= row[1] <= count
                or row[2] < 0 or row[1] in seen):
            raise ValueError("Некорректные времена результата")
        seen.add(row[1])
        score = scores.get(str(row[1]))
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            score = None
        evidence = onset_evidence(checks.get(row[1]), row[2], score, trace=trace)
        if not payload.get("raw_model_result") and persisted.get(row[1], {}).get("source") == "manual":
            evidence = None
        item = {"ayah": row[1], "time": row[2], "confidence": score}
        if evidence is not None:
            item["onset_evidence"] = evidence
        rows.append(item)
    partial = trace.get("result_contract") == "partial-v1"
    unresolved = list(trace.get("unresolved_ayahs", ())) if partial else []
    model_seen = {row[1] for row in payload.get("model_rows", payload.get("rows", []))}
    if partial and (set(unresolved) & model_seen or set(unresolved) | model_seen != set(range(1, count + 1))):
        raise ValueError("Некорректный частичный результат")
    if not rows and not partial:
        raise ValueError("Пустой результат")
    rows.sort(key=lambda item: item["ayah"])
    if any(a["time"] >= b["time"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Нарушен порядок времён результата")
    return {
        "id": run_id, "surah": surah, "mode": payload["mode"],
        "created_ms": int(payload.get("created_ms", path.stat().st_mtime * 1000)),
        "source": str(payload.get("source") or payload.get("trace", {}).get("model_source") or ""),
        "audio_signature": payload.get("audio_signature"), "legacy": legacy,
        "raw_model_result": payload.get("raw_model_result", False), "ayahs": rows,
        "result_complete": not unresolved, "unresolved_ayahs": unresolved,
    }


def list_runs(log_folder: Path, surah: int) -> list[dict[str, Any]]:
    folder = runs_folder(log_folder)
    ids = [p.stem for p in folder.glob(f"sura_{surah:03d}_*.json") if p.is_file() and not p.is_symlink()]
    # Legacy logs remain available until a successful new snapshot of that mode.
    for mode in ("fast", "full"):
        if not any(run_id.startswith(f"sura_{surah:03d}_{mode}_") for run_id in ids):
            ids.append(f"legacy-{mode}")
    runs = []
    for run_id in ids:
        try:
            run = load_run(log_folder, surah, run_id)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            continue
        run["ayah_count"] = len(run.pop("ayahs"))
        runs.append(run)
    return sorted(runs, key=lambda item: (item["created_ms"], item["id"]), reverse=True)
