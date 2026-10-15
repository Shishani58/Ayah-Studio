"""Alignment provenance and conservative resume checks, without selecting a model.

The caller must supply the identity of the prepared adapter/bundle and effective
settings. Reading ACTIVE_ID here would describe a selection, not the model that
actually produced the result. Unknown identity never authorizes skipping auto rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from surah_cutter.data.quran_text import surah_ayah_counts

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED = {
    "algorithm", "adapter", "model", "revision", "policy", "lag_ms",
    "text_sha256", "pcm_sha256", "sample_rate", "dsp_version", "decode",
    "normalization", "limits",
}


def fast_run_key(snapshot: Mapping) -> str:
    """Hash the complete captured execution identity (including effective env).

    Audio identity is the decoded PCM digest; text identity includes canonical
    words and their model representation. Additional settings are included too.
    The JSON encoding matches the experimental sequential executor's v2 key.
    """
    if not isinstance(snapshot, Mapping) or not _REQUIRED <= snapshot.keys():
        raise ValueError("Incomplete FAST execution identity")
    for field in ("algorithm", "adapter", "model", "revision", "dsp_version", "decode",
                  "normalization"):
        value = snapshot[field]
        if not isinstance(value, str) or not value.strip() or value.lower() == "unknown":
            raise ValueError(f"Unknown FAST identity: {field}")
    for field in ("text_sha256", "pcm_sha256"):
        if not isinstance(snapshot[field], str) or not _SHA256.fullmatch(snapshot[field]):
            raise ValueError(f"Invalid FAST digest: {field}")
    if type(snapshot["sample_rate"]) is not int or snapshot["sample_rate"] <= 0:
        raise ValueError("Invalid FAST sample rate")
    if type(snapshot["lag_ms"]) not in (int, float):
        raise ValueError("Invalid FAST lag")
    for field in ("policy", "limits"):
        if not isinstance(snapshot[field], Mapping) or not snapshot[field]:
            raise ValueError(f"Missing effective FAST settings: {field}")
    try:
        content = json.dumps(dict(snapshot), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("FAST identity must contain finite JSON values") from exc
    return hashlib.sha256(content.encode()).hexdigest()


def trace_run_key(trace: Mapping) -> str | None:
    """Legacy traces remain readable but cannot claim a current execution key."""
    declared = trace.get("fast_run_key")
    if declared is None:
        return None
    snapshot = trace.get("run_snapshot")
    if not isinstance(snapshot, Mapping) or declared != fast_run_key(snapshot):
        raise ValueError("FAST run key disagrees with its execution snapshot")
    return declared


def fast_resume_suras(
    db_path: str | Path | None,
    expected_keys: Mapping[int, str | None] | None = None,
    *,
    counts: dict[int, int] | None = None,
) -> set[int]:
    """Skip only complete protected surahs or auto rows with exact current keys.

    Without a prepared execution identity, only fully protected surahs qualify.
    No audio/model loading or hidden legacy key inference occurs in this check.
    """
    if db_path is None or not Path(db_path).is_file():
        return set()
    with sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True) as conn:
        try:
            rows = conn.execute(
                "SELECT t.sura, t.ayah, t.time, r.status, r.source FROM timings t "
                "LEFT JOIN review_meta r ON r.sura=t.sura AND r.ayah=t.ayah "
                "ORDER BY t.sura, t.ayah"
            ).fetchall()
        except sqlite3.OperationalError:
            return set()
    expected = counts if counts is not None else surah_ayah_counts()
    grouped: dict[int, list] = {}
    invalid_suras = set()
    for surah, ayah, time, status, source in rows:
        if type(surah) is int:
            grouped.setdefault(surah, []).append((ayah, time, status, source))
        else:
            try:
                invalid_suras.add(int(surah))
            except (TypeError, ValueError, OverflowError):
                pass
    done = set()
    for surah, items in grouped.items():
        count = expected.get(surah)
        if surah in invalid_suras or count is None or [r[0] for r in items] != list(range(1, count + 1)):
            continue
        times = [r[1] for r in items]
        if (any(type(r[0]) is not int for r in items)
                or any(type(t) is not int or t < 0 for t in times)
                or any(a >= b for a, b in zip(times, times[1:]))):
            continue
        auto_sources = [source for _, _, status, source in items
                        if status not in ("edited", "verified")]
        if not auto_sources:
            done.add(surah)
            continue
        current = (expected_keys or {}).get(surah)
        if not isinstance(current, str) or not _SHA256.fullmatch(current):
            continue
        keys = []
        for source in auto_sources:
            parts = source.split(":") if isinstance(source, str) else []
            saved_keys = [part.partition("=")[2] for part in parts if part.startswith("fast_run_key=")]
            modes = [part.partition("=")[2] for part in parts if part.startswith("requested_mode=")]
            keys.append(saved_keys[0] if len(saved_keys) == 1 and modes == ["fast"] else None)
        if all(key == current for key in keys):
            done.add(surah)
    return done


def full_resume_suras(db_path, *, model_id: str, onset_policy: str | None = None, counts=None) -> set[int]:
    """Do not reuse old FULL auto rows after a model/boundary-policy change."""
    if db_path is None or not Path(db_path).is_file():
        return set()
    with sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True) as conn:
        try:
            rows = conn.execute(
                "SELECT t.sura,t.ayah,t.time,r.status,r.source FROM timings t "
                "LEFT JOIN review_meta r ON r.sura=t.sura AND r.ayah=t.ayah "
                "ORDER BY t.sura,t.ayah"
            ).fetchall()
        except sqlite3.OperationalError:
            return set()
    expected = counts if counts is not None else surah_ayah_counts()
    grouped = {}
    for surah, ayah, time, status, source in rows:
        grouped.setdefault(surah, []).append((ayah, time, status, source))
    done = set()
    for surah, items in grouped.items():
        count = expected.get(surah)
        if type(surah) is not int or count is None or [r[0] for r in items] != list(range(1, count + 1)):
            continue
        times = [r[1] for r in items]
        if (any(type(r[0]) is not int for r in items) or any(type(t) is not int or t < 0 for t in times)
                or any(a >= b for a, b in zip(times, times[1:]))):
            continue
        auto_sources = [source for _, _, status, source in items if status not in ("edited", "verified")]
        current = True
        for source in auto_sources:
            parts = source.split(":") if isinstance(source, str) else []
            modes = [part.partition("=")[2] for part in parts if part.startswith("requested_mode=")]
            policies = [part.partition("=")[2] for part in parts if part.startswith("full_onset_policy=")]
            if (len(parts) < 2 or parts[1] != model_id or modes != ["full"]
                    or (model_id in ("quran-phonetics", "quran-phonetics-fp32")
                        and (not onset_policy or policies != [onset_policy]))):
                current = False
                break
        if current:
            done.add(surah)
    return done
