"""Empirical probability of matching the timing benchmark within 500 ms.

Recognition scores are never converted into timing percentages.
Inference reads only saved prediction diagnostics, not reference timestamps.
"""

from functools import lru_cache
import json
import math
from pathlib import Path

FEATURE_VERSION = 2
MODEL_SHA256 = "423777f2d721bc5c40426463d4eab6e305d6aa8cde0ecd69628ac4a69eb1589a"


def pause_stratum(check, time_ms, trace):
    """Geometry classes fixed by the 500 ms target, not fitted to a reference."""
    local = [
        p
        for p in check.get("pause_local_decisions", [])
        if p.get("accepted") is True and p.get("owned") is True
    ]
    if len(local) > 1 or (check.get("pause_center_reason") == "positioned" and len(local) != 1):
        return "ambiguous"

    def contains(p):
        return (
            all(
                type(p.get(k)) in (int, float) and math.isfinite(p[k])
                for k in ("start_ms", "end_ms")
            )
            and 0 <= p["start_ms"] < p["end_ms"]
            and p["start_ms"] - 80 <= time_ms <= p["end_ms"] + 80
        )

    if local and not contains(local[0]):
        return "ambiguous"
    # Use the same global DSP measurement in calibration and prediction. The
    # newer, narrower local positioning crop has a different edge convention.
    pauses = [
        p
        for p in trace.get("full_windows", {}).get("pause_decisions", [])
        if p.get("accepted") is True
    ]
    containing = [p for p in pauses if contains(p)]
    if len(containing) > 1:
        return "ambiguous"
    if not containing:
        return "none"
    p = containing[0]
    span = max(abs(time_ms - p["start_ms"]), abs(p["end_ms"] - time_ms))
    return "narrow" if span <= 500 else "middle" if span <= 1000 else "wide"


@lru_cache(maxsize=1)
def calibration():
    return json.loads(Path(__file__).with_name("timing_calibration.json").read_text())


def timing_probability(check, time_ms, trace):
    """Return an empirically calibrated estimate, or abstain outside its scope."""
    if type(check.get("ayah")) is not int or check.get("onset_ms") != time_ms:
        return None
    source = str(trace.get("model_source", ""))
    snapshot = trace.get("run_snapshot", {})
    model = snapshot.get("model", source.split(":", 1)[0])
    if source.startswith("onnx-ctc:quran-phonetics:"):
        model = "quran-phonetics"
    elif source.startswith("onnx-ctc:quran-phonetics-fp32:"):
        model = "quran-phonetics-fp32"
    # Calibration currently covers the measured INT8 Quran Phonetics model.
    if model != "quran-phonetics":
        return None
    if MODEL_SHA256 not in source and snapshot.get("revision") != MODEL_SHA256:
        return None
    mode = (
        "fast"
        if (
            trace.get("fast_version")
            or trace.get("mode") == "fast"
            or trace.get("backend") == "ctc_greedy_probability"
        )
        else "full"
    )
    if mode == "fast" and trace.get("fast_version") != "sequential-fast-v3-sparse-12-split-opening":
        return None
    try:
        artifact = calibration()
        fit = artifact["models"][mode]
    except (OSError, ValueError, KeyError):
        return None
    if (
        artifact.get("feature_version") != FEATURE_VERSION
        or artifact.get("target") != "reference-cut"
        or artifact.get("tolerance_ms") != 500
    ):
        return None
    key = pause_stratum(check, time_ms, trace)
    if mode == "fast":
        owners = [b for b in trace.get("boundaries", []) if b.get("ayah") == check["ayah"]]
        if len(owners) != 1 or owners[0].get("start_ms") != time_ms:
            return None
        kind = owners[0].get("evidence_kind")
        if kind not in {"estimate", "pause", "first_onset"}:
            return None
        if kind != "estimate":
            proof = owners[0].get("proof") or {}
            if (
                proof.get("complete") is not True
                or proof.get("ambiguous") is not False
                or proof.get("competitors_checked") is not True
            ):
                return None
        key = "estimate" if kind == "estimate" else "proof"
    bucket = fit.get(key, {})
    p = bucket.get("probability")
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 < p < 1:
        return None
    return dict(
        probability=bucket["probability"],
        tolerance_ms=500,
        target="reference-cut",
        calibration=artifact["id"],
        calibration_n=bucket["n"],
        calibration_recordings=bucket["recordings"],
        calibration_successes=bucket["successes"],
        interval95=bucket.get("interval95"),
        benchmark="packaged-reference",
        model=mode,
    )
