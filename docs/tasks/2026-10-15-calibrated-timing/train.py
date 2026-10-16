"""Fit/save timing probabilities. Whole-recording splits, label-free inference.

Run as module from repository root. No acoustic calls or project DB writes.
Numpy is already a runtime dependency; no ML package required in the app.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

from surah_cutter.data.timing_probability import FEATURE_VERSION, pause_stratum

HERE = Path(__file__).parent
INPUTS = {}
RECORDINGS = {}
TEST_SHATRI = {9, 32, 79, 83, 98, 114}  # frozen before fitting
RECITERS = [
    "saad_al_ghamidi",
    "mohammed_siddiq_al_minshawi",
    "mahmoud_khalil_al_hussary",
    "abdul_basit_mujawwad",
]


def read(path):
    data = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def reference(reciter):
    p = Path("data/references") / f"{reciter}.db"
    INPUTS[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as db:
        return {(s, a): t for s, a, t in db.execute("select sura,ayah,time from timings")}


def corpus():
    REFERENCES = {r: reference(r) for r in ["abu_bakr_ash_shatri", *RECITERS]}

    def add(reciter, surah, mode, trace, origin):
        key = (reciter, surah, mode)
        trace["model_source"] = trace.get("model_source", "quran-phonetics")
        # One result per audio/mode. Later current snapshots replace old configs.
        RECORDINGS[key] = (trace, origin)

    corpus = read(Path("docs/tasks/2026-10-14-full-pause-windows/corpus-2-pause-20-2-0.json"))
    for s in corpus["surahs"]:
        checks = [dict(ayah=a, onset_ms=t, conf=c) for a, t, c in s["rows"]]
        add(
            "abu_bakr_ash_shatri",
            s["surah"],
            "full",
            dict(ayahs=checks, full_windows=s["windows"], total_ms=s["duration_s"] * 1000),
            "full-corpus",
        )

    for reciter in RECITERS:
        path = Path("docs/tasks/2026-10-14-full-other-reciters") / f"{reciter}.json"
        case = next(c for c in read(path)["cases"] if c["case"] == "strong:20:3:0")
        for s in case["surahs"]:
            add(
                reciter,
                s["surah"],
                "full",
                dict(
                    ayahs=s["onset_checks"],
                    full_windows=s["windows"],
                    total_ms=s["duration_s"] * 1000,
                ),
                str(path),
            )

    for path in sorted(
        Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/Abu_Bakr_Ash-Shatri.logs").glob(
            "sura_*_full.json"
        )
    ):
        d = read(path)
        add("abu_bakr_ash_shatri", d["surah"], "full", d["trace"], str(path))

    base = Path("docs/tasks/2026-10-15-fast-systematic-execution")
    for reciter, names in [
        ("abu_bakr_ash_shatri", ["baseline-a2.json", "baseline-e.json"]),
        *((r, [f"baseline-{r}.json"]) for r in RECITERS),
    ]:
        for name in names:
            path = base / name
            for s in read(path):
                if s["model"] == "quran-phonetics":
                    add(reciter, s["surah"], "fast", s["trace"], str(path))

    # Current FAST result replaces that recording's old run, never duplicates it.
    path = Path(
        "/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/Abu_Bakr_Ash-Shatri.logs/sura_002_fast.json"
    )
    d = read(path)
    add("abu_bakr_ash_shatri", 2, "fast", d["trace"], str(path))

    rows = []
    for (reciter, surah, mode), (trace, origin) in sorted(RECORDINGS.items()):
        group = f"{reciter}:{surah}"  # shared between modes; cannot cross splits
        test = surah in TEST_SHATRI if reciter == "abu_bakr_ash_shatri" else surah == 98
        for c in trace["ayahs"]:
            a, t = c["ayah"], c["onset_ms"]
            error = abs(t - REFERENCES[reciter][surah, a])
            split = (
                "control"
                if reciter == "abu_bakr_ash_shatri" and surah == 2
                else "test"
                if test
                else "train"
            )
            rows.append(
                dict(
                    group=group,
                    mode=mode,
                    ayah=a,
                    time=t,
                    error_ms=error,
                    split=split,
                    kind=next(
                        (
                            b.get("evidence_kind")
                            for b in trace.get("boundaries", [])
                            if b.get("ayah") == a
                        ),
                        None,
                    ),
                    stratum=pause_stratum(c, t, trace),
                    y=int(error <= 500),
                    origin=origin,
                )
            )
    return rows


if "--frozen" in sys.argv:
    frozen = json.loads((HERE / "evaluation.json").read_text())
    rows, INPUTS = frozen["rows"], frozen["inputs_sha256"]
else:
    rows = corpus()


def key_for(row):
    return (
        ("estimate" if row["kind"] == "estimate" else "proof")
        if row["mode"] == "fast"
        else row["stratum"]
    )


def fit(data):
    buckets = {}
    for key in sorted({key_for(r) for r in data}):
        subset = [r for r in data if key_for(r) == key]
        by_group = defaultdict(list)
        for r in subset:
            by_group[r["group"]].append(r["y"])
        contributions = np.array([(sum(v), len(v)) for v in by_group.values()])
        rng = np.random.default_rng(20261015)
        samples = contributions[
            rng.integers(len(contributions), size=(2000, len(contributions)))
        ].sum(1)
        successes = sum(r["y"] for r in subset)
        buckets[key] = dict(
            n=len(subset),
            successes=successes,
            probability=(successes + 1) / (len(subset) + 2),
            recordings=len(by_group),
            interval95=np.quantile(
                (samples[:, 0] + 1) / (samples[:, 1] + 2), [0.025, 0.975]
            ).tolist(),
        )
    return buckets


def predict(buckets, data):
    return np.array([buckets[key_for(r)]["probability"] for r in data])


def metrics(data, probabilities):
    y = np.array([r["y"] for r in data])
    probabilities = np.array(probabilities)
    histogram = []
    for lower, upper in zip([0, 0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8, 1.00001]):
        mask = (probabilities >= lower) & (probabilities < upper)
        if mask.any():
            histogram.append(
                dict(
                    n=int(mask.sum()),
                    predicted=float(probabilities[mask].mean()),
                    observed=float(y[mask].mean()),
                )
            )
    queue = probabilities < 0.8
    errors = y == 0
    return dict(
        n=len(data),
        recordings=len({r["group"] for r in data}),
        brier=float(np.mean((probabilities - y) ** 2)),
        ece=sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in histogram) / len(data),
        histogram=histogram,
        over500=int(errors.sum()),
        review=int(queue.sum()),
        found=int((queue & errors).sum()),
        missed=int((~queue & errors).sum()),
        probabilities=[float(min(probabilities)), float(max(probabilities))],
    )


models = {}
report = {}
for mode in ("fast", "full"):
    train = [r for r in rows if r["mode"] == mode and r["split"] == "train"]
    test = [r for r in rows if r["mode"] == mode and r["split"] == "test"]
    buckets = fit(train)
    models[mode] = buckets
    probabilities = predict(buckets, test)
    for r, p in zip(test, probabilities):
        r["probability"] = float(p)
    base_rate = sum(r["y"] for r in train) / len(train)
    report[mode] = dict(
        train_n=len(train),
        train_recordings=len({r["group"] for r in train}),
        test=metrics(test, probabilities),
        constant_test_brier=float(np.mean((base_rate - np.array([r["y"] for r in test])) ** 2)),
        estimator="Beta(1,1) posterior mean in fixed label-free timing strata; recording bootstrap interval",
    )
    control = [r for r in rows if r["mode"] == mode and r["split"] == "control"]
    cp = predict(buckets, control)
    report[mode]["excluded_shatri2_control"] = metrics(control, cp)
    for r, p in zip(control, cp):
        r["probability"] = float(p)
    report[mode]["thresholds"] = {}
    for threshold in (0.5, 0.8, 0.9, 0.95):
        report[mode]["thresholds"][str(threshold)] = {}
        for label, batch, ps in [("test", test, probabilities), ("control", control, cp)]:
            queue = [r for r, p in zip(batch, ps) if p <= threshold]
            report[mode]["thresholds"][str(threshold)][label] = dict(
                review=len(queue),
                found=sum(not r["y"] for r in queue),
                total_errors=sum(not r["y"] for r in batch),
            )

corpus_sha = hashlib.sha256(json.dumps(INPUTS, sort_keys=True).encode()).hexdigest()
artifact = dict(
    id=f"timing-reference-500-v2-{corpus_sha[:8]}",
    feature_version=FEATURE_VERSION,
    tolerance_ms=500,
    target="reference-cut",
    models=models,
    validation=report,
    corpus_sha256=hashlib.sha256(json.dumps(INPUTS, sort_keys=True).encode()).hexdigest(),
)
Path("surah_cutter/data/timing_calibration.json").write_text(json.dumps(artifact, indent=2) + "\n")
(HERE / "evaluation.json").write_text(
    json.dumps(
        dict(
            metrics=report,
            inputs_sha256=INPUTS,
            split=dict(test_shatri=sorted(TEST_SHATRI), test_others=[98], excluded_control=[2]),
            rows=rows,
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(report, indent=2))
