"""Fresh explicit FULL, comparison after prediction, then optional normal save."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from surah_cutter.core import pipeline
from surah_cutter.data import quran_text, references
from surah_cutter.data.alignment_result import write_alignment_result
from surah_cutter.data.project_db import ProjectDB

TASK = Path(__file__).resolve().parent
ROOT = Path.home() / "Desktop/Abu_Bakr_Ash-Shatri"
DB = ROOT / "Abu_Bakr_Ash-Shatri.db"
AUDIO = ROOT / "002.mp3"
BASELINE = ROOT / "Abu_Bakr_Ash-Shatri.logs/runs/sura_002_full_3aefe26fcfbb494b9043475b8cca3a0f.json"
CACHE = Path.home() / ".cache/surah-cutter/full-local-pause-v5"
CACHE.mkdir(parents=True, exist_ok=True)
RAW = CACHE / "baqara-fresh.json"


def stats(rows, reference):
    errors = np.array([abs(ms - reference[2, a]) for _, a, ms in rows])
    return dict(n=len(errors), median_ms=float(np.median(errors)),
                mean_ms=float(np.mean(errors)), p95_ms=float(np.percentile(errors, 95)),
                max_ms=int(max(errors)), over500=int(sum(errors > 500)))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("action", choices=["generate", "save"])
    action = args.parse_args().action
    baseline = json.loads(BASELINE.read_text())
    if action == "generate":
        trace, logs = {"overwrite_manual": False}, []

        def log(line):
            logs.append(str(line))
            if not str(line).startswith("Аят "):
                print(line, flush=True)

        started = time.monotonic()
        rows = pipeline.process_surah(str(AUDIO), 2, mode="full", trace=trace, log_event=log)
        assert trace["audio_signature"] == baseline["audio_signature"]
        assert [a for _, a, _ in rows] == list(range(1, quran_text.ayah_count(2) + 1))
        assert all(left[2] < right[2] for left, right in zip(rows, rows[1:]))
        payload = dict(rows=rows, trace=trace, logs=logs, elapsed_s=time.monotonic() - started)
        RAW.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        payload = json.loads(RAW.read_text())
        rows, trace = payload["rows"], payload["trace"]
        # Snapshot SQLite through its backup API, including any WAL changes.
        backup = CACHE / f"before-save-{time.time_ns()}.db"
        with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)
        with ProjectDB(sqlite3.connect(DB)) as db:
            protected = {a: (ms, db.get_review(2, a)) for s, a, ms in db.get_timings()
                         if s == 2 and db.get_review(s, a).get("status") in ("edited", "verified")}
            saved = write_alignment_result(db, rows, trace=trace, requested_mode="full",
                                           db_path=DB, log_lines=payload["logs"])
            stored = {a: ms for s, a, ms in db.get_timings() if s == 2}
            assert len(stored) == 286
            for a, (ms, review) in protected.items():
                assert stored[a] == ms and db.get_review(2, a) == review
        (TASK / "saved.json").write_text(json.dumps(dict(backup=str(backup), written=saved.written,
                                                        protected=len(protected),
                                                        log_paths=saved.log_paths), indent=2) + "\n")
        print("Saved", saved, flush=True)

    # Reference data is evaluation-only, opened after inference finishes.
    reference = references.load_reference("abu_bakr_ash_shatri")
    old_checks = {c["ayah"]: c for c in baseline["trace"]["ayahs"]}
    current = trace["ayahs"]
    comparison = [dict(ayah=c["ayah"], old_ms=old_checks[c["ayah"]]["onset_ms"],
                       new_ms=c["onset_ms"], model_ms=c["pre_center_onset_ms"],
                       reference_ms=reference[2, c["ayah"]], reason=c["pause_center_reason"],
                       pause_start_ms=c.get("center_pause_start_ms"),
                       pause_end_ms=c.get("center_pause_end_ms"), lead_ms=c.get("pause_lead_ms"))
                  for c in current]
    model_changes = [c["ayah"] for c in current
                     if c["pre_center_onset_ms"] != old_checks[c["ayah"]]["pre_center_onset_ms"]]
    report = dict(baseline=str(BASELINE), raw_fresh=str(RAW), elapsed_s=payload["elapsed_s"],
                  acoustic_windows=len(trace["full_windows"]["windows"]),
                  n_chunks=trace.get("n_chunks"), policy=trace["full_windows"]["onset_policy"],
                  reasons=dict(Counter(c["pause_center_reason"] for c in current)),
                  old=stats(baseline["rows"], reference), new=stats(rows, reference),
                  model_changes=model_changes, comparison=comparison)
    (TASK / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparison"}, indent=2), flush=True)
    print(json.dumps(comparison[:10], indent=2), flush=True)


if __name__ == "__main__":
    main()
