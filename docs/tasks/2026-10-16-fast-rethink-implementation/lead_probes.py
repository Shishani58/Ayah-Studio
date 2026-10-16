"""Independent persistence/export probes, run after the implementation is ready."""

import hashlib
import json
import tempfile
from pathlib import Path
from threading import Event
from unittest.mock import patch

from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope
from surah_cutter.data.alignment_identity import fast_resume_suras, is_fast_source
from surah_cutter.data.alignment_result import result_source, write_alignment_result
from surah_cutter.data.export_db import ExportError, export_db
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.data.quran_text import surah_ayah_counts


OLD_FAST = "review-model:requested_mode=fast:actual_mode=fast:fast_run_key=" + "a" * 64
OLD_FULL = "review-model:requested_mode=full:actual_mode=full"


def empty_trace():
    return {
        "mode": "fast", "fast_version": "lead-review",
        "model_source": "review-model", "backend": "review", "align_lag_ms": 0,
        "surah": 112, "ayah_count": 4, "duration_ms": 10000,
        "result_contract": "partial-v1", "result_complete": False,
        "generated_rows": [], "unresolved_ayahs": [1, 2, 3, 4],
        "review_required_ayahs": [1, 2, 3, 4],
        "boundaries": [{"ayah": a, "start_ms": None, "evidence_kind": "unresolved",
                        "confidence": 0, "review_reason": "lead_unresolved"}
                       for a in range(1, 5)],
        "ayahs": [{"ayah": a, "onset_ms": None, "conf": 0, "reliable": False}
                  for a in range(1, 5)],
    }


def check_empty_replacement(folder, requested_mode):
    path = folder / f"empty-{requested_mode}.db"
    with ProjectDB.open(str(path), "lead-review", 1) as db:
        db.write_timings([(112, a, a * 1000) for a in range(1, 5)], OLD_FAST, 0)
        db.set_status(112, 1, "edited")
        db.set_status(112, 2, "verified")
        db.write_timings([(112, 3, 3000)], OLD_FULL, .9)
        protected = {a: db.get_review(112, a) for a in (1, 2)}
        full = db.get_review(112, 3)
        result = write_alignment_result(db, [], trace=empty_trace(),
                                        requested_mode=requested_mode, db_path=path)
        assert db.get_timings() == [(112, 1, 1000), (112, 2, 2000), (112, 3, 3000)]
        assert {a: db.get_review(112, a) for a in (1, 2)} == protected
        assert db.get_review(112, 3) == full
        assert result.written == 0
        assert list(result.persisted_rows) == db.get_timings()
        assert result.log_paths and Path(result.log_paths["json"]).is_file()
        saved = json.loads(Path(result.log_paths["json"]).read_text())
        assert saved["trace"]["result_complete"] is False
        assert saved["model_rows"] == []
    assert 112 not in fast_resume_suras(path, {112: "a" * 64})
    return {"passed": True, "requested_mode": requested_mode,
            "preserved": ["edited", "verified", "auto_FULL"], "removed": ["auto_FAST"]}


def check_export(folder):
    path = folder / "export-source.db"
    target = folder / "output.db"
    with ProjectDB.open(str(path), "lead-review", 1) as db:
        db.write_timings([(s, a, a * 1000) for s, count in surah_ayah_counts().items()
                          for a in range(1, count + 1)], OLD_FULL, .9)
        db.write_timings([(112, 4, 4000)], OLD_FAST, 0)
    target.write_bytes(b"preserve previous export")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rejected = False
    try:
        export_db(str(path), str(target))
    except ExportError:
        rejected = True
    assert rejected, "complete canonical shape must not legitimize an old FAST estimate"
    assert target.read_bytes() == b"preserve previous export"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with ProjectDB.open(str(path), "lead-review", 1) as db:
        db.set_status(112, 4, "verified")
    export_db(str(path), str(target))
    return {"passed": True, "old_estimate_rejected": True,
            "human_verified_time_exported": True}


def check_implicit_fast_origin():
    trace = empty_trace()
    trace.pop("mode")
    source = result_source(trace, "words")
    assert is_fast_source(source), "implicit FAST must remain FAST provenance inside words"
    assert ":actual_mode=fast:" in source
    return {"passed": True, "source": source}


def check_cancelled_cleanup(folder):
    path = folder / "cancelled.db"
    with ProjectDB.open(str(path), "lead-review", 1) as db:
        db.write_timings([(112, a, a * 1000) for a in range(1, 5)], OLD_FAST, 0)
        db.set_status(112, 2, "verified")
        before = (db.get_timings(), [db.get_review(112, a) for a in range(1, 5)])
        cancel = Event()
        validate = db._validate_order

        def cancel_after_cleanup(surah):
            validate(surah)
            cancel.set()

        caught = False
        with patch.object(db, "_validate_order", cancel_after_cleanup), cancellation_scope(cancel):
            try:
                write_alignment_result(db, [], trace=empty_trace(),
                                       requested_mode="fast", db_path=path)
            except GenerationCancelled:
                caught = True
        assert caught
        assert (db.get_timings(), [db.get_review(112, a) for a in range(1, 5)]) == before
        assert not list(folder.glob("cancelled.logs/*.json"))
    return {"passed": True, "cleanup_rolled_back": True, "manual_metadata_unchanged": True}


def main():
    with tempfile.TemporaryDirectory(prefix="fast-lead-review-") as name:
        folder = Path(name)
        results = {
            "empty_fast": check_empty_replacement(folder, "fast"),
            "empty_words_using_fast": check_empty_replacement(folder, "words"),
            "export": check_export(folder),
            "implicit_fast_origin": check_implicit_fast_origin(),
            "cancelled_cleanup": check_cancelled_cleanup(folder),
            "new_acoustic_calls": 0,
        }
    destination = Path(__file__).with_name("lead-probes.json")
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
