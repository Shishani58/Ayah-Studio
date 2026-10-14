"""Actual normal FAST pipeline, charged dispatches and temporary DB safety."""
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from surah_cutter.core import align, decode, fast_executor, model_store, pipeline
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data.alignment_identity import fast_run_key
from surah_cutter.data.alignment_result import write_alignment_result
from surah_cutter.data.project_db import ProjectDB
from tools.bench_fast_models import dispatch_spy, validate, verify_dispatch

root = Path(__file__).parent
assert Path(pipeline.__file__).resolve().is_relative_to(Path.cwd().resolve()), "Verification imported another checkout"
sources = [Path("surah_cutter/core") / (name + ".py") for name in (
    "pipeline", "align", "align_fast", "fast_executor", "fast_adapters", "fast_ctc",
    "fast_reads", "emissions", "model_runtime", "model_worker", "model_store", "candidates")]
sha = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
items = []
for model in ("quran-phonetics", "whisper-tarteel-tiny", "whisper-tarteel-base"):
    release_runtimes()
    align._ONNX_CACHE.clear()
    model_store.ACTIVE_ID = model
    bundle = align.default_bundle()
    for surah, count in ((108, 3), (109, 6)):
        path = Path(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3")
        pcm = decode.decode(str(path))
        trace = {}
        with TemporaryDirectory(prefix="unified-fast-default-") as folder:
            db_path = Path(folder) / "project.db"
            with ProjectDB.open(str(db_path), "temporary-verification", 1) as db:
                db.write_timings([(surah, 1, 10), (surah, 2, 20)], "manual-fixture", .7)
                db.set_status(surah, 1, "verified")
                db.set_status(surah, 2, "edited")
                before = [db.get_review(surah, a) for a in (1, 2)]
                with dispatch_spy(bundle, pcm) as actual:
                    rows = pipeline.process_surah(str(path), surah, mode="fast", project_db=db, trace=trace)
                verify_dispatch(actual, pcm, trace)
                model_rows = [(b["ayah"], b["start_ms"], b["confidence"]) for b in trace["boundaries"]]
                assert not validate(model_rows, trace, list(range(1, count + 1)))
                assert trace["fast_version"] == fast_executor.VERSION
                assert trace["run_snapshot"]["pcm_sha256"] == hashlib.sha256(memoryview(pcm)).hexdigest()
                assert trace["fast_run_key"] == fast_run_key(trace["run_snapshot"])
                assert rows[:2] == [(surah, 1, 10), (surah, 2, 20)]
                result = write_alignment_result(db, rows, trace=trace, requested_mode="fast", db_path=db_path)
                assert result.outcome == "saved" and result.diagnostics_error is None
                assert [db.get_review(surah, a) for a in (1, 2)] == before
                assert [(s, a, t) for s, a, t in result.persisted_rows][:2] == rows[:2]
                assert "fast_run_key=" + trace["fast_run_key"] in db.get_review(surah, 3)["source"]
                assert len(json.loads(Path(result.log_paths["run"]).read_text())["persisted_rows"]) == count
            items.append({"model": model, "surah": surah, "rows": model_rows,
                          "trace": trace, "actual_dispatches": actual,
                          "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "temporary_manual_preserved": True})
            print(model, surah, "confirmed", sum(c > 0 for a, t, c in model_rows), "reads", len(actual), flush=True)
release_runtimes()
# Evaluation starts only after every model has finished inference.
reference = Path("data/references/abu_bakr_ash_shatri.db")
with sqlite3.connect(f"{reference.resolve().as_uri()}?mode=ro", uri=True) as connection:
    for item in items:
        truth = dict(connection.execute("SELECT ayah,time FROM timings WHERE sura=?", (item["surah"],)))
        item["errors_ms"] = [abs(t - truth[a]) for a, t, c in item["rows"]]
        item["positive_over500"] = [[a, e] for (a, t, c), e in zip(item["rows"], item["errors_ms"]) if c > 0 and e > 500]
        assert not item["positive_over500"]
assert sha == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(root / "verification.json").write_text(json.dumps({"source_sha256": sha,
    "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "results": items}, indent=2) + "\n")
print("Six real default FAST pipeline/transaction cases passed", flush=True)
