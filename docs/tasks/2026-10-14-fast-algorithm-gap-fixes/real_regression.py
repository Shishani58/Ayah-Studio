"""Small paired real regression; HEAD executor vs current, shared adapters."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from surah_cutter.core import align, decode, model_store, fast_executor
from surah_cutter.core.fast_adapters import make_adapter
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

ROOT = Path(__file__).parent
baseline_source = subprocess.check_output(
    ["git", "show", "37f3309:surah_cutter/core/fast_executor.py"]
)
rows = []
with tempfile.TemporaryDirectory(prefix="fast-gap-baseline-") as folder:
    path = Path(folder) / "baseline.py"
    path.write_bytes(baseline_source)
    spec = importlib.util.spec_from_file_location("surah_cutter.core._gap_baseline", path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    for model, surah in [
        ("quran-phonetics", 108),
        ("whisper-tarteel-tiny", 109),
        ("whisper-tarteel-base", 109),
    ]:
        model_store.ACTIVE_ID = model
        align._ONNX_CACHE.clear()
        release_runtimes()
        bundle = align.default_bundle()
        pcm = decode.decode(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3")
        comparison = []
        for variant, executor in [("HEAD37f3309", baseline), ("current", fast_executor)]:
            trace = {}
            with dispatch_spy(bundle, pcm) as calls:
                result = executor.align_surah_sequential(
                    pcm, 16000, quran_text.surah_words(surah), bundle=bundle, trace=trace
                )
            verify_dispatch(calls, pcm, trace)
            comparison.append(
                {"variant": variant, "rows": result, "trace": trace, "actual_dispatches": calls}
            )
        old, new = [item["rows"] for item in comparison]
        changed = [
            {"ayah": a, "old": list(x), "new": list(y)}
            for a, x, y in zip(range(1, len(old) + 1), old, new)
            if x != y
        ]
        rows.append(
            {
                "model": model,
                "surah": surah,
                "paired": comparison,
                "changed_rows": changed,
                "runtime_identity": make_adapter(bundle).runtime_identity,
            }
        )
        print(
            model,
            surah,
            "old/new confirmed",
            sum(r[2] > 0 for r in old),
            sum(r[2] > 0 for r in new),
            "changed",
            changed,
            flush=True,
        )
release_runtimes()
(ROOT / "real-regression.json").write_text(
    json.dumps(
        {
            "comparisons": rows,
            "baseline_source_sha256": hashlib.sha256(baseline_source).hexdigest(),
            "current_executor_sha256": hashlib.sha256(
                Path("surah_cutter/core/fast_executor.py").read_bytes()
            ).hexdigest(),
            "scope": "fresh real acoustic paired regression, no reference inputs/no DB/settings writes; not corpus accuracy or listening validation",
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
