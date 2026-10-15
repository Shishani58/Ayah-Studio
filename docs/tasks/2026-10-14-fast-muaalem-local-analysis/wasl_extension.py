"""One finite 500ms crop control after an independently heard incomplete seam."""
import json
from pathlib import Path

from adapter import MuaalemLocal
from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from tools.bench_fast_models import dispatch_spy, verify_dispatch

task = Path(__file__).parent
cause = json.loads((task / "incomplete-wasl-cause.json").read_text())
model_store.ACTIVE_ID = "muaalem"
bundle = align.default_bundle()
adapter = MuaalemLocal(bundle)
pcm = decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/001.mp3")
ledger = LocalEvidenceReads(pcm, 16000, adapter)
records = []
combined = tuple(cause["seam"]["combined"])
with dispatch_spy(bundle, pcm) as actual:
    for right in (cause["current_window_ms"][1], cause["max_same_start_end_ms"]):
        left = cause["current_window_ms"][0]
        obs = ledger.read(left, right, reason="measured_incomplete_wasl_word_edge")
        offsets = [i for i in range(len(obs.units)) if obs.units[i:i + len(combined)] == combined]
        matches = [{"offset": i, "min_quality": min(obs.unit_quality[i:i + len(combined)]),
                    "onset_ms": left + obs.positions_ms[i + len(cause["seam"]["tail"])]} for i in offsets]
        records.append({"window_ms": [left, right], "greedy": obs.units,
                        "quality": obs.unit_quality, "complete_seam_matches": matches})
        (task / "wasl-extension-partial.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
trace = {"total_ms": len(pcm) / 16, **ledger.diagnostics()}
verify_dispatch(actual, pcm, trace)
(task / "wasl-extension.json").write_text(json.dumps({"records": records,
    "actual_dispatches": actual, "ledger": trace, "runtime_identity": adapter.runtime_identity,
    "scope": "bounded local control only; no production boundary or corpus claim"}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(records, ensure_ascii=False, indent=1))
release_runtimes()
