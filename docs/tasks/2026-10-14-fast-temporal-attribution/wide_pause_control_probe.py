"""Independent local seam endpoint probe for two newly reached temporal outliers."""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import sys
from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

task = Path(__file__).parent.resolve()
parent = task.parent / "2026-10-14-fast-muaalem-local-analysis"
sys.path.insert(0, str(parent))
from adapter import MuaalemLocal  # noqa: E402
source = task.parent / "2026-10-14-fast-pronunciation-variants/collapse_raw_tempo_probe.py"
tree = ast.parse(source.read_text())
definitions = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "collapse")
               or (isinstance(n, ast.ClassDef) and n.name == "DurationNeutral")]
namespace = {"LONG": {"ا", "ۦ", "ۥ", "ـ"}, "Original": MuaalemLocal, "Path": Path,
             "hashlib": hashlib, "replace": replace, "__file__": str(source)}
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),"exec"),namespace)
original_observe = MuaalemLocal.observe

def captured_observe(self, pcm, sr, *, total_samples):
    raw = original_observe(self, pcm, sr, total_samples=total_samples)
    self.temporal_raw = raw
    return raw

MuaalemLocal.observe = captured_observe
case_source = task.parent / "2026-10-14-fast-late-future-anchor/late-corpus.json"
cases = {c["surah"]:c for c in json.loads(case_source.read_text())}
model_store.ACTIVE_ID = "muaalem"
bundle = align._get_onnx_bundle()
result = []
try:
    for surah, ayah in [(2,21),(2,26),(2,29),(2,44),(77,25)]:
        case = cases[surah]
        b = next(b for b in case["trace"]["boundaries"] if b["ayah"]==ayah)
        audio = Path(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03}.mp3")
        assert hashlib.sha256(audio.read_bytes()).hexdigest()==case["audio_sha256"]
        pcm = decode.decode(str(audio))
        adapter = namespace["DurationNeutral"](bundle)
        words = quran_text.surah_words(surah)
        represented = adapter.represent(words)
        previous = max(i for i,w in enumerate(words) if w.ayah==ayah-1)
        tail = represented[previous]
        phrase = tuple(b["proof"]["phrase"])
        left,right = b["proof"]["pause_ms"]
        reads = LocalEvidenceReads(pcm,16000,adapter,cache_limit_bytes=32*1024*1024)
        with dispatch_spy(bundle,pcm) as dispatches:
            observed = reads.read(left-2500,right+min(1500.,6000.-(right-left)-2500.),reason=f"ayah={ayah}:temporal_seam_diagnostic")
            raw = adapter.temporal_raw
            ids = raw.posterior.argmax(-1).tolist()
            runs = []
            start = 0
            for frame in range(1, len(ids) + 1):
                if frame == len(ids) or ids[frame] != ids[start]:
                    token = ids[start]
                    if token != adapter.blank_id:
                        unit = adapter.inv[token]
                        assert len(unit) == 1
                        runs.append({"unit":unit,"start_ms":start*raw.stride_ms,
                                     "end_ms":frame*raw.stride_ms,
                                     "end_probability":float(np.exp(raw.posterior[frame-1,token]))})
                    start = frame
            assert tuple(r["unit"] for r in runs) == raw.units
            matches = []
            for label,target in [("previous_last_word",tail),("current_distinct_opening",phrase)]:
                for offset in range(len(observed.units)-len(target)+1):
                    if tuple(observed.units[offset:offset+len(target)])==tuple(target):
                        final_owner = observed.attribution[offset+len(target)-1]["raw_unit_indices"][-1]
                        final_run = runs[final_owner]
                        matches.append({"label":label,"offset":offset,
                                        "first_ms":reads.calls[0]["window_ms"][0]+observed.positions_ms[offset],
                                        "last_frame_ms":reads.calls[0]["window_ms"][0]+observed.positions_ms[offset+len(target)-1],
                                        "last_argmax_support_end_ms":reads.calls[0]["window_ms"][0]+final_run["end_ms"],
                                        "last_argmax_end_probability":final_run["end_probability"],
                                        "min_quality":min(observed.unit_quality[offset:offset+len(target)])})
            result.append({"surah":surah,"ayah":ayah,"runtime_identity":adapter.runtime_identity,
                           "audio_sha256":case["audio_sha256"],"boundary":b,
                           "previous_canonical_word":words[previous].text,"previous_units":tail,
                           "raw_argmax_runs":runs,"matches":matches,"reads":reads.calls,"actual_dispatches":dispatches,
                           "scope":"independent greedy seam diagnostic; no boundary accepted; no forced target"})
            (task / "wide-pause-control-probes.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        verify_dispatch(dispatches,pcm,{"acoustic_reads":reads.calls})
        print(surah,ayah,matches,flush=True)
finally:
    release_runtimes()
