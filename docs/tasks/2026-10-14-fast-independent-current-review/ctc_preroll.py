"""Fresh CTC local-only diagnostic; never writes timings or settings."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from surah_cutter.core import align, candidates, decode, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_executor import plan_phrase
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

ROOT = Path(__file__).parent
model_store.ACTIVE_ID = "quran-phonetics"
audio = Path("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/001.mp3")
pcm = decode.decode(str(audio))
bundle = align.default_bundle()
adapter = CTCAdapter(bundle)
words = quran_text.surah_words(1)
rendered = adapter.represent(words)
groups, ends = {}, {}
for word, units in zip(words, rendered):
    base = len(groups.get(word.ayah, ()))
    groups.setdefault(word.ayah, []).extend(units)
    ends.setdefault(word.ayah, []).append(base + len(units))
groups = {a: tuple(units) for a, units in groups.items()}
phrase = plan_phrase(groups, 2, ends)["units"]
decisions = candidates.pause_pipeline_decisions(pcm, 16000, surah=1)
pause = next(d.pause for d in decisions if d.accepted)
reads = LocalEvidenceReads(pcm, 16000, adapter)
items = []
with dispatch_spy(bundle, pcm) as actual:
    for preroll in (200, 400, 600):
        obs = reads.read(max(pause.start_ms, pause.end_ms - preroll), pause.end_ms + 1300,
                         reason=f"independent-prefix-clipping:preroll={preroll}")
        items.append({"preroll_ms": preroll, "observation": {
            key: value for key, value in asdict(obs).items() if key != "posterior"},
                      "greedy_complete": obs.units[:len(phrase)] == phrase})
trace = {"total_ms": len(pcm) / 16, **reads.diagnostics()}
verify_dispatch(actual, pcm, trace)
payload = {"audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
           "pcm_sha256": hashlib.sha256(memoryview(pcm)).hexdigest(),
           "canonical_ayah": "1:2", "phrase": phrase, "pause": asdict(pause),
           "model_runtime": adapter.runtime_identity, "reads": items,
           "actual_dispatches": actual, "trace": trace,
           "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path("surah_cutter/core/fast_adapters.py"),
                                       Path("surah_cutter/core/fast_reads.py"),
                                       Path("surah_cutter/core/fast_executor.py")]}}
(ROOT / "ctc-preroll.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"phrase": "".join(phrase), "pause": asdict(pause),
                  "reads": [{"preroll_ms": i["preroll_ms"],
                             "text": i["observation"]["text"],
                             "greedy_complete": i["greedy_complete"]} for i in items],
                  "actual_dispatches": len(actual), "coverage": trace["coverage"],
                  "audio_ms": trace["audio_ms"]}, ensure_ascii=False))
