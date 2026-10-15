"""One final batch: fake recovery and 3 bounded CTC crops, no production edits."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

from surah_cutter.core import align, candidates, decode, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_ctc import prefix_path, prefix_scores
from surah_cutter.core.fast_executor import plan_phrase, plan_seam
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

ROOT = Path(__file__).parent
helpers = runpy.run_path("tests/test_fast_executor.py")
Fake, observation, pause, run = [helpers[k] for k in ("Fake", "observation", "pause", "run")]
fake = Fake(lambda f, start, duration: observation(("first",) if start == 1000 else ("shared",), positions=False))
fake.timing = False
rendered = list(fake.rendered)
for ayah, label in ((1, "first"), (2, "shared"), (3, "shared"), (4, "last")):
    index = next(i for i, word in enumerate(fake.words) if word.ayah == ayah)
    rendered[index] = (label,)
fake.rendered = tuple(rendered)
groups, ends = {}, {}
for word, units in zip(fake.words, fake.rendered):
    base = len(groups.get(word.ayah, ()))
    groups.setdefault(word.ayah, []).extend(units)
    ends.setdefault(word.ayah, []).append(base + len(units))
groups = {a: tuple(u) for a, u in groups.items()}
opening, seam = plan_phrase(groups, 2, ends), plan_seam(groups, 2, ends)
rows, trace = run(fake, [pause(1000), pause(6000)], duration=40000)
context = [r for r in trace["acoustic_reads"] if "previous_tail" in r["reason"]]
assert not opening["ambiguous"] and len(opening["units"]) > len(seam["units"])
assert rows[0][2] > 0 and rows[1][2] == 0 and not context
result = {"similar_opening_fake": {"opening_plan": opening, "seam_plan": seam,
                                    "rows": rows, "previous_tail_reads": context,
                                    "trace": trace}}

# Actual known audio, before reference access. Inference only on these local crops.
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
groups = {a: tuple(u) for a, u in groups.items()}
phrase = plan_phrase(groups, 2, ends)["units"]
pause = next(d.pause for d in candidates.pause_pipeline_decisions(pcm, 16000, surah=1) if d.accepted)
windows = [(pause.end_ms - 200, pause.end_ms + 1300),
           (pause.start_ms, pause.end_ms + 1300),
           (max(0, pause.start_ms - 800), pause.end_ms + 2500)]
reads = LocalEvidenceReads(pcm, 16000, adapter)
items = []
with dispatch_spy(bundle, pcm) as actual:
    for left, right in windows:
        obs = reads.read(left, right, reason="independent-missing-initial-a")
        a_token, l_token = adapter.vocab["a"], adapter.vocab["l"]
        matrix = obs.posterior
        support = prefix_path(matrix, [adapter.vocab[u] for u in phrase], adapter.blank_id, obs.stride_ms, adapter.lag_ms)
        # Compare canonical prefix and missing-a decode on the SAME charged frames.
        scores = prefix_scores(matrix, [[adapter.vocab[u] for u in phrase],
                                        [adapter.vocab[u] for u in phrase[1:]]], adapter.blank_id)
        l_position = obs.units.index("l") if "l" in obs.units else None
        l_frame = (round((obs.positions_ms[l_position] + adapter.lag_ms) / obs.stride_ms)
                   if l_position is not None else 0)
        before = matrix[:min(len(matrix), l_frame + 1)]
        top_a = sorted(range(len(before)), key=lambda i: before[i, a_token], reverse=True)[:5]
        items.append({"window_ms": [left, right], "greedy": obs.text,
                      "positions_ms": obs.positions_ms, "unit_quality": obs.unit_quality,
                      "hypotheses": obs.hypotheses,
                      "canonical_prefix_probability": float(np.exp(scores[0])),
                      "missing_a_prefix_probability": float(np.exp(scores[1])),
                      "first_a_before_first_l": [{"frame": i, "absolute_frame_ms": left + i * obs.stride_ms,
                                                  "p_a": float(np.exp(before[i, a_token])),
                                                  "p_l": float(np.exp(before[i, l_token])),
                                                  "p_blank": float(np.exp(before[i, adapter.blank_id])),
                                                  "winner": adapter.inv[int(before[i].argmax())]} for i in top_a],
                      "canonical_support": support,
                      "posterior_sha256": obs.posterior_sha256})
trace = {"total_ms": len(pcm) / 16, **reads.diagnostics()}
verify_dispatch(actual, pcm, trace)
result["ctc_actual"] = {"canonical_phrase": phrase, "pause": asdict(pause), "items": items,
                         "trace": trace, "actual_dispatches": actual,
                         "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                         "model_runtime": adapter.runtime_identity}
result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path("surah_cutter/core/fast_executor.py"),
                                      Path("surah_cutter/core/fast_adapters.py"),
                                      Path("surah_cutter/core/fast_ctc.py"),
                                      Path("surah_cutter/core/emissions.py")]}
(ROOT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"fake_opening": opening["units"], "shorter_context_opening": seam["units"],
                  "context_reads": len(context), "actual_ctc": [{k: i[k] for k in
                    ("window_ms", "greedy", "canonical_prefix_probability", "missing_a_prefix_probability",
                     "first_a_before_first_l", "canonical_support")} for i in items],
                  "real_reads": len(actual), "sum_ms": trace["audio_ms"], "coverage": trace["coverage"]}, ensure_ascii=False))
