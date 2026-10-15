"""Fresh bounded CTC recovery and DSP geometry controls, no DB/settings writes."""

import hashlib
import json
from pathlib import Path
from surah_cutter.core import align, candidates, decode, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_executor import _noisy_opening, pause_geometry
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

ROOT = Path(__file__).parent
model_store.ACTIVE_ID = "quran-phonetics"
pcm = decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/001.mp3")
bundle = align.default_bundle()
adapter = CTCAdapter(bundle)
words = quran_text.surah_words(1)
rendered = adapter.represent(words)
groups, ends = {}, {}
for w, u in zip(words, rendered):
    groups[w.ayah] = groups.get(w.ayah, ()) + u
    ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
reads = LocalEvidenceReads(pcm, 16000, adapter)
items = []
with dispatch_spy(bundle, pcm) as actual:
    for width in (1700, 1900, 2100):
        obs = reads.read(6420, 6620 + width, reason="missing-onset-additional-word-recovery")
        proof, expand = _noisy_opening(obs, groups, ends, 2, 0.25)
        items.append(
            {
                "window_ms": [6420, 6620 + width],
                "greedy": obs.text,
                "unit_quality": obs.unit_quality,
                "identity_proof": proof,
                "authorized_expansion": expand,
            }
        )
trace = {"total_ms": len(pcm) / 16, **reads.diagnostics()}
verify_dispatch(actual, pcm, trace)
geometry = []
for surah in (1, 79, 108):
    signal = (
        pcm
        if surah == 1
        else decode.decode(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3")
    )
    decisions = candidates.pause_pipeline_decisions(signal, 16000, surah=surah)
    pauses = [d.pause for d in decisions if d.accepted]
    rows = [
        {"interval_ms": [p.start_ms, p.end_ms], **pause_geometry(signal, 16000, p)} for p in pauses
    ]
    geometry.append(
        {
            "surah": surah,
            "n_candidates": len(rows),
            "n_accepted": sum(r["accepted"] for r in rows),
            "rows": rows,
        }
    )
(ROOT / "minimal-extensions.json").write_text(
    json.dumps(
        {
            "ctc_actual": items,
            "ledger": trace,
            "actual_dispatches": actual,
            "model_runtime": adapter.runtime_identity,
            "geometry_dsp_only": geometry,
            "source_sha256": hashlib.sha256(
                Path("surah_cutter/core/fast_executor.py").read_bytes()
            ).hexdigest(),
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {
            "ctc": items,
            "geometry": [
                {k: g[k] for k in ("surah", "n_candidates", "n_accepted")} for g in geometry
            ],
        },
        ensure_ascii=False,
    )
)
