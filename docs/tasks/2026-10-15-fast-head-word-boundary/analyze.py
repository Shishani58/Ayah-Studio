"""Find recorded ASR word-boundary candidates without model calls."""

import json
from pathlib import Path
from surah_cutter.core.fast_adapters import arabic_units
from surah_cutter.core.fast_executor import Policy, plan_phrase
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.core.model_worker import whisper_alignment_text
from surah_cutter.data import quran_text

task = Path(__file__).parent.resolve()
prior = task.parent / "2026-10-14-fast-asr-raw-ownership"
threshold = Policy().asr_quality
out = []
for model in ["whisper-tarteel-tiny", "whisper-tarteel-base"]:
    for c in json.loads((prior / f"{model}-after.json").read_text()):
        words = quran_text.surah_words(c["surah"])
        rendered = tuple(
            arabic_units(whisper_alignment_text(t)) for t in inference_words(words, model)
        )
        groups, ends = {}, {}
        for w, u in zip(words, rendered):
            groups.setdefault(w.ayah, []).extend(u)
            ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
        groups = {a: tuple(u) for a, u in groups.items()}
        owners = {}
        for i in range(len(words) - 1):
            owners.setdefault(rendered[i] + rendered[i + 1], []).append(i)
        for b in c["trace"]["boundaries"]:
            ayah = b["ayah"]
            if b["confidence"] > 0 or sum(r[2] > 0 for r in c["rows"][: ayah - 1]) < 2:
                continue
            current = [i for i, w in enumerate(words) if w.ayah == ayah]
            first = rendered[current[0]]
            plan = plan_phrase(groups, ayah, ends)
            if len(first) != 1 or len(first[0]) < 3 or len(plan["units"]) <= 1 or plan["ambiguous"]:
                continue
            reads = [
                r
                for r in c["trace"]["acoustic_reads"]
                if r["reason"].startswith(f"ayah={ayah}:") and "previous_tail" not in r["reason"]
            ]
            bodies = []
            for r in reads:
                o = r["observed"]
                if o["truncated"]:
                    continue
                for wi in current[1:-1]:
                    pair = rendered[wi] + rendered[wi + 1]
                    if owners[pair] != [wi]:
                        continue
                    for offset in range(len(o["units"]) - len(pair) + 1):
                        if (
                            tuple(o["units"][offset : offset + len(pair)]) == pair
                            and min(o["unit_quality"][offset : offset + len(pair)]) >= threshold
                        ):
                            bodies.append({"read_id": r["id"], "word_index": wi, "pair": pair})
            if not bodies:
                continue
            for r in reads:
                o = r["observed"]
                if o["truncated"] or not o["units"] or not o["unit_quality"]:
                    continue
                head = o["units"][0]
                if (
                    head.startswith(first[0])
                    and len(head) > len(first[0])
                    and o["unit_quality"][0] > 0
                    and not any("unsupported" in a for a in o.get("attribution", ()))
                ):
                    out.append(
                        {
                            "model": model,
                            "surah": c["surah"],
                            "ayah": ayah,
                            "head_read_id": r["id"],
                            "window_ms": r["window_ms"],
                            "observed_head": head,
                            "expected_first": first[0],
                            "plan": plan["units"],
                            "body_cues": bodies,
                        }
                    )
(task / "eligibility.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
