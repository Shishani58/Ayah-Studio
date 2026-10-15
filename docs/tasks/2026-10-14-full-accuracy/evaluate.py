"""Evaluate generic pause policies, including all good controls, without inference GT input."""
import json
from pathlib import Path
import numpy as np

TASK = Path(__file__).resolve().parent
data = json.loads((TASK / "diagnostics.json").read_text())


def stats(errors):
    values = np.abs(errors)
    return {"n": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": int(max(values)),
            "over500": int(np.sum(values > 500))}


out = []
for reciter in dict.fromkeys(s["reciter"] for s in data):
    sources = [s for s in data if s["reciter"] == reciter]
    for policy in ["baseline", "midpoint", "left_plus_400", "right_minus_200", "weak_prefix_right"]:
        errors, new, removed, moves = [], [], [], []
        unflagged = 0
        for s in sources:
            pauses = s["pauses"]
            for row, before, check, detail in zip(s["rows"], s["errors_ms"], s["checks"], s["details"]):
                ayah, onset, _ = row
                # Decision only from predicted tokens / signal, before GT evaluation below.
                speech_token_start = detail["tokens"][0]["start_ms"]
                nearby = [p for p in pauses if p["start_ms"] <= speech_token_start <= p["end_ms"]+200]
                chosen = min(nearby, key=lambda p: abs(p["end_ms"]-speech_token_start), default=None)
                after_onset = onset
                if chosen and policy in ["midpoint", "left_plus_400", "right_minus_200"]:
                    after_onset = round(chosen["center_ms"] if policy == "midpoint" else
                                        min(chosen["end_ms"], chosen["start_ms"]+400)
                                        if policy == "left_plus_400" else chosen["end_ms"]-200)
                if policy == "weak_prefix_right" and check["first_token_conf"] < .25:
                    tokens = detail["tokens"][:detail["first_word_tokens"]]
                    strong = next((t for t in tokens if t["conf"] >= .45), None)
                    candidates = [p for p in pauses if strong and speech_token_start < p["end_ms"] <= strong["start_ms"]
                                  and all(t["conf"] < .25 for t in tokens if t["start_ms"] < p["end_ms"])]
                    if len(candidates) == 1:
                        after_onset = round(candidates[0]["end_ms"]-200)
                reference = onset-before
                after = after_onset-reference
                errors.append(after)
                event = {"surah": s["surah"], "ayah": ayah, "before_ms": before, "after_ms": after,
                         "onset_ms": onset, "candidate_ms": after_onset}
                if abs(before) <= 500 < abs(after):
                    new.append(event)
                if abs(after) <= 500 < abs(before):
                    removed.append(event)
                if onset != after_onset:
                    moves.append(event)
                unflagged += abs(after) > 500 and check["reliable"]
        out.append({"reciter":reciter,"policy":policy,"stats":stats(errors),"new":new,"removed":removed,
                    "moves":moves,"unflagged_large":unflagged})
(TASK / "candidate-evaluation.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
for r in out:
    print(r["reciter"], r["policy"], r["stats"], "new/removed", len(r["new"]),len(r["removed"]))
