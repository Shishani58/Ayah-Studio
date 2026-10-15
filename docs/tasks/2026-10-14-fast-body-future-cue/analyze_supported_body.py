"""Static review of retained Tarteel body cues; no model calls or tests."""
import json
from pathlib import Path
from surah_cutter.core.fast_adapters import arabic_units
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.core.model_worker import whisper_alignment_text
from surah_cutter.data import quran_text

task = Path(__file__).parent.resolve()
prior = task.parent / "2026-10-14-fast-previous-body-rejection"
results = []
for model in ["whisper-tarteel-tiny","whisper-tarteel-base"]:
    for c in json.loads((prior / f"{model}-after.json").read_text()):
        words = quran_text.surah_words(c["surah"])
        rendered = tuple(arabic_units(whisper_alignment_text(t)) for t in inference_words(words,model))
        owners = {}
        for i in range(len(words)-1):
            owners.setdefault(rendered[i]+rendered[i+1],[]).append(i)
        unresolved = {b["ayah"] for b in c["trace"]["boundaries"] if b["confidence"]<=0}
        for r in c["trace"]["acoustic_reads"]:
            reason = r["reason"]
            if not reason.startswith("ayah=") or "previous_tail" in reason:
                continue
            ayah = int(reason.split(":")[0].split("=")[1])
            if ayah not in unresolved:
                continue
            o = r["observed"]
            if o["truncated"]:
                continue
            current = [i for i,w in enumerate(words) if w.ayah==ayah]
            for i in current[1:-1]:
                pair = rendered[i]+rendered[i+1]
                if owners[pair]!=[i]:
                    continue
                for offset in range(len(o["units"])-len(pair)+1):
                    q = o["unit_quality"][offset:offset+len(pair)]
                    if tuple(o["units"][offset:offset+len(pair)])==pair and min(q)>=.25:
                        results.append({"model":model,"surah":c["surah"],"ayah":ayah,"read_id":r["id"],
                                        "word_indices":[i,i+1],"canonical_words":[words[i].text,words[i+1].text],
                                        "body_units":pair,"min_quality":min(q),"read_window_ms":r["window_ms"],
                                        "positions_available":o["positions_ms"] is not None,
                                        "scope":"body search cue only, no current onset"})
(task / "supported-body-cue-analysis.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(results,ensure_ascii=False,indent=2))
