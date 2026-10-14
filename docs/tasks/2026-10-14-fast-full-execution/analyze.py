"""Recompute paired gates from frozen corpus artifacts; no model/reference access."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent

def metrics(items):
    rows = [(i, a, ms, conf, error) for i in items
            for (a, ms, conf), error in zip(i["rows"], i["errors_ms"])]
    errors = [r[4] for r in rows]
    model_time_available = all("seconds" in r for i in items for r in i["trace"]["acoustic_reads"])
    acoustic = (sum(sum(r["seconds"] for r in i["trace"]["acoustic_reads"]) for i in items)
                if model_time_available else None)
    executor = sum(i["seconds"] for i in items)
    supplemental_dsp = sum(i["trace"].get("dsp_seconds", 0) for i in items)
    stops = Counter(b["review_reason"] for i in items for b in i["trace"].get("boundaries", [])
                    if b["evidence_kind"] == "estimate" and b["review_reason"] != "blocked_by_unresolved_ayah")
    return {"recordings": len(items), "ayahs": len(rows),
            "positive_conf": sum(r[3] > 0 for r in rows),
            "positive_conf_and_review": sum(conf > 0 and a in i["trace"]["review_required_ayahs"] for i,a,ms,conf,err in rows),
            "confirmed": sum(conf > 0 and a not in i["trace"]["review_required_ayahs"] for i,a,ms,conf,err in rows),
            "review": sum(a in i["trace"]["review_required_ayahs"] for i,a,ms,conf,err in rows),
            "positive_conf_over500": [[i["surah"], a, err, conf] for i,a,ms,conf,err in rows if conf > 0 and err > 500],
            "confirmed_over500": [[i["surah"], a, err, conf] for i,a,ms,conf,err in rows
                                  if conf > 0 and err > 500 and a not in i["trace"]["review_required_ayahs"]],
            "all_over500": sum(e > 500 for e in errors), "median_ms": float(np.median(errors)),
            "p95_ms": float(np.percentile(errors,95)), "max_ms": max(errors),
            "executor_seconds": executor, "local_model_seconds": acoustic,
            "supplemental_dsp_seconds": supplemental_dsp,
            "executor_other_seconds": executor-acoustic-supplemental_dsp if acoustic is not None else None,
            "shared_dsp_seconds": sum(i["dsp_seconds"] for i in items),
            "decode_seconds": sum(i["decode_seconds"] for i in items),
            "model_calls": sum(len(i["actual_dispatches"]) for i in items),
            "denied_calls": sum(len(i["trace"].get("denied_acoustic_reads", [])) for i in items),
            "cache_hits": sum(i["trace"].get("cache_hits", 0) for i in items),
            "peak_accounted_cache_bytes": (max(i["trace"].get("peak_cache_bytes", 0) for i in items)
                if all("fast_adapter" in i["trace"] or i["model"] == "quran-phonetics" for i in items) else None),
            "first_ayah_confirmed": sum(i["rows"][0][2] > 0 and 1 not in i["trace"]["review_required_ayahs"] for i in items),
            "first_ayah_over500_without_review": sum(i["rows"][0][2] > 0 and 1 not in i["trace"]["review_required_ayahs"] and i["errors_ms"][0] > 500 for i in items),
            "boundary_evidence_kinds": dict(Counter(b["evidence_kind"] for i in items for b in i["trace"].get("boundaries", []))),
            "max_local_ms": max((r["samples"]/16 for i in items for r in i["actual_dispatches"]), default=0),
            "max_union_ratio": max(i["trace"]["coverage"] for i in items),
            "max_sum_ratio": max(i["trace"]["audio_ms"]/i["duration_ms"] for i in items),
            "total_model_audio_ms": sum(i["trace"]["audio_ms"] for i in items),
            "total_union_ms": sum(i["trace"]["covered_audio_ms"] for i in items),
            "timeline_violations": [{"surah":i["surah"],**v} for i in items for v in i["sequence_violations"]],
            "strict_stop_reasons": dict(stops),
            "max_cumulative_rss_self_bytes": max(i["peak_rss_self"] for i in items),
            "max_cumulative_rss_children_bytes": max(i["peak_rss_children"] for i in items)}

new=json.loads((ROOT/"corpus-final.json").read_text())
old=json.loads((ROOT/"corpus-legacy.json").read_text())
models={"quran-phonetics","whisper-tarteel-tiny","whisper-tarteel-base"}
for data,surahs,rows_per_model in ((new,{1,2,9,*range(78,115)},986),(old,set(range(78,115)),564)):
    pairs=[(i["model"],i["surah"]) for i in data]
    assert len(pairs)==len(set(pairs)), "duplicate model/surah pair"
    assert set(pairs)=={(model,surah) for model in models for surah in surahs}, "incomplete corpus"
    for model in models:
        assert sum(len(i["rows"]) for i in data if i["model"]==model)==rows_per_model
    assert all(len(i["errors_ms"])==len(i["rows"]) for i in data), "reference evaluation incomplete"
result={}
for model in sorted({i["model"] for i in new}):
    current=[i for i in new if i["model"]==model]
    legacy=[i for i in old if i["model"]==model]
    paired=[i for i in current if i["surah"] in {j["surah"] for j in legacy}]
    for i in paired:
        j=next(j for j in legacy if j["surah"]==i["surah"])
        for key in ("audio_sha256","initial_pauses_sha256","reference_sha256"):
            assert i[key]==j[key], (model,i["surah"],key)
    n,o=metrics(paired),metrics(legacy)
    previous={(i["surah"],a):(conf if a not in i["trace"]["review_required_ayahs"] else 0,err) for i in legacy for (a,_,conf),err in zip(i["rows"],i["errors_ms"])}
    disagreements=[]
    for surah,ayah,error,conf in n["confirmed_over500"]:
        prev_conf,prev_err=previous[surah,ayah]
        if prev_conf<=0 or prev_err<=500:
            disagreements.append([surah,ayah,error,conf])
    result[model]={"all40":metrics(current),"additional1_2_9":metrics([i for i in current if i["surah"]<78]),
        "paired37_new":n,"paired37_legacy":o,
        "coverage_gate":n["confirmed"]>=o["confirmed"],
        "false_confident_gate":len(n["confirmed_over500"])<=len(o["confirmed_over500"]),
        "new_unlistened_confirmed_disagreements":disagreements}
(ROOT/"paired-analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print("Complete paired corpus: 3 models x 37 surahs; coverage gate failed for all three.")
