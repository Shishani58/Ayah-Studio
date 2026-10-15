"""Isolated body cue only authorizes bounded future search after current failure."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

task = Path(__file__).parent.resolve()
prior = task.parent / "2026-10-14-fast-dsp-integrated-corpus"
original = Path((prior / "snapshot-path.txt").read_text().strip())
manifest = json.loads((prior / "source-manifest.json").read_text())
assert all(hashlib.sha256((original/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
source = Path(tempfile.mkdtemp(prefix="fast-body-future-")) / "source"
shutil.copytree(original,source,symlinks=True,ignore=shutil.ignore_patterns("__pycache__"))
file = source / "surah_cutter/core/fast_executor.py"
text = file.read_text().replace("sequential-fast-v3-sparse-15-late-future-only-probe",
                               "sequential-fast-v3-sparse-17-body-future-cue-probe")
start = text.index("            if (boundary is None and future is None and policy.sparse_anchors")
end = text.index("            if boundary is None:\n                if (future and policy.sparse_anchors",start)
original_late = text[start:end]
search_loop = original_late[original_late.index("                for index, pause in selected:"):]
search_loop = search_loop.replace("late_future_only", "body_future_only").replace("LATE_FUTURE_PROVED", "BODY_FUTURE_PROVED")
body = '''            if (boundary is None and future is None and policy.sparse_anchors
                    and provisional_current_pause is None and adapter.timing
                    and adapter.metric == "ctc_greedy_probability" and len(confirmed) >= 2
                    and getattr(adapter, "mapping", None) is not None):
                body_owners = {}
                for word_index in range(len(rendered) - 1):
                    pair = rendered[word_index] + rendered[word_index + 1]
                    body_owners.setdefault(pair, []).append(word_index)
                current_words = [i for i, w in enumerate(words) if w.ayah == ayah]
                body_cue = None
                for read in reads.calls:
                    if not read["reason"].startswith(f"ayah={ayah}:") or "previous_tail" in read["reason"]:
                        continue
                    observed = read.get("observed", {})
                    positions = observed.get("positions_ms")
                    if observed.get("truncated") or not positions:
                        continue
                    units, qualities = observed["units"], observed["unit_quality"]
                    for wi in current_words[1:-1]:
                        pair = rendered[wi] + rendered[wi + 1]
                        if body_owners[pair] != [wi]:
                            continue
                        for offset in range(len(units) - len(pair) + 1):
                            quality = qualities[offset:offset + len(pair)]
                            if (tuple(units[offset:offset + len(pair)]) == pair
                                    and len(quality) == len(pair) and all(math.isfinite(q) for q in quality)
                                    and min(quality) >= threshold
                                    and not any("spoken_alias" in item for item in
                                                observed.get("attribution", [])[offset:offset + len(pair)])):
                                body_cue = (read["window_ms"][0] + positions[offset], wi, read["id"],
                                            pair, offset, min(quality))
                                break
                        if body_cue:
                            break
                    if body_cue:
                        break
                if body_cue:
                    cue_ms, wi, cue_read_id, body_phrase, body_offset, body_quality = body_cue
                    raw_spans = adapter.mapping["spans"]
                    assert len(raw_spans) == len(words)
                    remaining = sum(raw_spans[i][1] - raw_spans[i][0] for i in current_words if i >= wi)
                    prior_ayah = confirmed[-1].ayah
                    consumed = sum(len(u) for a in ayahs if confirmed[0].ayah <= a < prior_ayah
                                   for u in tempo_groups[a])
                    tempo = (confirmed[-1].start_ms - confirmed[0].start_ms) / max(1, consumed)
                    span = tempo * remaining
                    forecast = cue_ms + span
                    left = cue_ms + span * policy.tempo_lower_ratio
                    right = min(duration, cue_ms + span * policy.tempo_upper_ratio)
                    forecasts = {ayah + 1: forecast,
                                 ayah + 2: forecast + tempo * sum(map(len, tempo_groups.get(ayah + 1, ())))}
                    pool = [(i, p) for i, p in enumerate(pauses[:initial_pause_count])
                            if cue_ms < p.center_ms and left <= p.center_ms <= right and geometry[i]["accepted"]]
                    reserved = {}
                    for expected in forecasts.values():
                        for index, pause in sorted(pool, key=lambda item: (abs(item[1].center_ms - expected), item[0]))[:policy.max_candidates // 2]:
                            reserved[index] = pause
                    for index, pause in sorted(pool, key=lambda item: (min(abs(item[1].center_ms - f) for f in forecasts.values()), item[0])):
                        if len(reserved) >= policy.max_candidates:
                            break
                        reserved[index] = pause
                    selected = sorted(reserved.items(), key=lambda item: item[1].center_ms)
                    event("BODY_FUTURE_SEARCH", ayah, cue_read_id=cue_read_id, word_index=wi,
                          body_phrase=body_phrase, body_offset=body_offset, body_quality=body_quality,
                          cue_ms=cue_ms, forecasts=forecasts, region_ms=[left, right],
                          candidates=[i for i, _ in selected], scope="body search only; current onset unresolved")
'''
# The established future proof/alias/expansion gates are reused under body_cue.
body += "\n".join("    " + line if line else line for line in search_loop.splitlines()) + "\n"
text = text[:end] + body + text[end:]
file.write_text(text)
(task / "snapshot-path.txt").write_text(str(source) + "\n")
out = {}
with zipfile.ZipFile(task / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for key in manifest:
        q = source/key
        out[key] = hashlib.sha256(q.read_bytes()).hexdigest()
        archive.write(q,key)
(task / "source-manifest.json").write_text(json.dumps(out,indent=2)+"\n")
print(source)
