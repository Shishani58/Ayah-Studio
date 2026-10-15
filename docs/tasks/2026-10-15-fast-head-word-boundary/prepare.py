"""Frozen causal head-boundary retry, leaving all confirmation gates intact."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

task=Path(__file__).parent.resolve()
base=Path(tempfile.mkdtemp(prefix="fast-head-boundary-"))
paths={}
for label in ["before","after"]:
    root=base/label
    root.mkdir()
    for name in ["surah_cutter","tools"]:
        shutil.copytree(name,root/name,ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy("pyproject.toml",root/"pyproject.toml")
    for name in ["data","models"]:
        (root/name).symlink_to(Path(name).resolve(),target_is_directory=True)
    paths[label]=str(root)
file=Path(paths["after"])/"surah_cutter/core/fast_executor.py"
text=file.read_text()
assert text.count("sequential-fast-v3-sparse-10")==1
text=text.replace("sequential-fast-v3-sparse-10","sequential-fast-v3-sparse-11-head-boundary-probe")
block='''            if (boundary is None and future is None and policy.sparse_anchors
                    and adapter.model_id in {"whisper-tarteel-tiny", "whisper-tarteel-base"}
                    and not adapter.timing and len(confirmed) >= 2 and not plan["ambiguous"]
                    and not plan.get("contextual_owner") and canonical_ends[ayah][0] == 1
                    and len(phrase) > 1 and len(phrase[0]) >= 3):
                current_words = [i for i, w in enumerate(words) if w.ayah == ayah]
                owners = {}
                for wi in range(len(rendered) - 1):
                    owners.setdefault(rendered[wi] + rendered[wi + 1], []).append(wi)
                current_reads = [r for r in reads.calls if r["reason"].startswith(f"ayah={ayah}:")
                                 and "previous_tail" not in r["reason"]]
                body_cue = None
                for r in current_reads:
                    observed = r.get("observed", {})
                    if observed.get("truncated"):
                        continue
                    units, qualities = observed.get("units", ()), observed.get("unit_quality", ())
                    for wi in current_words[1:-1]:
                        pair = rendered[wi] + rendered[wi + 1]
                        if owners[pair] != [wi]:
                            continue
                        for offset in range(len(units) - len(pair) + 1):
                            q = qualities[offset:offset + len(pair)]
                            if (tuple(units[offset:offset + len(pair)]) == pair
                                    and len(q) == len(pair) and all(math.isfinite(x) for x in q)
                                    and min(q) >= threshold and not any("spoken_alias" in a for a in
                                        observed.get("attribution", ())[offset:offset + len(pair)])):
                                body_cue = (r["id"], wi, pair)
                                break
                        if body_cue:
                            break
                    if body_cue:
                        break
                if body_cue:
                    tried_head_candidates = set()
                    for initial in current_reads:
                        observed = initial.get("observed", {})
                        units, qualities = observed.get("units", ()), observed.get("unit_quality", ())
                        if (observed.get("truncated") or not units or not qualities or qualities[0] <= 0
                                or not units[0].startswith(phrase[0]) or len(units[0]) <= len(phrase[0])
                                or any("unsupported" in a for a in observed.get("attribution", ()))):
                            continue
                        for kind, index, _, _, pause in candidates_here[:policy.max_candidates]:
                            if pause is None or index in tried_head_candidates or not geometry[index]["accepted"]:
                                continue
                            start = max(pause.start_ms, pause.end_ms - 200.)
                            if round(initial["window_ms"][0] * sr / 1000) != round(start * sr / 1000):
                                continue
                            tried_head_candidates.add(index)
                            for speech in policy.speech_ms[1:]:
                                event("HEAD_BOUNDARY_RECHECK", ayah, candidate=index,
                                      initial_read_id=initial["id"], body_read_id=body_cue[0],
                                      observed_word=units[0], expected_prefix=phrase[0],
                                      window_ms=[start, min(duration, pause.end_ms + speech)])
                                recheck = reads.read(start, min(duration, pause.end_ms + speech),
                                                     reason=f"ayah={ayah}:head_word_boundary_recheck")
                                if recheck is None:
                                    break
                                q = recheck.unit_quality[:len(phrase)]
                                conflict = any("spoken_alias" in a and tuple(a["source_units"]) != rendered[word_map[ayah][0]]
                                               for a in recheck.attribution[:len(phrase)])
                                if (_prefix(recheck.units, phrase) == len(phrase) and not recheck.truncated
                                        and len(q) == len(phrase) and min(q) >= threshold and not conflict):
                                    ms = round(pause.center_ms)
                                    if ms <= previous:
                                        break
                                    proof = {"phrase": phrase, "complete": True, "ambiguous": False,
                                             "competitors_checked": True, "competitor_count": plan["competitor_count"],
                                             "offset": 0, "quality_metric": adapter.metric,
                                             "pause_ms": [pause.start_ms, pause.end_ms], "pause_id": index,
                                             "head_boundary_recheck": True, "body_cue_read_id": body_cue[0],
                                             "initial_head_read_id": initial["id"]}
                                    boundary = Boundary(ayah, ms, "pause", min(q), read_id=reads.last_read_id, proof=proof)
                                    validate_boundaries([boundary], [ayah], duration, reads.calls,
                                                        groups=groups, word_ends=word_ends, canonical_ends=canonical_ends,
                                                        threshold=threshold, pauses=pauses, policy=policy, pause_geometry_data=geometry)
                                    break
                                if not _expansion_evidence(recheck, phrase, word_ends[ayah], threshold, groups, ayah, word_ends):
                                    break
                            if boundary:
                                break
                        if boundary:
                            break
'''
marker="            if boundary is None:\n                if (future and policy.sparse_anchors"
assert text.count(marker)==1
text=text.replace(marker,block+marker)
file.write_text(text)
(task/"snapshot-paths.json").write_text(json.dumps(paths,indent=2)+"\n")
manifest={}
with zipfile.ZipFile(task/"source-pair.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for label,d in paths.items():
        root=Path(d)
        manifest[label]={}
        for q in [x for group in ["surah_cutter","tools"] for x in (root/group).rglob("*.py")]+[root/"pyproject.toml"]:
            key=str(q.relative_to(root))
            manifest[label][key]=hashlib.sha256(q.read_bytes()).hexdigest()
            archive.write(q,f"{label}/{key}")
(task/"source-manifests.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(base)
