"""Late future-only search runs after all existing current recovery attempts fail."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

task = Path(__file__).parent.resolve()
prior = task.parent / '2026-10-14-fast-previous-body-rejection'
original = Path((prior / 'snapshot-path.txt').read_text().strip())
manifest = json.loads((prior / 'source-manifest.json').read_text())
assert all(hashlib.sha256((original / k).read_bytes()).hexdigest() == v for k, v in manifest.items())
source = Path(tempfile.mkdtemp(prefix='fast-late-future-')) / 'source'
shutil.copytree(original, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
file = source / 'surah_cutter/core/fast_executor.py'
text = file.read_text()


def change(old, new):
    global text
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)


change('sequential-fast-v3-sparse-14-previous-body-rejection-probe',
       'sequential-fast-v3-sparse-15-late-future-only-probe')
change('            boundary, blocked_prefix, future = None, False, None',
       '            boundary, blocked_prefix, future = None, False, None\n            provisional_current_pause = None')
cue = '''                    if (provisional_current_pause is None and pause is not None and not obs.truncated
                            and not alias_conflict
                            and _prefix(obs.units, groups[ayah][:canonical_ends[ayah][0]]) == canonical_ends[ayah][0]
                            and min(obs.unit_quality[:canonical_ends[ayah][0]], default=0.) >= threshold):
                        provisional_current_pause = (index, pause)
'''
change('                    future_expansion = None\n', cue + '                    future_expansion = None\n')
late = '''            if (boundary is None and future is None and policy.sparse_anchors
                    and len(confirmed) >= 2 and provisional_current_pause is not None):
                cue_index, cue_pause = provisional_current_pause
                prior_ayah = confirmed[-1].ayah
                consumed = sum(len(u) for a in ayahs if confirmed[0].ayah <= a < prior_ayah
                               for u in tempo_groups[a])
                tempo = (confirmed[-1].start_ms - confirmed[0].start_ms) / max(1, consumed)
                span = tempo * sum(map(len, tempo_groups[ayah]))
                center = cue_pause.center_ms + span
                left = cue_pause.center_ms + span * policy.tempo_lower_ratio
                right = min(duration, cue_pause.center_ms + span * policy.tempo_upper_ratio)
                pool = [(i, p) for i, p in enumerate(pauses[:initial_pause_count])
                        if p.center_ms > cue_pause.end_ms and left <= p.center_ms <= right
                        and geometry[i]["accepted"]]
                selected = sorted(sorted(pool, key=lambda item: (abs(item[1].center_ms - center), item[0]))
                                  [:policy.max_candidates], key=lambda item: item[1].center_ms)
                event("LATE_FUTURE_SEARCH", ayah, cue_pause_id=cue_index, forecast_ms=center,
                      region_ms=[left, right], candidates=[i for i, p in selected],
                      scope="current remains unresolved; globally distinct future only")
                for index, pause in selected:
                    for speech in policy.speech_ms:
                        obs = reads.read(max(pause.start_ms, pause.end_ms - 200.),
                                         min(duration, pause.end_ms + speech),
                                         reason=f"ayah={ayah}:late_future_only:pause={index}")
                        if obs is None:
                            break
                        read_id = reads.last_read_id
                        expand = False
                        for future_ayah in ayahs:
                            if not ayah < future_ayah <= ayah + policy.expansion_lookahead:
                                continue
                            next_plan = planned_opening(future_ayah)
                            if next_plan["ambiguous"] or next_plan.get("contextual_owner"):
                                continue
                            next_phrase = next_plan["units"]
                            view, future_proof = _beam_view(obs, next_phrase, threshold, policy.beam_log_margin)
                            owner = rendered[word_map[future_ayah][0]]
                            spoken = len(proof_ends[word_map[future_ayah][0]]) > 1
                            conflict = any("spoken_alias" in item and tuple(item["source_units"]) != owner
                                           for item in view.attribution[:len(next_phrase)])
                            if spoken:
                                conflict |= any(expected != actual and view.unit_quality[i] >= threshold
                                                for i, (expected, actual) in enumerate(zip(owner, view.units)))
                                future_proof["spoken_owner_units"] = owner
                            quality = min(view.unit_quality[:len(next_phrase)], default=0.)
                            if (not conflict and not view.truncated and quality >= threshold
                                    and _prefix(view.units, next_phrase) == len(next_phrase)):
                                future = (future_ayah, next_phrase, pause, index, read_id, quality, future_proof)
                                event("LATE_FUTURE_PROVED", ayah, future_ayah=future_ayah, read_id=read_id)
                                break
                            if (not conflict and _expansion_evidence(view, next_phrase, word_ends[future_ayah],
                                                                     threshold, groups, future_ayah, word_ends)):
                                expand = True
                        if future or not expand:
                            break
                    if future:
                        break
'''
change('''            if boundary is None:
                if (future and policy.sparse_anchors''', late + '''            if boundary is None:
                if (future and policy.sparse_anchors''')
file.write_text(text)
(task / 'snapshot-path.txt').write_text(str(source) + '\n')
out = {}
with zipfile.ZipFile(task / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    files = [p for group in ['surah_cutter', 'tools'] for p in (source / group).rglob('*.py')]
    files.append(source / 'pyproject.toml')
    for p in files:
        relative = p.relative_to(source).as_posix()
        archive.write(p, relative)
        out[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
(task / 'source-manifest.json').write_text(json.dumps(out, indent=2) + '\n')
print(source)
