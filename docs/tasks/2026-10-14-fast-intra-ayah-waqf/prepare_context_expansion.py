"""Isolate heard-context expansion and full contextual native onset validation."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

task = Path(__file__).parent.resolve()
prior = task.parent / '2026-10-14-fast-named-opening-long'
original = Path((prior / 'native-onset-snapshot-path.txt').read_text().strip())
source = Path(tempfile.mkdtemp(prefix='fast-heard-context-')) / 'source'
shutil.copytree(original, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
path = source / 'surah_cutter/core/fast_executor.py'
text = path.read_text()


def change(old, new, count=1):
    global text
    assert text.count(old) == count, (old, text.count(old))
    text = text.replace(old, new)


change('sequential-fast-v3-sparse-12-pause-native-onset-probe',
       'sequential-fast-v3-sparse-13-heard-context-expansion-probe')
change('if (kind == "pause" and confirmed and confirmed[-1].ayah == ayah - 1',
       'if (kind in ("pause", "pause_native_onset") and confirmed and confirmed[-1].ayah == ayah - 1')
change('if recovery or proof.get("contextual_owner") or offset != 0 or pauses is None:',
       'if recovery or offset != 0 or pauses is None:')
context_validator = '''            if proof.get("contextual_owner"):
                context_id = proof.get("previous_context_read_id")
                if type(context_id) is not int or not 0 <= context_id < len(reads):
                    raise ValueError("missing_native_previous_context")
                context = reads[context_id]
                tail = tuple(proof["previous_tail"])
                context_obs = context.get("observed", {})
                quality = context_obs.get("unit_quality", ())[-len(tail):]
                if (context["outcome"] != "completed" or abs(context["window_ms"][1] - pause.start_ms) > .1
                        or context_obs.get("truncated", True)
                        or tuple(context_obs.get("units", ())[-len(tail):]) != tail
                        or len(quality) != len(tail) or min(quality) < threshold
                        or boundary.confidence > min(quality)
                        or not all(math.isfinite(q) for q in quality)):
                    raise ValueError("unheard_native_previous_context")
'''
change('''                raise ValueError("native_pause_onset_mismatch")
        elif boundary.evidence_kind''',
       '''                raise ValueError("native_pause_onset_mismatch")
''' + context_validator + '        elif boundary.evidence_kind')
expansion = '''                    if (not expansion and kind == "pause_native_onset" and adapter.timing
                            and confirmed and confirmed[-1].ayah == ayah - 1
                            and not plan.get("contextual_owner") and step + 1 < len(speech_steps)):
                        seam = plan_seam(groups, ayah, word_ends)
                        matched = _prefix(obs.units, seam["units"]) if seam else 0
                        if (seam and 3 <= matched < len(seam["units"])
                                and min(obs.unit_quality[:matched], default=0.) >= threshold):
                            combined = seam["tail"] + seam["units"][:matched]
                            collisions = [(a, i) for a in groups
                                          for i in [0] + word_ends[a][:-1]
                                          if (a, i) != (ayah - 1, seam["start_unit"])
                                          and (groups[a] + groups.get(a + 1, ()))[i:i + len(combined)] == combined]
                            if not collisions:
                                for context_ms in policy.speech_ms:
                                    context = reads.read(max(previous, pause.start_ms - context_ms), pause.start_ms,
                                                         reason=f"ayah={ayah}:partial_opening_previous_tail")
                                    tail = seam["tail"]
                                    if (context and not context.truncated
                                            and tuple(context.units[-len(tail):]) == tail
                                            and min(context.unit_quality[-len(tail):], default=0.) >= threshold):
                                        expansion = "heard_distinctive_previous_tail_partial_opening"
                                        current_expansion = True
                                        reads.calls[opening_id]["contextual_expansion"] = {
                                            "previous_context_read_id": reads.last_read_id,
                                            "previous_tail": tail, "heard_opening_fragment": obs.units[:matched],
                                            "scope": "expansion only; full connected opening still required"}
                                        break
'''
change('''                    if not expansion and future_expansion:
                        expansion = "provisional_unique_anchor_partial:" + future_expansion[1]''',
       expansion + '''                    if not expansion and future_expansion:
                        expansion = "provisional_unique_anchor_partial:" + future_expansion[1]''')
path.write_text(text)
(task / 'context-snapshot-path.txt').write_text(str(source) + '\n')
manifest = {}
with zipfile.ZipFile(task / 'context-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for group in ['surah_cutter', 'tools']:
        for file in (source / group).rglob('*.py'):
            relative = file.relative_to(source).as_posix()
            archive.write(file, relative)
            manifest[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
    archive.write(source / 'pyproject.toml', 'pyproject.toml')
    manifest['pyproject.toml'] = hashlib.sha256((source / 'pyproject.toml').read_bytes()).hexdigest()
(task / 'context-source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(source)
