"""Isolated native timing at a DSP pause whose midpoint cannot be proved."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

task = Path(__file__).parent.resolve()
previous = task.parent / '2026-10-14-fast-pronunciation-variants'
original = Path((previous / 'raw-tempo-snapshot-path.txt').read_text().strip())
source = Path(tempfile.mkdtemp(prefix='fast-pause-native-onset-')) / 'source'
shutil.copytree(original, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
path = source / 'surah_cutter/core/fast_executor.py'
text = path.read_text()


def change(old, new, count=1):
    global text
    assert text.count(old) == count, (old, text.count(old))
    text = text.replace(old, new)


change('sequential-fast-v3-sparse-11-sampled-onset-probe',
       'sequential-fast-v3-sparse-12-pause-native-onset-probe')
change('''                    if not geometry[index]["accepted"]:
                        reason = geometry[index]["reason"]
                        event("SELECT_CANDIDATE", ayah, candidate=index, declined_geometry=reason)
                        continue''', '''                    if not geometry[index]["accepted"]:
                        reason = geometry[index]["reason"]
                        event("SELECT_CANDIDATE", ayah, candidate=index, declined_geometry=reason)
                        if (adapter.timing and reason == "decaying_tail_midpoint_unproved"
                                and not plan["ambiguous"] and not plan.get("contextual_owner")):
                            kind = "pause_native_onset"
                        else:
                            continue''')
change('if (_prefix(obs.units, phrase) != len(phrase) and pause and not plan["ambiguous"]):',
       'if (_prefix(obs.units, phrase) != len(phrase) and kind == "pause" and not plan["ambiguous"]):')
change('if (pause and adapter.metric == "ctc_greedy_probability"',
       'if (kind == "pause" and adapter.metric == "ctc_greedy_probability"')
change('if (pause and confirmed and confirmed[-1].ayah == ayah - 1',
       'if (kind == "pause" and confirmed and confirmed[-1].ayah == ayah - 1')
change('if pause is not None and obs.units:', 'if kind == "pause" and obs.units:')
change('if kind == "first_onset" and obs.positions_ms is None:',
       'if kind in ("first_onset", "pause_native_onset") and obs.positions_ms is None:')
change('''                        if pause:
                            ms = round(pause.center_ms)''', '''                        if kind == "pause":
                            ms = round(pause.center_ms)''')
change('''                        else:
                            ms = round(reads.calls[reads.last_read_id]["window_ms"][0] + obs.positions_ms[0])''',
       '''                        else:
                            ms = round(reads.calls[opening_id]["window_ms"][0] + obs.positions_ms[0])
                            if kind == "pause_native_onset":
                                proof.update(pause_ms=[pause.start_ms, pause.end_ms], pause_id=index)''')
validator = '''        elif boundary.evidence_kind == "pause_native_onset":
            if recovery or proof.get("contextual_owner") or offset != 0 or pauses is None:
                raise ValueError("unsupported_native_pause_owner")
            index = proof.get("pause_id")
            if type(index) is not int or not 0 <= index < len(pauses):
                raise ValueError("unknown_native_pause")
            pause = pauses[index]
            if proof.get("pause_ms") != [pause.start_ms, pause.end_ms]:
                raise ValueError("mutated_native_pause")
            if (pause_geometry_data is None or index >= len(pause_geometry_data)
                    or pause_geometry_data[index]["accepted"]
                    or pause_geometry_data[index]["reason"] != "decaying_tail_midpoint_unproved"):
                raise ValueError("unsupported_native_pause_geometry")
            r0, r1 = read["window_ms"]
            positions = observed["positions_ms"]
            if (positions is None or not positions or abs(r0 - max(pause.start_ms, pause.end_ms - 200.)) > .1
                    or r1 <= pause.end_ms or not r0 <= ms < r1
                    or ms != round(r0 + positions[0])):
                raise ValueError("native_pause_onset_mismatch")
'''
change('        elif boundary.evidence_kind in ("first_onset", "wasl"):',
       validator + '        elif boundary.evidence_kind in ("first_onset", "wasl"):')
path.write_text(text)
(task / 'native-onset-snapshot-path.txt').write_text(str(source) + '\n')
manifest = {}
with zipfile.ZipFile(task / 'native-onset-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for group in ['surah_cutter', 'tools']:
        for file in (source / group).rglob('*.py'):
            relative = file.relative_to(source).as_posix()
            archive.write(file, relative)
            manifest[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
(task / 'native-onset-source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(source)
