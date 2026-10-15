"""Reject only a strong complete previous-body rival behind a shared first word."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

task = Path(__file__).parent.resolve()
prior = task.parent / '2026-10-14-fast-intra-ayah-waqf'
original = Path((prior / 'context-snapshot-path.txt').read_text().strip())
manifest = json.loads((prior / 'context-source-manifest.json').read_text())
assert all(hashlib.sha256((original / k).read_bytes()).hexdigest() == v for k, v in manifest.items())
source = Path(tempfile.mkdtemp(prefix='fast-previous-body-')) / 'source'
shutil.copytree(original, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
path = source / 'surah_cutter/core/fast_executor.py'
text = path.read_text()
old = 'sequential-fast-v3-sparse-13-heard-context-expansion-probe'
assert text.count(old) == 1
text = text.replace(old, 'sequential-fast-v3-sparse-14-previous-body-rejection-probe')
hook = '''                    future_expansion = None
'''
assert text.count(hook) == 1
replacement = '''                    if (confirmed and confirmed[-1].ayah == ayah - 1 and not obs.truncated
                            and len(canonical_ends[ayah]) > 1
                            and len(phrase) > canonical_ends[ayah][0]):
                        first_end = canonical_ends[ayah][0]
                        first_word = groups[ayah][:first_end]
                        previous_ayah = confirmed[-1].ayah
                        previous_ends = canonical_ends[previous_ayah]
                        rival = None
                        if _prefix(obs.units, first_word) == len(first_word):
                            for word_index in range(1, len(previous_ends) - 1):
                                start_unit = previous_ends[word_index - 1]
                                end_first, end_second = previous_ends[word_index:word_index + 2]
                                previous_first = groups[previous_ayah][start_unit:end_first]
                                previous_second = groups[previous_ayah][end_first:end_second]
                                current_second = groups[ayah][first_end:canonical_ends[ayah][1]]
                                alternative = previous_first + previous_second
                                quality = obs.unit_quality[:len(alternative)]
                                if (previous_first == first_word and previous_second != current_second
                                        and words[word_map[previous_ayah][start_unit]].text == words[word_map[ayah][0]].text
                                        and words[word_map[previous_ayah][end_first]].text != words[word_map[ayah][first_end]].text
                                        and _prefix(obs.units, alternative) == len(alternative)
                                        and len(quality) == len(alternative)
                                        and all(math.isfinite(q) for q in quality)
                                        and min(quality, default=0.) >= threshold
                                        and _strong_continuation_conflict(obs.units, obs.unit_quality,
                                                                         groups[ayah], first_end, threshold)):
                                    rival = {"ayah": previous_ayah, "start_unit": start_unit,
                                             "units": alternative, "min_quality": min(quality)}
                                    break
                        if rival:
                            reason = "decoded_previous_ayah_body"
                            event("DECLINE_CANDIDATE", ayah, candidate=index, reason=reason, rival=rival,
                                  read_id=opening_id)
                            had_prefix = False
                            blocked_prefix = False
                            break
''' + hook
text = text.replace(hook, replacement)
path.write_text(text)
(task / 'snapshot-path.txt').write_text(str(source) + '\n')
out = {}
with zipfile.ZipFile(task / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    files = [f for group in ['surah_cutter', 'tools'] for f in (source / group).rglob('*.py')]
    files.append(source / 'pyproject.toml')
    for file in files:
        name = file.relative_to(source).as_posix()
        archive.write(file, name)
        out[name] = hashlib.sha256(file.read_bytes()).hexdigest()
(task / 'source-manifest.json').write_text(json.dumps(out, indent=2) + '\n')
print(source)
