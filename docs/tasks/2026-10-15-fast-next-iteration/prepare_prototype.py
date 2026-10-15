import json,shutil,hashlib,difflib,zipfile
from pathlib import Path
p=Path(__file__).parent.resolve()
paths=json.loads((p/'snapshot-paths.json').read_text())
before=Path(paths['baseline']);after=before.parent/(before.name+'-split')
shutil.copytree(before,after,symlinks=True)
f=after/'surah_cutter/core/fast_executor.py';old=f.read_text();text=old
helper=(p/'split_prototype.py').read_text().split('def split_opening',1)[1]
text=text.replace('def _noisy_opening(', 'def _split_opening'+helper+'\n\ndef _noisy_opening(',1)
text=text.replace('sequential-fast-v3-sparse-11-head-boundary','sequential-fast-v3-sparse-12-split-probe',1)
needle='        recovery = proof.get("opening_recovery")\n        if recovery:'
replacement='''        recovery = proof.get("opening_recovery")
        segmented = proof.get("segmented_opening")
        if segmented:
            from surah_cutter.core.fast_adapters import Observation
            if (boundary.evidence_kind != "pause" or groups is None or word_ends is None
                    or offset != 0 or recovery or proof.get("contextual_owner")
                    or "prefix_comparison_id" in proof or "decoder_hypothesis" in proof):
                raise ValueError("unsupported_segmented_proof")
            decoded = Observation(tuple(observed["units"]), None, observed["quality"],
                observed["truncated"], unit_quality=tuple(observed["unit_quality"]),
                attribution=tuple(observed.get("attribution", ())))
            expected, _ = _split_opening(decoded, groups, word_ends, boundary.ayah, threshold)
            if not expected or expected["segmented_opening"] != segmented or expected["phrase"] != phrase:
                raise ValueError("unheard_segmented_phrase")
            quality = observed["unit_quality"][:segmented["source_count"]]
        elif recovery:'''
assert needle in text;text=text.replace(needle,replacement,1)
text=text.replace('if ((not recovery and len(quality) != len(phrase))','if ((not recovery and not segmented and len(quality) != len(phrase))',1)
text=text.replace('elif not recovery and (phrase != planned["units"]','elif not recovery and not segmented and (phrase != planned["units"]',1)
needle='                    noisy_proof, noisy_expansion = (None, False)'
text=text.replace(needle,'''                    segmented_proof, segmented_expansion = (None, False)
                    if (pause and not adapter.timing and complete != len(phrase)
                            and not beam_proof and not plan.get("contextual_owner")):
                        segmented_proof, segmented_expansion = _split_opening(obs, groups, word_ends, ayah, threshold)
'''+needle,1)
needle='                    if noisy_proof:\n                        recovery = noisy_proof'
text=text.replace(needle,'''                    if segmented_proof:
                        phrase_quality = min(obs.unit_quality[:segmented_proof["segmented_opening"]["source_count"]])
'''+needle,1)
text=text.replace('if ((complete == len(phrase) or noisy_proof)','if ((complete == len(phrase) or noisy_proof or segmented_proof)',1)
needle='                        if noisy_proof:\n                            proof.update(noisy_proof)'
text=text.replace(needle,needle+'''\n                        if segmented_proof:
                            proof.update(segmented_proof)''',1)
needle='                    if not expansion and noisy_expansion:'
text=text.replace(needle,'''                    if not expansion and segmented_expansion:
                        expansion = "split_head_requires_whole_continuation"
'''+needle,1)
f.write_text(text)
(p/'prototype.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True))))
paths['prototype']=str(after);(p/'snapshot-paths.json').write_text(json.dumps(paths,indent=2))
manifest={str(q.relative_to(after)):hashlib.sha256(q.read_bytes()).hexdigest() for q in after.rglob('*.py') if '__pycache__' not in str(q) and q.is_relative_to(after/'surah_cutter')}
(p/'prototype-sha.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(p/'prototype-source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['surah_cutter','tools']:
  for q in (after/name).rglob('*.py'):z.write(q,str(q.relative_to(after)))
 z.write(after/'pyproject.toml','pyproject.toml')
print(after)
