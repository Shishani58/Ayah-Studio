"""Keep wide nonstationary cuts for review in newly recovered ASR chains."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(roots['B2']).with_name('B3')
assert not root.exists()
shutil.copytree(roots['B2'], root, ignore=shutil.ignore_patterns('__pycache__'))
file = root / 'surah_cutter/core/fast_executor.py'
code = file.read_text()
needle = 'def validate_boundaries('
assert needle in code
helper = '''def _review_addressed_recovery_onsets(boundaries, geometry, policy):
    """Newly accessible identity anchors do not resolve a wide decaying cut.

    Retain the internally validated owner/order for sparse recovery. Publish
    uncertain timestamps for review; never infer a replacement onset from ASR.
    The original search path has no new publication restrictions.
    """
    recovered = False
    result, reviews = [], []
    for boundary in boundaries:
        proof = boundary.proof or {}
        recovered |= "partial_asr_head_cue_id" in proof
        index = proof.get("pause_id")
        if (recovered and boundary.evidence_kind == "pause" and index is not None
                and not geometry[index].get("quiet_stationarity", False)
                and geometry[index]["midpoint_interval_radius_ms"] > policy.short_word_pause_radius_ms):
            reason = "recovered_wide_nonstationary_midpoint_unproved"
            reviews.append({"ayah": boundary.ayah, "reason": reason, "read_id": boundary.read_id,
                            "internal_anchor_ms": boundary.start_ms, "identity_proof": proof})
            boundary = replace(boundary, evidence_kind="estimate", confidence=0., review_reason=reason,
                               read_id=None, proof=None)
        result.append(boundary)
    return result, reviews


'''
code = code.replace(needle, helper + needle, 1)
needle = '        for review in temporal_reviews:\n'
assert needle in code
code = code.replace(needle, '''        result, recovery_reviews = _review_addressed_recovery_onsets(result, geometry, policy)
        temporal_reviews.extend(recovery_reviews)
''' + needle, 1)
compile(code, str(file), 'exec')
file.write_text(code)
roots['B3'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(roots, indent=2) + '\n')
(P / 'B3-sha.json').write_text(json.dumps({'sha256': hashlib.sha256(code.encode()).hexdigest(),
    'development_rejection': 'B2 new downstream80:37 has wide530ms nonstationary midpoint, benchmark672ms',
    'rule': 'existing500ms radius cap; identity retained internally, uncertain new cuts review',
    'control_results_seen': False}, indent=2) + '\n')
