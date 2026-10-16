"""Narrow rollout to tested Base, bound new future recovery, verify cue provenance."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(roots['B3']).with_name('B4')
assert not root.exists()
shutil.copytree(roots['B3'], root, ignore=shutil.ignore_patterns('__pycache__'))
file = root / 'surah_cutter/core/fast_executor.py'
code = file.read_text()
code = code.replace('if (not anchor and _partial_asr_head(next_obs, next_phrase, threshold)):',
                    'if (not anchor and adapter.model_id == "whisper-tarteel-base"\n                                        and _partial_asr_head(next_obs, next_phrase, threshold)):', 1)
code = code.replace('if not expansion and _partial_asr_head(obs, phrase, threshold):',
                    'if (not expansion and adapter.model_id == "whisper-tarteel-base"\n                            and _partial_asr_head(obs, phrase, threshold)):', 1)
needle = 'and (partial_head_route is None or not _strong_continuation_conflict(\n                                        next_obs.units'
assert needle in code
code = code.replace(needle, '''and (partial_head_route is None or future_ayah - ayah <= policy.expansion_lookahead)
                                    and (partial_head_route is None or not _strong_continuation_conflict(
                                        next_obs.units''', 1)
needle = '        recovery = proof.get("opening_recovery")'
assert needle in code
code = code.replace(needle, '''        if "partial_asr_head_cue_id" in proof:
            cue_id = proof["partial_asr_head_cue_id"]
            if type(cue_id) is not int or not 0 <= cue_id < boundary.read_id:
                raise ValueError("invalid_partial_head_cue")
            cue = reads[cue_id]
            if (cue.get("outcome") != "completed"
                    or cue["window_ms"][0] != read["window_ms"][0]
                    or cue["window_ms"][1] >= read["window_ms"][1]):
                raise ValueError("unrelated_partial_head_cue")
            if groups is not None:
                from surah_cutter.core.fast_adapters import Observation
                initial = cue.get("observed", {})
                cue_obs = Observation(tuple(initial.get("units", ())), initial.get("positions_ms"),
                    initial.get("quality", 0.), initial.get("truncated", True),
                    unit_quality=tuple(initial.get("unit_quality", ())),
                    attribution=tuple(initial.get("attribution", ())))
                if not _asr_head_competitors(cue_obs, groups, word_ends, threshold):
                    raise ValueError("unsupported_partial_head_cue")
                if _strong_continuation_conflict(observed["units"], observed["unit_quality"],
                                                groups[boundary.ayah], len(phrase), threshold):
                    raise ValueError("contradictory_partial_head_continuation")
''' + needle, 1)
compile(code, str(file), 'exec')
file.write_text(code)
roots['B4'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(roots, indent=2) + '\n')
(P / 'B4-sha.json').write_text(json.dumps({'sha256': hashlib.sha256(code.encode()).hexdigest(),
    'changes': ['Base-only rollout', 'new future anchor within2 ayahs', 'validator cue provenance and continuation'],
    'control_results_seen': False}, indent=2) + '\n')
