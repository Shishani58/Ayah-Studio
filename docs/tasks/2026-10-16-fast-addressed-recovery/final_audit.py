"""Verify protected bytes and the actual merged application's evidence."""
import hashlib
import json
from pathlib import Path
import tempfile

from surah_cutter.core import model_store

P = Path(__file__).resolve().parent
REPO = P.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    return json.loads((P / name).read_text())


protected = load('protected-data-before.json')
assert all(sha(path) == value for path, value in protected.items())
assert (REPO / 'AGENTS.md').read_bytes() == (REPO / 'CLAUDE.md').read_bytes()
assert sha(REPO / 'surah_cutter/core/fast_executor.py') == load('production-adoption.json')['after_sha256']
assert (REPO / 'tests/test_fast_next_pause_recovery.py').read_bytes() == (P / 'production_tests.py').read_bytes()
baseline = load('A-wide.json')
checked_audio, checked_refs = {}, {}
for label, folder in [('A-wide', Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri')),
        *[('final-A-' + r, Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / r)
          for r in ('saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi')]]:
    cases = load(label + '.json')
    command = load(label + '-command.json')['argv']
    reference = Path(command[command.index('--reference') + 1])
    assert sha(reference) == cases[0]['reference_sha256']
    checked_refs[str(reference)] = cases[0]['reference_sha256']
    for case in cases:
        audio = folder / f'{case["surah"]:03d}.mp3'
        assert sha(audio) == case['audio_sha256']
        checked_audio[str(audio)] = case['audio_sha256']
assets = baseline[0]['trace']['run_snapshot']['text_runtime']['assets']
model_path = model_store.installed_path(model_store.spec('whisper-tarteel-base'))
assert all(sha(model_path / name) == value for name, value in assets.items())
silero = baseline[0]['trace']['run_snapshot']['dsp_models']['silero']
assert sha(REPO / 'models/silero_vad.onnx') == silero['sha256']
application = load('application-smoke.json')
assert application['protected_full_review_and_time_unchanged']
assert application['actual_dispatches_equal_frozen_PF']
assert application['old_resume_key_invalidated']
assert Path(application['database']).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
assert Path(application['database']).parent.name.startswith('fast-application-smoke-')
gate = load('SF-corpus-gate.json')
assert gate['passed'] and gate['all_hard_limits_passed']
assert gate['canonical_wide_rows'] == 6236 and gate['paired_recordings'] == 141
cost = load('PF-costs.json')
assert cost['fresh_warm_and_profile_owners_times_match_full_replay']
output = {'passed': True, 'protected_data_files_unchanged': len(protected),
    'audio_files_unchanged': len(checked_audio), 'reference_files_unchanged': len(checked_refs),
    'installed_Base_assets_unchanged': len(assets), 'silero_unchanged': True,
    'agents_claude_identical': True, 'production_matches_measured_candidate': True,
    'application_actual_PCM_and_auto_times_match': True,
    'temporary_project_only': application['database'],
    'edited_verified_full_metadata_and_times_unchanged': True,
    'soft_cost_target_passed': cost['proposed_soft_cost_target']['passed'],
    'checked_audio': checked_audio, 'checked_references': checked_refs}
(P / 'final-audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print({k: v for k, v in output.items() if not k.startswith('checked_')})
