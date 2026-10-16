import sqlite3

import pytest

from surah_cutter.data.alignment_identity import full_resume_suras


def project(tmp_path, statuses, sources, times=(100, 200)):
    path = tmp_path / 'project.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE timings(sura INTEGER,ayah INTEGER,time INTEGER)')
        db.execute('CREATE TABLE review_meta(sura INTEGER,ayah INTEGER,status TEXT,source TEXT)')
        db.executemany('INSERT INTO timings VALUES(1,?,?)', enumerate(times, 1))
        db.executemany('INSERT INTO review_meta VALUES(1,?,?,?)',
                       [(a, status, source) for a, (status, source) in enumerate(zip(statuses, sources), 1)])
    return path


def source(model='quran-phonetics', policy='qpc-phrase-boundary-v3'):
    return f'onnx-ctc:{model}:requested_mode=full:full_onset_policy={policy}'


def test_pause_center_policy_recalculates_former_auto_edges(tmp_path):
    from surah_cutter.core.full_windows import FULL_ONSET_POLICY

    path = project(tmp_path, ['auto', 'auto'], [source()] * 2)
    assert full_resume_suras(path, model_id='quran-phonetics',
                             onset_policy=FULL_ONSET_POLICY, counts={1: 2}) == set()


@pytest.mark.parametrize("old_policy", ["qpc-local-pause-stable-v6", "qpc-two-sided-transition-v8"])
def test_signal_policy_recalculates_old_auto_but_preserves_manual_rows(tmp_path, old_policy):
    from surah_cutter.core.full_windows import FULL_ONSET_POLICY

    path = project(tmp_path, ['verified', 'auto'], ['manual', source(policy=old_policy)])
    assert full_resume_suras(path, model_id='quran-phonetics',
                             onset_policy=FULL_ONSET_POLICY, counts={1: 2}) == set()
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT time,status FROM timings JOIN review_meta USING(sura,ayah) '
                          'WHERE ayah=1').fetchone() == (100, 'verified')


def done(path, model='quran-phonetics'):
    return full_resume_suras(path, model_id=model, onset_policy='qpc-phrase-boundary-v3', counts={1: 2})


def test_old_full_policy_must_recalculate_auto_rows(tmp_path):
    path = project(tmp_path, ['auto', 'auto'], [source(policy='qpc-local-weak-prefix-v1')] * 2)
    assert done(path) == set()


def test_matching_full_model_and_policy_cannot_prove_execution_identity(tmp_path):
    path = project(tmp_path, ['auto', 'auto'], [source()] * 2)
    assert done(path) == set()


@pytest.mark.parametrize('provenance', [
    source() + ':revision=old:audio_signature=old-recording',
    source() + ':revision=current:audio_signature=current-recording',
    source() + ':align_lag_ms=40:lag_adjust_ms=-10',
])
def test_partial_full_provenance_never_skips_automatic_rows(tmp_path, provenance):
    path = project(tmp_path, ['verified', 'auto'], ['manual', provenance])
    before = path.read_bytes()
    # No prepared FULL execution identity compares the current audio, actual
    # weights and effective settings. Matching selection labels are insufficient.
    assert done(path) == set()
    assert full_resume_suras(path, counts={1: 2}) == set()
    assert path.read_bytes() == before


def test_changing_full_model_does_not_reuse_other_model_results(tmp_path):
    path = project(tmp_path, ['auto', 'auto'], [source()] * 2)
    assert done(path, 'muaalem') == set()


def test_fully_protected_surah_keeps_manual_values_and_does_not_need_provenance(tmp_path):
    path = project(tmp_path, ['edited', 'verified'], ['manual', None])
    assert done(path) == {1}
    assert done(path, 'muaalem') == {1}
    assert full_resume_suras(path, counts={1: 2}) == {1}


def test_mixed_surah_recalculates_stale_auto_without_changing_manual_row(tmp_path):
    path = project(tmp_path, ['verified', 'auto'], ['manual', source(policy='old')])
    assert done(path) == set()
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT status FROM review_meta WHERE ayah=1').fetchone() == ('verified',)
        assert db.execute('SELECT time FROM timings WHERE ayah=1').fetchone() == (100,)


def test_missing_or_duplicate_policy_is_not_current_identity(tmp_path):
    path = project(tmp_path, ['auto', 'auto'], [source(), source() + ':full_onset_policy=old'])
    assert done(path) == set()


def test_incomplete_or_invalid_timings_are_not_complete_results(tmp_path):
    path = project(tmp_path, ['edited', 'verified'], ['manual', 'manual'], (100, 100))
    assert done(path) == set()
