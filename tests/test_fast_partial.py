"""Partial FAST is a model result, never synthetic timing data."""
import json
from pathlib import Path

import pytest

from surah_cutter.data.alignment_result import result_source, write_alignment_result
from surah_cutter.data.alignment_runs import load_run
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.data.export_db import ExportError, export_db
from test_fast_executor import Fake, Policy, observation, pause, run


def partial_trace(rows=()):
    resolved = {a: t for _, a, t in rows}
    return dict(mode='fast', result_contract='partial-v1', surah=112, ayah_count=4,
        generated_rows=list(rows), unresolved_ayahs=[a for a in range(1, 5) if a not in resolved],
        duration_ms=40000, ayahs=[dict(ayah=a, onset_ms=resolved.get(a),
               conf=.9 if a in resolved else 0, reliable=a in resolved) for a in range(1, 5)])


@pytest.mark.parametrize('requested', ['fast', 'words'])
def test_empty_fast_supersedes_old_fast_but_protects_full_and_humans(tmp_path, requested):
    path = tmp_path / 'p.db'
    with ProjectDB.open(str(path), 'test', 1) as db:
        db.write_timings([(112, 1, 1000), (112, 2, 2000)], 'old:actual_mode=fast', confidence=0)
        db.write_timings([(112, 3, 3000)], 'model:actual_mode=full', confidence=.7)
        db.set_time(112, 4, 4000)
        before = [db.get_review(112, a) for a in (3, 4)]
        result = write_alignment_result(db, [], trace=partial_trace(), requested_mode=requested, db_path=path)
        assert db.get_timings() == [(112, 3, 3000), (112, 4, 4000)]
        assert [db.get_review(112, a) for a in (3, 4)] == before
        assert result.written == 0 and result.unresolved_ayahs == (1, 2, 3, 4)
        assert result.persisted_rows == ((112, 3, 3000), (112, 4, 4000))
        payload = json.loads(Path(result.log_paths['json']).read_text())
        assert payload['model_rows'] == [] and len(payload['rows']) == 2
        if requested == 'fast':
            history = load_run(Path(result.log_paths['json']).parent, 112, Path(result.log_paths['run']).stem)
            assert history['ayahs'] == [] and history['unresolved_ayahs'] == [1, 2, 3, 4]
            latest = load_run(Path(result.log_paths['json']).parent, 112, 'legacy-fast')
            assert [a['ayah'] for a in latest['ayahs']] == [3, 4]


def test_partial_replacement_rolls_back_deletion_and_keeps_new_manual_status(tmp_path):
    with ProjectDB.open(str(tmp_path / 'p.db'), 'test', 1) as db:
        db.write_timings([(112, 1, 1000), (112, 2, 2000)], 'old:fast')
        db.set_time(112, 3, 3000)
        before = db.get_timings()
        with pytest.raises(ValueError, match='порядку'):
            write_alignment_result(db, [(112, 2, 3500)], trace=partial_trace([(112, 2, 3500)]),
                                   requested_mode='fast')
        assert db.get_timings() == before


def test_partial_contract_rejects_unproven_numeric_rows(tmp_path):
    trace = partial_trace([(112, 1, 1000)])
    trace['ayahs'][0]['conf'] = 0
    with ProjectDB.open(str(tmp_path / 'p.db'), 'test', 1) as db:
        with pytest.raises(ValueError, match='resolved boundary'):
            write_alignment_result(db, [(112, 1, 1000)], trace=trace, requested_mode='fast')
        assert db.get_timings() == []


def test_legacy_estimates_cannot_escape_through_export(tmp_path):
    path = tmp_path / 'p.db'
    with ProjectDB.open(str(path), 'test', 1) as db:
        db.write_timings([(112, 1, 1000)], 'old:actual_mode=fast', confidence=0)
    with pytest.raises(ExportError, match='unresolved legacy FAST'):
        export_db(str(path), str(tmp_path / 'out.db'), strict=False)
    assert not (tmp_path / 'out.db').exists()


def test_implicit_words_fast_origin_is_unambiguous():
    trace = partial_trace()
    del trace['mode']
    assert ':actual_mode=fast:' in result_source(trace, 'words')


def test_geometry_failure_can_locate_identity_without_publishing_its_time(monkeypatch):
    import surah_cutter.core.fast_executor as executor
    original = executor.pause_geometry
    def geometry(pcm, sr, p):
        data = original(pcm, sr, p)
        return {**data, 'accepted': p.start_ms != 6000, 'reason': 'uncertain_cut'}
    monkeypatch.setattr(executor, 'pause_geometry', geometry)
    fake = Fake(lambda f, s, d: observation(f.phrase({1000: 1, 6000: 2, 12000: 3}.get(s)))
                if s in (1000, 6000, 12000) else observation(()))
    rows, trace = run(fake, [pause(1000), pause(6000), pause(12000)])
    assert [a for a, _, _ in rows] == [1, 3]
    assert trace['boundaries'][1]['start_ms'] is None
    assert [a['ayah'] for a in trace['search_anchors']] == [1, 2, 3]


def test_independent_next_target_reenters_after_missing_owner():
    fake = Fake(lambda f, s, d: observation(f.phrase({1000: 1, 6000: 2, 12000: 4}.get(s)))
                if s in (1000, 6000, 12000) else observation(()))
    fake.timing = False
    rows, trace = run(fake, [pause(1000), pause(6000), pause(12000)],
                      policy=Policy(sparse_anchors=False))
    assert 3 not in [a for a, _, _ in rows]
    assert 4 in [a for a, _, _ in rows]
    assert any(e['state'] == 'REENTER_SEARCH' for e in trace['transitions'])


def test_landmark_wide_pause_is_identity_only_and_provenance_is_validated():
    from dataclasses import replace
    from surah_cutter.core.fast_executor import (
        Boundary, _review_addressed_recovery_onsets, validate_boundaries,
    )
    anchor = Boundary(4, 1000, 'pause', .99, read_id=0,
        proof={'pause_id': 0, 'independent_region_origin_ayah': 1,
               'independent_region_attempt': 1, 'phrase': ('unique',), 'complete': True,
               'ambiguous': False, 'competitors_checked': True, 'offset': 0,
               'pause_ms': [0, 2000]})
    reviewed, reasons = _review_addressed_recovery_onsets([anchor],
        [{'quiet_stationarity': True, 'midpoint_interval_radius_ms': 1000}], Policy())
    assert reviewed[0].start_ms is None and reviewed[0].confidence == 0
    assert reasons[0]['identity_proof'] == anchor.proof
    read = dict(outcome='completed', window_ms=[1800, 3300],
                reason='ayah=1:independent_region_identity', observed=dict(
                    units=('unique',), unit_quality=(.99,), truncated=False))
    validate_boundaries([anchor], [4], 10000, [read], threshold=.6, search_only=True)
    for proof in [{**anchor.proof, 'independent_region_origin_ayah': 4},
                  {**anchor.proof, 'independent_region_attempt': 2}]:
        with pytest.raises(ValueError, match='independent_region_provenance'):
            validate_boundaries([replace(anchor, proof=proof)], [4], 10000, [read], threshold=.6)
    with pytest.raises(ValueError, match='independent_region_provenance'):
        validate_boundaries([anchor], [4], 10000, [{**read, 'reason': 'unrelated'}], threshold=.6)


def test_disconnected_search_does_not_publish_an_unstable_narrow_midpoint():
    from surah_cutter.core.fast_executor import Boundary, _review_addressed_recovery_onsets
    anchor = Boundary(3, 1000, 'pause', .99, read_id=0,
                      proof={'pause_id': 0, 'independent_region_origin_ayah': 1})
    following = Boundary(4, 2000, 'pause', .99, read_id=1, proof={'pause_id': 1})
    stable = Boundary(5, 3000, 'pause', .99, read_id=2, proof={'pause_id': 2})
    geometry = [{'quiet_stationarity': s, 'midpoint_interval_radius_ms': 400}
                for s in (True, False, True)]
    rows, reviews = _review_addressed_recovery_onsets([anchor, following, stable], geometry, Policy())
    assert rows[0] == anchor and rows[2] == stable
    assert rows[1].start_ms is None
    assert reviews[0]['reason'] == 'reacquired_nonstationary_midpoint_unproved'


def test_first_landmark_denied_by_budget_returns_partial_instead_of_crashing(monkeypatch):
    from surah_cutter.core.fast_reads import LocalEvidenceReads
    original = LocalEvidenceReads.read
    def read(self, start, end, *, reason):
        if reason.endswith(':independent_region_identity'):
            self.audio_ms = self.total_ms * 2  # fault-inject an exhausted guard
        return original(self, start, end, reason=reason)
    monkeypatch.setattr(LocalEvidenceReads, 'read', read)
    fake = Fake(lambda *args: observation(()))
    rows, trace = run(fake, [pause(t) for t in range(1000, 55000, 3000)], duration=60000,
                      policy=Policy(sparse_anchors=True))
    assert rows == [] and trace['result_complete'] is False
    assert any(e['state'] == 'REACQUIRE_REGION' for e in trace['transitions'])
    assert any(e['reason'] == 'inference_budget' for e in trace['denied_acoustic_reads'])
    assert trace['transitions'][-1]['state'] == 'RESULT_READY'


def test_legacy_estimates_are_hidden_from_review_and_words_without_mutation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from surah_cutter.data.project_state import complete_suras
    from surah_cutter.webgui import server
    path = tmp_path / 'p.db'
    with ProjectDB.open(str(path), 'test', 1) as db:
        db.write_timings([(112, a, a*1000) for a in range(1, 5)], 'old:actual_mode=fast', confidence=0)
        db.set_status(112, 2, 'verified')
    before = path.read_bytes()
    monkeypatch.setattr(server, 'STATE', SimpleNamespace(db_path=path))
    times, _ = server._sura_timings_reviews(112)
    assert times == {2: 2000}
    assert complete_suras(path) == set()
    assert path.read_bytes() == before


def test_json_downstream_proof_retains_unknown_landmark_root_and_rejects_unstable_time(monkeypatch):
    from surah_cutter.core import fast_executor as executor
    from surah_cutter.data import quran_text
    fake = Fake(lambda *args: observation(()), words=quran_text.surah_words(114))
    fake.timing = False
    fake.responder = lambda f, start, duration: observation(f.phrase(3 if start == 31800 else 4), positions=False) \
        if start in (31800, 40000) else observation((), positions=False)
    def geometry(pcm, sr, p):
        return dict(accepted=True, reason='measured_dsp_interval', quiet_stationarity=False,
                    midpoint_interval_radius_ms=(p.end_ms-p.start_ms)/2)
    monkeypatch.setattr(executor, 'pause_geometry', geometry)
    pauses = [pause(1000), pause(30000, 2000), pause(40000)]
    _, trace = run(fake, pauses, duration=60000, policy=Policy(max_candidates=1))
    assert trace['boundaries'][2]['start_ms'] is None  # wide landmark remains unknown
    assert trace['boundaries'][3]['start_ms'] is None  # narrow unstable successor too
    raw = json.loads(json.dumps(trace['search_anchors']))
    downstream = next(b for b in raw if b['ayah'] == 4)
    assert downstream['proof']['search_region_root']['ayah'] == 3
    with pytest.raises(ValueError, match='unproved_reacquired_midpoint'):
        executor.validate_boundaries([executor.Boundary(**downstream)], [4], 60000,
            trace['acoustic_reads'], threshold=.6, pauses=[p.pause for p in pauses],
            pause_geometry_data=trace['pause_geometry'])


@pytest.mark.parametrize('resolved', [[], [(112, 2, 2000)]])
def test_benchmark_validators_accept_explicit_partial_and_reject_fabrication(resolved):
    from dataclasses import asdict
    from surah_cutter.core.fast_executor import Boundary
    from tools.bench_fast import sequence_contract_violations
    from tools.bench_fast_models import validate
    trace = partial_trace(resolved)
    starts = {a:t for _,a,t in resolved}
    trace.update(total_ms=40000, acoustic_reads=[], covered_audio_ms=0, audio_ms=0,
        boundaries=[asdict(Boundary(a, starts[a], 'pause', .9, read_id=0, proof={})) if a in starts
                    else asdict(Boundary(a, None, 'unresolved', 0, 'no_evidence')) for a in range(1,5)])
    rows = [(a,t,.9) for _,a,t in resolved]
    assert validate(rows, trace, [1,2,3,4]) == []
    assert sequence_contract_violations(rows, [1,2,3,4])  # legacy stays strict
    trace['boundaries'][0]['start_ms'] = 1234
    assert sequence_contract_violations(rows, [1,2,3,4], trace)
    with pytest.raises(AssertionError):
        validate(rows, trace, [1,2,3,4])
