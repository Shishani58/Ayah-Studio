import json
from pathlib import Path

import pytest

from surah_cutter.data.alignment_result import write_alignment_result
from surah_cutter.data.alignment_runs import load_run
from surah_cutter.data.onset_evidence import onset_evidence, saved_onset_evidence
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.webgui import server
from surah_cutter.webgui.state import AppState, DB_CODE, DB_VERSION


def check(**changes):
    return dict(ayah=1, raw_onset_ms=1000, onset_ms=1000, conf=.01,
                first_token_conf=.99, reason='weak-word', **changes)


def test_weak_word_body_does_not_hide_supported_first_sound():
    assert onset_evidence(check(), 1000, .01) == {
        'first_sound': .99, 'word': .01, 'boundary_warning': False,
        'boundary': {'level': 'unknown', 'reason': 'insufficient-timing-evidence'},
    }
    c = check()
    c.update(conf=0, first_token_conf=.0002, reason='weak-onset')
    assert onset_evidence(c, 1000, 0)['first_sound'] == .0002


@pytest.mark.parametrize('score', [None, float('nan'), float('inf'), -.1, 1.1, '0.99', True])
def test_invalid_acoustic_score_is_unknown(score):
    c = check()
    c['first_token_conf'] = score
    assert onset_evidence(c, 1000, .01) is None


def test_moved_or_unmatched_mark_cannot_inherit_old_sound():
    c = check()
    assert onset_evidence(c, 1001, .01) is None
    assert onset_evidence(c, 1000, .02) is None
    c.update(onset_ms=1100, repaired=True, reason='weak-prefix-across-pause')
    assert onset_evidence(c, 1100, .01)['first_sound'] is None
    assert onset_evidence(c, 1100, .01)['boundary']['level'] == 'medium'
    c['reason'] = 'split-word-local-recheck'
    assert onset_evidence(c, 1100, .01)['first_sound'] == .99
    c['reason'] = 'late-onset-in-pause'
    c['raw_onset_ms'] = 1100
    assert onset_evidence(c, 1100, .01)['boundary_warning']


def test_api_and_snapshot_use_same_evidence_and_preserve_database(tmp_path, monkeypatch):
    state = AppState()
    state.db_path = tmp_path / 'p.db'
    audio = tmp_path / '114.mp3'
    audio.write_bytes(b'test recording')
    state.audio_map = {114: audio}
    monkeypatch.setattr(server, 'STATE', state)
    stat = audio.stat()
    signature = dict(name=audio.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    trace = dict(mode='full', model_source='test', audio_signature=signature, ayahs=[check()])
    with ProjectDB.open(str(state.db_path), DB_CODE, DB_VERSION) as db:
        saved = write_alignment_result(db, [(114, 1, 1000)], trace=trace,
                                       requested_mode='full', db_path=state.db_path)
    before = state.db_path.read_bytes()
    route = next(r.endpoint for r in server.app.routes if r.path == '/api/sura/{surah}')
    item = route(114)['ayahs'][0]
    snapshot = load_run(tmp_path / 'p.logs', 114, Path(saved.log_paths['run']).stem)
    assert item['confidence'] == .01
    assert item['onset_evidence']['first_sound'] == .99
    assert item['onset_evidence'] == snapshot['ayahs'][0]['onset_evidence']
    assert state.db_path.read_bytes() == before
    with ProjectDB.open(str(state.db_path), DB_CODE, DB_VERSION) as db:
        db.set_time(114, 1, 1000)  # manual, even at the same time
    assert route(114)['ayahs'][0]['onset_evidence'] is None
    assert route(114)['ayahs'][0]['confidence'] is None


def test_stale_audio_source_time_and_conflicting_history_are_unknown(tmp_path):
    path = tmp_path / 'p.db'
    folder = tmp_path / 'p.logs'
    folder.mkdir()
    signature = dict(name='002.mp3', size=10, mtime_ns=123)
    stored = dict(ayah=1, source='model', time=1000, confidence=.01)
    rv = {1: dict(source='model', status='auto', confidence=.01)}
    payload = dict(surah=2, trace=dict(audio_signature=signature, ayahs=[check()]),
                   persisted_review=[stored])
    latest = folder / 'sura_002_full.json'
    latest.write_text(json.dumps(payload))
    def read(**kw):
        return saved_onset_evidence(path, 2, kw.get('timings', {1: 1000}),
                                    kw.get('reviews', rv), kw.get('signature', signature))
    assert read()[1]['first_sound'] == .99
    assert not read(signature={**signature, 'mtime_ns': 124})
    assert not read(timings={1: 1001})
    assert not read(reviews={1: {**rv[1], 'source': 'different'}})
    assert not read(reviews={1: {**rv[1], 'status': 'edited'}})
    history = folder / 'runs'
    history.mkdir()
    payload['trace']['ayahs'][0]['first_token_conf'] = .2
    (history / 'sura_002_full_conflict.json').write_text(json.dumps(payload))
    assert not read()
    latest.write_text('broken')
    assert read()[1]['first_sound'] == .2
    latest.unlink()
    latest.symlink_to(history / 'sura_002_full_conflict.json')
    (history / 'sura_002_full_conflict.json').unlink()
    assert not read()


def test_boundary_confirmation_requires_local_phrase_proof_at_current_time():
    c = check()
    c.update(conf=.99, first_token_conf=.99, reliable=True, reason='ctc')
    # Strong text/sound probabilities alone do not establish timing accuracy.
    assert onset_evidence(c, 1000, .99)['boundary'] == {
        'level': 'medium', 'reason': 'alignment-only',
    }
    c.update(raw_onset_ms=800, repaired=True, reason='split-word-local-recheck')
    assert onset_evidence(c, 1000, .99)['boundary']['level'] == 'unknown'
    record = dict(ayah=1, accepted=True, outcome='completed', after_ms=1000,
                  reason='split-word-local-recheck')
    trace = dict(full_boundary_reads=[record])
    assert onset_evidence(c, 1000, .99, trace=trace)['boundary'] == {
        'level': 'high', 'reason': 'local-phrase-and-pause',
    }
    for field, value in [('ayah', 2), ('after_ms', 999), ('accepted', False),
                         ('outcome', 'failed'), ('reason', 'other')]:
        assert onset_evidence(c, 1000, .99, trace={
            'full_boundary_reads': [{**record, field: value}],
        })['boundary']['level'] == 'unknown'


def test_pause_edge_support_is_preliminary_and_does_not_override_weak_onset():
    c = check()
    trace = dict(full_windows=dict(pause_decisions=[dict(accepted=True, end_ms=1100)]))
    assert onset_evidence(c, 1000, .01, trace=trace)['boundary'] == {
        'level': 'medium', 'reason': 'pause-edge-only',
    }
    c.update(first_token_conf=.0002, conf=0, reason='weak-onset')
    assert onset_evidence(c, 1000, 0, trace=trace)['boundary']['level'] == 'low'
    c.update(first_token_conf=.99, conf=.99, reason='late-onset-in-pause')
    assert onset_evidence(c, 1000, .99, trace=trace)['boundary']['level'] == 'low'
