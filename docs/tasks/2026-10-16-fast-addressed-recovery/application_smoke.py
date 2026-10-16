"""Real /api/generate → pipeline → FAST → persistence in a disposable project."""
import json
import hashlib
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from surah_cutter.core import align, decode, fast_executor, model_store
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.webgui import server
from surah_cutter.webgui.state import AppState, DB_CODE, DB_VERSION
from tools.bench_fast_models import dispatch_spy, validate, verify_dispatch

P = Path(__file__).resolve().parent
REPO = P.parents[2]
assert Path(fast_executor.__file__).resolve() == REPO / 'surah_cutter/core/fast_executor.py'
assert fast_executor.VERSION == 'sequential-fast-v3-sparse-14-next-dsp-recovery'
adoption = json.loads((P / 'production-adoption.json').read_text())
assert hashlib.sha256(Path(fast_executor.__file__).read_bytes()).hexdigest() == adoption['after_sha256']
baseline = next(c for c in json.loads((P / 'current-baseline-smoke.json').read_text()) if c['surah'] == 80)
expected = next(c for c in json.loads((P / 'PF-SF-development-replay.json').read_text()) if c['surah'] == 80)
folder = Path(tempfile.mkdtemp(prefix='fast-application-smoke-'))
database = folder / 'project.db'
with ProjectDB.open(str(database), DB_CODE, DB_VERSION) as db:
    db.write_timings([(80, a, ms) for a, ms, _ in baseline['rows']], 'old-baseline')
    db.set_status(80, 1, 'edited')
    db.set_status(80, 3, 'verified')
    protected = {a: {'time': next(t for s, n, t in db.get_timings() if s == 80 and n == a),
                     'review': db.get_review(80, a)} for a in (1, 3)}
state = AppState()
state.set_db_path(str(database))
state.audio_map = {80: Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/080.mp3')}
model_store.ACTIVE_ID = 'whisper-tarteel-base'
align._ONNX_CACHE.clear()
pcm = decode.decode(str(state.audio_map[80]))
bundle = align.default_bundle()
events = []
endpoint = next(r.endpoint for r in server.app.routes if r.path == '/api/generate')
with dispatch_spy(bundle, pcm) as dispatched, \
        patch.object(server, 'STATE', state), patch.object(server, '_JOB', None), \
        patch.object(server.settings, 'update', lambda **kwargs: kwargs), \
        patch.object(server._GenJob, 'start', server._GenJob._run), \
        patch.object(server._GenJob, '_emit', lambda job, event: events.append(event)):
    response = endpoint({'suras': [80], 'mode': 'fast', 'overwrite_manual': False,
                         'db_path': str(database)})
assert events[-1]['type'] == 'done', events[-1]
with ProjectDB.open(str(database), DB_CODE, DB_VERSION) as db:
    rows = db.get_timings()
    assert len(rows) == 42
    for a, before in protected.items():
        assert db.get_review(80, a) == before['review']
        assert next(t for s, n, t in rows if s == 80 and n == a) == before['time']
    wanted = {a: ms for a, ms, _ in expected['rows']}
    assert all(t == wanted[a] for s, a, t in rows if a not in protected)
(P / 'application-smoke-precheck.json').write_text(json.dumps({'response': response,
    'events': events, 'database': str(database), 'protected': protected, 'rows': rows,
    'actual_dispatches': dispatched, 'protected_full_review_and_time_unchanged': True},
    ensure_ascii=False, indent=2) + '\n')
trace_path = database.with_name('project.logs') / 'sura_080_fast.json'
diagnostic = json.loads(trace_path.read_text())
trace = diagnostic['trace']
assert diagnostic['mode'] == 'fast' and diagnostic['overwrite_manual'] is False
assert trace['fast_version'] == fast_executor.VERSION
assert trace['fast_run_key'] != baseline['trace']['fast_run_key']
assert trace['run_snapshot']['model'] == 'whisper-tarteel-base'
assert trace['bounded_ctc_fallback'] is None
assert trace['acoustic_reads']
assert trace['boundaries'][36]['evidence_kind'] == 'estimate'
verify_dispatch(dispatched, pcm, trace)
assert not validate(expected['rows'], trace, list(range(1, 43)))
assert dispatched == expected['actual_dispatches']
shutil.copyfile(trace_path, P / 'application-smoke-trace.json')
(P / 'application-smoke.json').write_text(json.dumps({'response': response,
    'events': events, 'database': str(database), 'temporary_project_only': True,
    'protected': protected, 'protected_full_review_and_time_unchanged': True,
    'auto_rows_equal_measured_PF': True, 'source_version': fast_executor.VERSION,
    'actual_dispatches_equal_frozen_PF': True, 'actual_dispatches': dispatched,
    'old_resume_key_invalidated': True, 'rows': rows}, ensure_ascii=False, indent=2) + '\n')
release_runtimes()
print('real API/pipeline/save smoke:42 rows, edited/verified intact, measured PF auto times', flush=True)
