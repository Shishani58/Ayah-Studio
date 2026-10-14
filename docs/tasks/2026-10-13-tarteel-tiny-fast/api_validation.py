"""Real HTTP selection/generation on temporary settings and a protected project."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import urllib.request

import uvicorn

ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def client(app):
    runner = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8796, log_level="warning"))
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    while not runner.started:
        assert thread.is_alive()
        time.sleep(.05)

    def request(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request("http://127.0.0.1:8796" + path, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.read().decode()

    try:
        yield request
    finally:
        runner.should_exit = True
        thread.join(timeout=5)


def main():
    sys.path.insert(0, str(ROOT))
    protected = [ROOT / "data/quran_mushaf.db", ROOT / "data/references/abu_bakr_ash_shatri.db",
                 Path.home() / ".surah-cutter/settings.json"]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    with tempfile.TemporaryDirectory(prefix="tarteel-fast-api-") as directory:
        directory = Path(directory)
        # Override settings before importing server: its module creates an app.
        from surah_cutter.webgui import settings

        settings.SETTINGS_PATH = directory / "settings.json"
        settings.update(strict=True, alignment_model="quran-phonetics", mode="fast", language="ru")
        from surah_cutter.webgui import server
        from surah_cutter.webgui.state import AppState, DB_CODE, DB_VERSION
        from surah_cutter.core import pipeline
        from surah_cutter.core.model_runtime import release_runtimes
        from surah_cutter.data.project_db import ProjectDB

        # An actual FAST request must not invoke any complete/tiled alignment.
        def forbidden(*args, **kwargs):
            raise AssertionError("hidden full alignment")

        pipeline._full_align = forbidden
        pipeline._tiled_align = forbidden
        server.STATE = AppState()
        server.STATE.set_db_path(str(directory / "test.scproj"))
        audio = Path.home() / "Desktop/Abu_Bakr_Ash-Shatri/114.mp3"
        server.STATE.audio_dir = audio.parent
        server.STATE.audio_map = {114: audio}
        data = json.loads(Path(__file__).with_name("comparison.json").read_text())
        seed = next(x for x in data if x["model"] == "whisper-tarteel-tiny" and x["surah"] == 114)
        with ProjectDB.open(str(server.STATE.db_path), DB_CODE, DB_VERSION) as db:
            db.write_timings([(114, a, ms) for a, ms, _ in seed["rows"]], "seed")
            db.set_time(114, 1, seed["rows"][0][1] + 10)
            db.set_status(114, 2, "verified")
            manual_times = db.get_timings()[:2]
            manual_meta = [db.get_review(114, a) for a in (1, 2)]
        server._restore_from_settings = lambda: None
        server._JOB = None
        with client(server.create_app()) as request:
            selected = json.loads(request("/api/models/whisper-tarteel-tiny/select", {}))
            assert selected["active_id"] == "whisper-tarteel-tiny" and selected["mode"] == "fast"
            cfg = json.loads(request("/api/settings", {"mode": "fast"}))
            assert settings.load()["mode"] == "fast"
            started = time.monotonic()
            generated = json.loads(request("/api/generate", {"mode": "fast", "suras": [114]}))
            events = [json.loads(line[5:].strip()) for line in request("/api/generate/stream").splitlines()
                      if line.startswith("data:")]
            elapsed = time.monotonic() - started
            assert not any(e.get("type") == "error" for e in events), events
            done = next(e for e in events if e.get("type") == "done")
            with ProjectDB.open(str(server.STATE.db_path), DB_CODE, DB_VERSION) as db:
                assert db.get_timings()[:2] == manual_times
                assert [db.get_review(114, a) for a in (1, 2)] == manual_meta
                auto_source = db.get_review(114, 3)["source"]
                assert "local-asr-pause:whisper-tarteel-tiny" in auto_source and "fast" in auto_source
            report = {"selected": selected["active_id"], "mode": selected["mode"], "settings": cfg,
                      "generation": generated, "done": done, "seconds": elapsed,
                      "edited_verified_preserved": True, "auto_source": auto_source}
        release_runtimes()
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    assert before == after
    report["protected_sha256"] = after
    Path(__file__).with_name("api-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Tiny HTTP FAST, source and protected manual rows: passed", flush=True)


if __name__ == "__main__":
    main()
