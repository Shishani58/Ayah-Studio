"""Real API probes and temporary-project regeneration safety."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

from contextlib import contextmanager
from types import SimpleNamespace
import threading
import urllib.request
import uvicorn


@contextmanager
def TestClient(app):
    runner = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8795, log_level="warning"))
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    while not runner.started:
        assert thread.is_alive()
        time.sleep(0.05)

    class Client:
        def request(self, path, method):
            request = urllib.request.Request(
                "http://127.0.0.1:8795" + path,
                data=b"{}" if method == "POST" else None,
                method=method,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                text = response.read().decode()
                return SimpleNamespace(
                    status_code=response.status, text=text, json=lambda: json.loads(text)
                )

        def post(self, path):
            return self.request(path, "POST")

        def get(self, path):
            return self.request(path, "GET")

    try:
        yield Client()
    finally:
        runner.should_exit = True
        thread.join(timeout=5)


ROOT = Path(__file__).resolve().parents[3]


def main():
    sys.path.insert(0, str(ROOT))
    from surah_cutter.webgui import server, settings
    from surah_cutter.core.model_runtime import release_runtimes
    from surah_cutter.data.project_db import ProjectDB

    reports = []
    validation = json.loads(Path(__file__).with_name("validation.json").read_text())
    protected = [
        ROOT / "data/quran_mushaf.db",
        ROOT / "data/references/abu_bakr_ash_shatri.db",
        Path.home() / ".surah-cutter/settings.json",
    ]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    with tempfile.TemporaryDirectory(prefix="tarteel-api-") as directory:
        settings.SETTINGS_PATH = Path(directory) / "settings.json"
        settings.update(strict=True, language="ru", alignment_model="quran-phonetics", mode="fast")
        server._restore_from_settings = lambda: None
        app = server.create_app()
        with TestClient(app) as client:
            for model_id in ("whisper-tarteel-tiny", "whisper-tarteel-base"):
                started = time.monotonic()
                response = client.post(f"/api/models/{model_id}/select")
                assert response.status_code == 200, response.text
                saved = settings.load()
                assert saved["alignment_model"] == model_id and saved["mode"] == "full"
                assert client.get("/api/models").json()["active_id"] == model_id
                item = next(x for x in validation if x["model"] == model_id and x["surah"] == 114)
                rows = [(114, a, ms) for a, ms, c in item["rows"]]
                path = Path(directory) / (model_id + ".scproj")
                with ProjectDB.open(str(path), "test", 1) as db:
                    db.write_timings(rows, item["source"])
                    db.set_time(114, 1, rows[0][2] + 10)
                    db.set_status(114, 2, "verified")
                    old = [db.get_review(114, a) for a in (1, 2)]
                    old_times = db.get_timings()[:2]
                    n = db.write_timings([(s, a, ms + 20) for s, a, ms in rows], item["source"])
                    assert n == 4 and db.get_timings()[:2] == old_times
                    assert [db.get_review(114, a) for a in (1, 2)] == old
                    start, end = 6000, 12000
                    word_rows = [(i, start + ms, conf) for i, ms, conf in item["first_ayah_words"]]
                    db.write_word_timings(114, 1, word_rows, start, end, item["source"])
                    db.set_word_time(114, 1, 0, word_rows[0][1] + 10, start, end)
                    db.verify_word(114, 1, 1, start, end)
                    old_words = db.get_word_timings(114, 1)[:2]
                    nwords = db.write_word_timings(
                        114,
                        1,
                        [(i, ms + 20, c) for i, ms, c in word_rows],
                        start,
                        end,
                        item["source"],
                    )
                    assert nwords == 2 and db.get_word_timings(114, 1)[:2] == old_words
                reports.append(
                    dict(
                        model=model_id,
                        http_status=response.status_code,
                        saved=saved,
                        probe_seconds=time.monotonic() - started,
                        protected_ayahs=True,
                        protected_words=True,
                        source=item["source"],
                    )
                )
        release_runtimes()
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    assert before == after
    Path(__file__).with_name("api-validation.json").write_text(
        json.dumps(dict(results=reports, protected_sha256=after), ensure_ascii=False, indent=2)
        + "\n"
    )
    print("real API probes and edited/verified ayah/word merge: passed", flush=True)


if __name__ == "__main__":
    main()
