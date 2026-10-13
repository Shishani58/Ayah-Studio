# Disposable synthetic audio/metadata; canonical Quran text comes from the real API.
import sys
import math
import wave
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from surah_cutter.webgui import settings

root = Path("/tmp/ayah-studio-verify")
settings.SETTINGS_PATH = root / "settings.json"
settings.save({"theme": "light", "mode": "full"})
from surah_cutter.webgui import server  # noqa: E402 - configure disposable settings before server import
from surah_cutter.webgui.state import AppState, DB_CODE, DB_VERSION  # noqa: E402 - configure disposable settings before server import
from surah_cutter.data.project_db import ProjectDB  # noqa: E402 - configure disposable settings before server import
import sqlite3  # noqa: E402 - configure disposable settings before server import

state = AppState()
state.set_db_path(str(root / "fixture.db"))
with ProjectDB.open(str(state.db_path), DB_CODE, DB_VERSION):
    pass
with sqlite3.connect(state.db_path) as c:
    c.execute("delete from timings")
    c.execute("delete from review_meta")
    for surah, count, status in [
        (105, 5, "verified"),
        (106, 4, "verified"),
        (107, 7, "auto"),
        (108, 3, "verified"),
        (110, 3, "verified"),
        (111, 5, "auto"),
        (112, 4, "verified"),
        (113, 5, "auto"),
        (114, 3, "auto"),
    ]:
        for a in range(1, count + 1):
            c.execute("insert into timings values (?,?,?)", (surah, a, (a - 1) * 2000))
            c.execute(
                "insert into review_meta(sura,ayah,status) values(?,?,?)",
                (surah, a, "verified" if surah == 114 and a == 1 else status),
            )
    c.execute("delete from word_timings")
state.audio_dir = root
for surah in range(105, 115):
    path = root / f"{surah:03}.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(
            b"".join(
                struct.pack(
                    "<h",
                    int(
                        13000 * math.sin(i * 0.07) * max(0, math.sin(i / 16000 * math.pi / 2)) ** 2
                    ),
                )
                for i in range(224000)
            )
        )
state.audio_map = {s: root / f"{s:03}.wav" for s in range(105, 115)}
server.STATE = state
import uvicorn  # noqa: E402 - configure disposable settings before server import

uvicorn.run(server.app, host="127.0.0.1", port=8878, log_level="warning")
