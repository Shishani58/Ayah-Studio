# Read source DB via SQLite read-only backup; serve a disposable copy and original audio.
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from surah_cutter.webgui import settings
from surah_cutter.webgui.state import AppState, discover_audio

cfg = settings.load()
root = Path("/tmp/ayah-studio-verify")
source = sqlite3.connect(f"file:{cfg['db_path']}?mode=ro", uri=True)
destination = sqlite3.connect(root / "real-copy.db")
source.backup(destination)
source.close()
destination.close()
settings.SETTINGS_PATH = root / "real-settings.json"
settings.save({"theme": "light", "mode": "full"})
from surah_cutter.webgui import server  # noqa: E402 - configure disposable settings before server import

state = AppState()
state.set_db_path(str(root / "real-copy.db"))
state.audio_dir = Path(cfg["audio_dir"])
state.audio_map = discover_audio(state.audio_dir)
server.STATE = state
import uvicorn  # noqa: E402 - configure disposable settings before server import

uvicorn.run(server.app, host="127.0.0.1", port=8879, log_level="warning")
