"""Read-only audio fixture, independent of the user's running application."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import Response
import uvicorn

app = FastAPI()


@app.get("/api/project")
def project():
    return {"fixture": "fatiha-playback-only"}


@app.get("/api/audio/1")
def audio():
    return FileResponse(Path.home() / "Desktop/Abu_Bakr_Ash-Shatri/001.mp3")


@app.get("/api/audio/1/clip")
def clip(start_ms: int, end_ms: int):
    from surah_cutter.webgui.playback import audio_clip

    return Response(audio_clip(Path.home() / "Desktop/Abu_Bakr_Ash-Shatri/001.mp3", start_ms, end_ms),
                    media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8769, log_level="warning")
