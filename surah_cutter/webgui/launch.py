"""Desktop entry: serve the web GUI locally and open it in a native window.

Starts the FastAPI app (:mod:`surah_cutter.webgui.server`) on a free localhost
port in a background thread, then opens a pywebview window (system WebView) at
that URL. Falls back to the default browser if pywebview is unavailable or has no
display. This is the ``surah-cutter-gui`` entry point.
"""

from __future__ import annotations

import socket
import sys
import threading
import time


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve(port: int) -> None:
    import uvicorn

    from surah_cutter.webgui.server import app

    # Windows windowed executables have no stdout; automatic color detection
    # calls stdout.isatty() and prevents the local server from starting.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", use_colors=False)


def _wait_until_up(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


class _NativeApi:
    """Bridge exposed to JS as ``window.pywebview.api`` for native file dialogs.

    Lets the UI offer real "Browse…" buttons (folder picker, save-file picker)
    instead of asking the user to type filesystem paths — only available inside
    the pywebview window, not the browser fallback.
    """

    def __init__(self) -> None:
        self.window = None  # set after the window is created

    def pick_folder(self) -> str | None:
        import webview

        res = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        return res[0] if res else None

    def pick_db(self) -> str | None:
        """Pick an existing project DB to open."""
        import webview

        res = self.window.create_file_dialog(
            webview.OPEN_DIALOG, file_types=("SQLite DB (*.db)", "All files (*.*)")
        )
        return res[0] if res else None

    def save_db(self, suggested: str = "reciter.db") -> str | None:
        """Pick a destination path for a new project / export."""
        import webview

        res = self.window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=suggested
        )
        if not res:
            return None
        return res if isinstance(res, str) else res[0]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--self-check" in args:
        from surah_cutter.selfcheck import main as self_check

        return self_check()
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"

    server = threading.Thread(target=_serve, args=(port,), daemon=True)
    server.start()
    if not _wait_until_up(port):
        print("Server failed to start.")
        return 1

    # Prefer a native window (with file-dialog bridge); fall back to the browser.
    try:
        import webview

        api = _NativeApi()
        window = webview.create_window(
            "Ayah Studio", url, js_api=api, width=1180, height=840, min_size=(820, 600)
        )
        api.window = window
        webview.start()
        return 0
    except Exception as exc:  # pywebview missing or no display
        print(f"Native window unavailable ({exc}); opening in browser: {url}")
        import webbrowser

        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
