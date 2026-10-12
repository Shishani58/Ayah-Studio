"""Regression for the no-console Windows executable."""

import subprocess
import sys
import textwrap


def test_server_starts_without_console_streams():
    script = textwrap.dedent("""\
        import sys
        import threading
        import urllib.request
        from surah_cutter.webgui.launch import _free_port, _serve, _wait_until_up
        port = _free_port()
        sys.stdout = None
        sys.stderr = None
        threading.Thread(target=_serve, args=(port,), daemon=True).start()
        if not _wait_until_up(port):
            raise SystemExit(1)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            assert response.status == 200
            assert b"Ayah Studio" in response.read()
    """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=20)
