import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

from surah_cutter.webgui import updates


def test_update_ui_flow():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to validate update UI")
    subprocess.run([node, str(Path(__file__).with_name("app_updates.cjs"))],
                   check=True, timeout=10)


@pytest.mark.parametrize("value", ["v1.2.3-rc1", "main", "1.2", "1.2.3.4"])
def test_only_stable_versions(value):
    with pytest.raises(ValueError):
        updates._version(value)


def test_version_order_and_source_not_replaceable():
    assert updates._version("v0.10.0") > updates._version("0.9.9")
    if not getattr(sys, "frozen", False):
        assert updates._target() is None
        updater = updates.AppUpdater()
        assert updater.check()["phase"] == "unsupported"
        with pytest.raises(RuntimeError):
            updater.install()


@pytest.mark.parametrize("name", ["../evil", "Ayah Studio/../evil", "Ayah Studio/C:/evil",
                                 "Ayah Studio/../../evil", "/Ayah Studio/evil"])
def test_zip_traversal_rejected(tmp_path, name):
    archive = tmp_path / "update.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, "evil")
    with pytest.raises(ValueError):
        updates._extract_windows(archive, tmp_path / "stage")
    assert not (tmp_path / "evil").exists()


def release_fixture(monkeypatch, tmp_path, *, corrupt=False, older=False):
    target = tmp_path / "Installed app"
    target.mkdir()
    (target / "keep").write_text("original")
    monkeypatch.setattr(updates, "_target", lambda: target)
    monkeypatch.setattr(updates.sys, "platform", "win32")
    monkeypatch.setattr(updates, "__version__", "1.2.3")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Ayah Studio/Ayah Studio.exe", b"new binary")
    binary = archive.getvalue()
    name = "Ayah-Studio-1.3.0-Windows-x64.zip"
    metadata = {"tag_name": "v1.2.2" if older else "v1.3.0", "assets": [
        {"name": name, "size": len(binary), "browser_download_url": "binary"},
        {"name": name + ".sha256", "browser_download_url": "checksum"},
    ]}
    digest = "0" * 64 if corrupt else hashlib.sha256(binary).hexdigest()
    payloads = {updates.LATEST: json.dumps(metadata).encode(), "binary": binary,
                "checksum": f"{digest}  {name}\n".encode()}
    monkeypatch.setattr(updates, "_fetch", lambda url: io.BytesIO(payloads[url]))
    return updates.AppUpdater(), target


def test_download_verifies_and_stages_without_touching_app(monkeypatch, tmp_path):
    updater, target = release_fixture(monkeypatch, tmp_path)
    updater._download()
    assert updater.status()["phase"] == "ready"
    assert (updater.staged / "Ayah Studio.exe").read_bytes() == b"new binary"
    assert (target / "keep").read_text() == "original"
    assert not list(updater.work.glob("*.zip"))
    assert updater.check()["phase"] == "ready"  # no duplicate download
    work = updater.work
    updater.cleanup()
    assert not work.exists()


def test_checksum_failure_cleans_stage_and_keeps_app(monkeypatch, tmp_path):
    updater, target = release_fixture(monkeypatch, tmp_path, corrupt=True)
    updater._download()
    assert updater.status()["phase"] == "error"
    assert "checksum mismatch" in updater.status()["error"]
    assert updater.staged is None
    assert not list(tmp_path.glob(".ayah-update-*"))
    assert (target / "keep").read_text() == "original"


def test_no_downgrade(monkeypatch, tmp_path):
    updater, _ = release_fixture(monkeypatch, tmp_path, older=True)
    updater._download()
    assert updater.status()["phase"] == "current"
    assert updater.work is None


def test_untrusted_release_url_rejected():
    with pytest.raises(ValueError, match="Unexpected release URL"):
        updates._fetch("https://example.com/update.zip")


@pytest.mark.skipif(os.name == "nt", reason="POSIX helper integration")
@pytest.mark.parametrize("failure", [None, "missing_stage", "launch_failed"])
def test_helper_waits_for_exit_and_replaces_or_rolls_back(tmp_path, failure):
    work = tmp_path / "update work"
    work.mkdir()
    target = tmp_path / "Ayah Studio.app"
    target.mkdir()
    (target / "version").write_text("old")
    stage = work / "Ayah Studio.app"
    if failure != "missing_stage":
        stage.mkdir()
        (stage / "version").write_text("new")
    script = work / "install.sh"
    script.write_text(updates._MAC_HELPER.replace(
        "/usr/bin/open", "false" if failure == "launch_failed" else "true"
    ))
    app = subprocess.Popen(["/bin/sleep", "20"])
    helper = subprocess.Popen(["/bin/sh", str(script), str(work), str(app.pid),
                               str(target), str(stage)])
    try:
        deadline = time.monotonic() + 5
        while not (work / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (work / "ready").exists()
        assert (target / "version").read_text() == "old"
        assert not (work / "previous").exists()
        app.terminate()
        app.wait(timeout=5)
        assert helper.wait(timeout=5) == (1 if failure else 0)
        assert (target / "version").read_text() == ("old" if failure else "new")
        assert (work / "installed").exists() == (failure is None)
        if not failure:
            assert (work / "previous/version").read_text() == "old"
    finally:
        if app.poll() is None:
            app.kill()
            app.wait()
        if helper.poll() is None:
            helper.kill()
            helper.wait()


def test_bridge_refuses_install_during_model_work(monkeypatch):
    from surah_cutter.core import model_store
    from surah_cutter.webgui.launch import _NativeApi

    api = _NativeApi()
    monkeypatch.setattr(api._updater, "install", lambda: pytest.fail("must not install"))
    ready, release = threading.Event(), threading.Event()

    def busy():
        with model_store.RUNTIME_LOCK:
            ready.set()
            release.wait(5)

    thread = threading.Thread(target=busy)
    thread.start()
    assert ready.wait(5)
    try:
        assert "error" in api.install_update()
    finally:
        release.set()
        thread.join(5)
