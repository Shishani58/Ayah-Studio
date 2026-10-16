"""Opt-in desktop updates. Never replace the application in the running process."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from surah_cutter import __version__

REPOSITORY = "Shishani58/Ayah-Studio"
LATEST = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"


def _version(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError("Unsupported release version")
    return tuple(map(int, match.groups()))


def _target() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    executable = Path(sys.executable).resolve()
    if sys.platform == "darwin":
        return next((p for p in executable.parents if p.suffix == ".app"), None)
    if sys.platform == "win32" and platform.machine().lower() in ("amd64", "x86_64"):
        return executable.parent
    return None


def _fetch(url: str):
    if url != LATEST and not url.startswith(
        f"https://github.com/{REPOSITORY}/releases/download/"
    ):
        raise ValueError("Unexpected release URL")
    request = urllib.request.Request(url, headers={"User-Agent": "Ayah-Studio-Updater"})
    response = urllib.request.urlopen(request, timeout=30)
    if not response.url.startswith("https://"):
        response.close()
        raise ValueError("Insecure release redirect")
    return response


def _extract_windows(archive: Path, folder: Path) -> Path:
    with zipfile.ZipFile(archive) as bundle:
        total = 0
        for member in bundle.infolist():
            path = member.filename.replace("\\", "/")
            if (not path.startswith("Ayah Studio/") or ".." in path.split("/")
                    or ":" in path or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("Unsafe update archive")
            total += member.file_size
            if total > 8 * 1024**3:
                raise ValueError("Update archive is too large")
        bundle.extractall(folder)
    result = folder / "Ayah Studio"
    if not (result / "Ayah Studio.exe").is_file():
        raise ValueError("Missing application executable")
    return result


def _extract_macos(archive: Path, folder: Path, version: str) -> Path:
    mount = folder / "mount"
    mount.mkdir()
    subprocess.run(["/usr/bin/hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint",
                    str(mount), str(archive)], check=True, capture_output=True)
    result = folder / "Ayah Studio.app"
    try:
        source = mount / result.name
        with (source / "Contents/Info.plist").open("rb") as file:
            info = plistlib.load(file)
        if (info.get("CFBundleIdentifier") != "org.quranacademy.ayahstudio"
                or _version(info.get("CFBundleShortVersionString", "")) != _version(version)):
            raise ValueError("Unexpected application bundle")
        subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(source)],
                       check=True, capture_output=True)
        subprocess.run(["/usr/bin/ditto", str(source), str(result)],
                       check=True, capture_output=True)
    finally:
        subprocess.run(["/usr/bin/hdiutil", "detach", str(mount)],
                       check=True, capture_output=True)
    return result


# Helpers live beside (never inside) the installed app. Arguments/config contain
# paths as data, not interpolated shell code. Keep the backup until next update.
_MAC_HELPER = '''#!/bin/sh
cd "$1" || exit 1
exec >>install.log 2>&1
touch ready
i=0
while kill -0 "$2" 2>/dev/null; do
  i=$((i+1)); [ "$i" -lt 120 ] || exit 1
  sleep 1
done
backup="$1/previous"
[ ! -e "$backup" ] || exit 1
if ! /bin/mv "$3" "$backup"; then
  /usr/bin/open "$3"
  exit 1
fi
if ! /bin/mv "$4" "$3"; then
  /bin/mv "$backup" "$3"
  /usr/bin/open "$3"
  exit 1
fi
if ! /usr/bin/open "$3"; then
  /bin/mv "$3" "$4"
  /bin/mv "$backup" "$3"
  /usr/bin/open "$3"
  exit 1
fi
touch installed
'''

_WINDOWS_HELPER = '''$ErrorActionPreference = 'Stop'
$c = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'update.json') -Raw | ConvertFrom-Json
$backup = Join-Path $PSScriptRoot 'previous'
$exited = $false
try {
    Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ready') -Value 'ready'
    $process = Get-Process -Id $c.pid -ErrorAction SilentlyContinue
    if ($process -and -not $process.WaitForExit(120000)) { throw 'App did not exit' }
    $exited = $true
    if (Test-Path -LiteralPath $backup) { throw 'Backup already exists' }
    Move-Item -LiteralPath $c.target -Destination $backup
    try {
        Move-Item -LiteralPath $c.source -Destination $c.target
        Start-Process -FilePath (Join-Path $c.target 'Ayah Studio.exe')
    } catch {
        if (Test-Path -LiteralPath $c.target) {
            Move-Item -LiteralPath $c.target -Destination $c.source
        }
        Move-Item -LiteralPath $backup -Destination $c.target
        throw
    }
    Set-Content -LiteralPath (Join-Path $PSScriptRoot 'installed') -Value 'installed'
} catch {
    $_ | Out-File -LiteralPath (Join-Path $PSScriptRoot 'install.log')
    if ($exited -and (Test-Path -LiteralPath $c.target)) {
        Start-Process -FilePath (Join-Path $c.target 'Ayah Studio.exe')
    }
    exit 1
}
'''


class AppUpdater:
    def __init__(self):
        self.target = _target()
        self.work: Path | None = None
        self.staged: Path | None = None
        self.lock = threading.Lock()
        self.state = {"phase": "idle" if self.target else "unsupported",
                      "current": __version__, "version": "", "percent": 0, "error": ""}

    def status(self):
        with self.lock:
            return dict(self.state)

    def _set(self, **fields):
        with self.lock:
            self.state.update(fields)

    def check(self):
        with self.lock:
            if not self.target or self.state["phase"] in (
                "checking", "downloading", "preparing", "ready", "installing"
            ):
                return dict(self.state)
            self.state.update(phase="checking", error="", percent=0)
        threading.Thread(target=self._download, daemon=True).start()
        return self.status()

    def _download(self):
        try:
            with _fetch(LATEST) as response:
                release = json.loads(response.read(4 * 1024 * 1024))
            version = release["tag_name"]
            if release.get("draft") or release.get("prerelease"):
                raise ValueError("Expected a stable release")
            if _version(version) <= _version(__version__):
                self._set(phase="current")
                return
            suffix = (f"macOS-{platform.machine()}.dmg" if sys.platform == "darwin"
                      else "Windows-x64.zip")
            name = f"Ayah-Studio-{version.removeprefix('v')}-{suffix}"
            assets = {asset["name"]: asset for asset in release["assets"]}
            asset, checksum = assets[name], assets[name + ".sha256"]
            with _fetch(checksum["browser_download_url"]) as response:
                fields = response.read(1024).decode("utf-8-sig").split()
            if len(fields) != 2 or fields[1] != name or not re.fullmatch(r"[a-fA-F0-9]{64}", fields[0]):
                raise ValueError("Invalid release checksum")
            # Same filesystem as target allows atomic renames after exit.
            self.work = Path(tempfile.mkdtemp(prefix=".ayah-update-", dir=self.target.parent))
            archive = self.work / name
            size = int(asset["size"])
            if not 0 < size <= 4 * 1024**3:
                raise ValueError("Invalid release size")
            self._set(phase="downloading", version=version)
            digest, received = hashlib.sha256(), 0
            with _fetch(asset["browser_download_url"]) as response, archive.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > size:
                        raise ValueError("Release size mismatch")
                    file.write(chunk)
                    digest.update(chunk)
                    self._set(percent=round(received / size * 100))
            if received != size or digest.hexdigest() != fields[0].lower():
                raise ValueError("Release checksum mismatch")
            self._set(phase="preparing")
            self.staged = (_extract_macos(archive, self.work, version)
                           if sys.platform == "darwin" else _extract_windows(archive, self.work))
            archive.unlink()
            self._set(phase="ready")
        except Exception as exc:
            self.cleanup()
            self._set(phase="error", error=str(exc))

    def install(self):
        """Start an independent helper; caller closes the window after readiness."""
        with self.lock:
            if self.state["phase"] != "ready":
                raise RuntimeError("Update is not ready")
            self.state["phase"] = "installing"
        try:
            if sys.platform == "darwin":
                script = self.work / "install.sh"
                script.write_text(_MAC_HELPER, encoding="utf-8")
                command = ["/bin/sh", str(script), str(self.work), str(os.getpid()),
                           str(self.target), str(self.staged)]
                options = {"start_new_session": True}
            else:
                script = self.work / "install.ps1"
                script.write_text(_WINDOWS_HELPER, encoding="utf-8-sig")
                (self.work / "update.json").write_text(json.dumps({
                    "pid": os.getpid(), "target": str(self.target), "source": str(self.staged)
                }), encoding="utf-8-sig")
                command = [str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"),
                           "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)]
                options = {"creationflags": subprocess.CREATE_NO_WINDOW}
            process = subprocess.Popen(command, cwd=self.work, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
            deadline = time.monotonic() + 10
            while not (self.work / "ready").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    process.terminate() if process.poll() is None else None
                    raise RuntimeError("Unable to start update installer")
                time.sleep(0.05)
            return process
        except Exception:
            self._set(phase="ready")
            raise

    def cleanup(self):
        if self.work and self.status()["phase"] != "installing":
            shutil.rmtree(self.work, ignore_errors=True)
            self.work = self.staged = None
