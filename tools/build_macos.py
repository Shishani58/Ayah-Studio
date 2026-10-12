"""Build a self-contained Ayah Studio DMG. Run with the project's [build] extra."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "Ayah Studio"


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="Package the existing .app")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("DMG builds require macOS")
    if not args.skip_build:
        run(sys.executable, "-m", "PyInstaller", "surah_cutter.spec", "--noconfirm", "--clean")
    bundle = ROOT / "dist" / f"{APP_NAME}.app"
    with (bundle / "Contents/Info.plist").open("rb") as file:
        version = plistlib.load(file)["CFBundleShortVersionString"]
    # Check the shipped executable, not the source interpreter.
    run(str(bundle / "Contents/MacOS" / APP_NAME), "--self-check")
    run("codesign", "--verify", "--deep", "--strict", str(bundle))
    dmg = ROOT / "dist" / f"Ayah-Studio-{version}-macOS-{platform.machine()}.dmg"
    with tempfile.TemporaryDirectory(prefix="ayah-studio-dmg-") as directory:
        staging = Path(directory)
        run("ditto", str(bundle), str(staging / bundle.name))
        (staging / "Applications").symlink_to("/Applications", target_is_directory=True)
        (staging / "Установка.txt").write_text(
            "Ayah Studio\n\nПеретащите Ayah Studio.app в Applications (Программы).\n"
            "Извлеките диск и откройте приложение из Программ.\n"
            "Python, Терминал и исходники для работы не нужны.\n\n"
            "Drag Ayah Studio.app to Applications, eject this disk, then launch the app.\n",
            encoding="utf-8",
        )
        run("hdiutil", "create", "-volname", APP_NAME, "-srcfolder", str(staging),
            "-format", "UDZO", "-ov", str(dmg))
    profile = os.environ.get("AYAH_NOTARY_PROFILE")
    if profile:
        if not os.environ.get("AYAH_CODESIGN_IDENTITY"):
            raise SystemExit("Notarization requires AYAH_CODESIGN_IDENTITY (Developer ID Application)")
        run("xcrun", "notarytool", "submit", str(dmg), "--keychain-profile", profile, "--wait")
        run("xcrun", "stapler", "staple", str(dmg))
        run("xcrun", "stapler", "validate", str(dmg))
    run("hdiutil", "verify", str(dmg))
    hasher = hashlib.sha256()
    with dmg.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    dmg.with_suffix(".dmg.sha256").write_text(f"{digest}  {dmg.name}\n", encoding="utf-8")
    print(f"Ready: {dmg}")
    if not profile:
        print("Local build: not notarized by Apple. Public distribution needs Developer ID + notarization.")


if __name__ == "__main__":
    main()
