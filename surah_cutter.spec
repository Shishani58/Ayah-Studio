# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the current Ayah Studio desktop app.

Builds a windowed application from ``surah_cutter/webgui/launch.py``.

Bundle layout (critical):
    The runtime resolves bundled data via
    ``Path(__file__).resolve().parents[2] / "data" / ...`` from inside
    ``surah_cutter/data/*.py`` (see ``quran_text.py``).
    In a frozen bundle ``__file__`` lives under ``<bundle>/surah_cutter/data/``,
    so ``parents[2]`` is the bundle root. Therefore ``data/`` and ``assets/``
    MUST be placed at the bundle root (datas target ``.``), as siblings of the
    ``surah_cutter`` package — otherwise the Quran DB and font are not found.

Platform outputs:
    * macOS   -> ``dist/Ayah Studio.app`` (BUNDLE, windowed .app)
    * Windows -> ``dist/Ayah Studio/Ayah Studio.exe`` (windowed .exe)
    * Linux   -> ``dist/Ayah Studio/Ayah Studio``

Build:  pyinstaller surah_cutter.spec --noconfirm
Do not edit generated bundles by hand.
"""

import os
import sys
import uv
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# ``__file__`` is not defined when PyInstaller execs the spec; SPECPATH is.
ROOT = Path(SPECPATH).resolve()  # noqa: F821  (SPECPATH injected by PyInstaller)

block_cipher = None


# --- bundled data (target "." => bundle root, sibling of surah_cutter/) -------
datas = [
    (str(ROOT / "data" / "quran_mushaf.db"), "data"),
    (str(ROOT / "data" / "word_tr_cache.json"), "data"),
    (str(ROOT / "data" / "model_text_targets.json"), "data"),
    (str(ROOT / "surah_cutter/core/model_catalog.json"), "surah_cutter/core"),
    (str(ROOT / "surah_cutter/core/runtime_locks"), "surah_cutter/core/runtime_locks"),
    (str(ROOT / "surah_cutter/core/model_worker.py"), "surah_cutter/core"),
    (str(ROOT / "assets" / "fonts" / "qpc_hafs.ttf"), "assets/fonts"),
    # The FastAPI single-page UI resolves this relative to server.py.
    (str(ROOT / "surah_cutter" / "webgui" / "static"), "surah_cutter/webgui/static"),
    # Ship licensing/attribution alongside the binary.
    (str(ROOT / "LICENSE"), "."),
]

# Offline alignment uses the int8 CTC model, its vocabulary and Silero VAD.
# Do not include the 360 MB FP32 export: it is a development oracle only.
for _name in (
    "wav2vec2-quran-phonetics.onnx",
    "wav2vec2-quran-phonetics.onnx.vocab.json",
    "wav2vec2-quran-phonetics.onnx.meta.json",
    "silero_vad.onnx",
):
    _path = ROOT / "models" / _name
    if not _path.is_file():
        raise FileNotFoundError(f"Required runtime model is missing: {_path}")
    datas.append((str(_path), "models"))


# --- hidden imports -----------------------------------------------------------
# Imports are dynamic in the local web server and in pywebview's platform
# selector, so make the macOS backend and Uvicorn's auto-selected pieces explicit.
hiddenimports = collect_submodules("onnxruntime") + [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "soundfile",
    "imageio_ffmpeg",
]

if sys.platform == "darwin":
    hiddenimports.append("webview.platforms.cocoa")
elif sys.platform == "win32":
    hiddenimports.extend(["webview.platforms.winforms", "webview.platforms.edgechromium"])

# Keep the dev oracle (torch/transformers) out of the shipped app.
excludes = [
    "torch",
    "transformers",
    "pytest",
    "tkinter",
    "matplotlib",
]


a = Analysis(
    [str(ROOT / "surah_cutter" / "webgui" / "launch.py")],
    pathex=[str(ROOT)],
    binaries=[(uv.find_uv_bin(), "uv-bin")],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Ayah Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=os.environ.get("AYAH_CODESIGN_IDENTITY"),
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Ayah Studio",
)

# macOS: wrap the COLLECT tree into a proper .app bundle.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Ayah Studio.app",
        icon=str(ROOT / "assets" / "AyahStudio.icns"),
        bundle_identifier="org.quranacademy.ayahstudio",
        info_plist={
            "CFBundleName": "Ayah Studio",
            "CFBundleDisplayName": "Ayah Studio",
            "CFBundleShortVersionString": "0.3.1",
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.music",
        },
    )
