# Packaging / distribution

Frozen desktop builds of the GUI app (`surah-cutter-gui`) via **PyInstaller**.
Config lives in [`surah_cutter.spec`](../../surah_cutter.spec); CI in
[`.github/workflows/build.yml`](../../.github/workflows/build.yml).

## What gets bundled

- `surah_cutter` package (entry: `surah_cutter/gui/app.py`).
- `data/quran_mushaf.db` and `data/references/*.db` — Quran text + reciter
  reference timecodes.
- `assets/fonts/qpc_hafs.ttf` — QPC Hafs font.
- `models/` — ONNX alignment model dir, **only if present and non-empty**
  (graceful: the spec skips it otherwise; ship it for an offline-ready build).
- `LICENSE`.

### Bundle layout matters

At runtime the data loaders resolve files with
`Path(__file__).resolve().parents[2] / "data" / ...`
(`surah_cutter/data/quran_text.py`, `references.py`). In a frozen bundle that
makes the bundle root the anchor, so `data/` and `assets/` are placed at the
bundle root as **siblings of the `surah_cutter` package**. Do not move them.

The dev oracle (`torch`, `transformers`, `ctc-forced-aligner`) is excluded —
the shipped app aligns through `onnxruntime` only.

## Local build

```bash
pip install -e .
pip install pyinstaller
pyinstaller surah_cutter.spec --noconfirm --clean
```

Outputs in `dist/`:

| Platform | Output                                       |
| -------- | -------------------------------------------- |
| macOS    | `dist/surah-cutter.app` (windowed `.app`)    |
| Windows  | `dist/surah-cutter/surah-cutter-gui.exe`     |
| Linux    | `dist/surah-cutter/surah-cutter-gui`         |

Build on the target OS — PyInstaller does not cross-compile.

## Releases (CI)

`build.yml` runs a matrix on `macos-latest` + `windows-latest`:

1. **`workflow_dispatch`** — manual run from the Actions tab; produces per-OS
   artifacts (`surah-cutter-macos.zip`, `surah-cutter-windows.zip`).
2. **Tag push `v*`** (e.g. `git tag v0.1.0 && git push --tags`) — same build,
   plus a `release` job that attaches both zips to a GitHub Release.

The zip preserves the bundle directory structure so the runtime data siblings
stay intact.

## Licensing / attribution

The Quran text DB, reciter reference timecodes, the QPC Hafs font, and any
bundled ONNX model carry their own licenses. Ship the corresponding attribution
(`LICENSE` plus the upstream notices for fonts/data/models) inside the
distributed bundle. Verify each source's terms before publishing a release.
