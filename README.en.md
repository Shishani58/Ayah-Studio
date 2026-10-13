<div align="center">
  <img src="surah_cutter/webgui/static/mark.svg" width="80" alt="Ayah Studio">
  <h1>Ayah Studio</h1>
  <p>Turn Qur’an recitation audio into ayah timings.</p>
  <p><a href="README.md">Русский</a> · <a href="https://github.com/Shishani58/Ayah-Studio/releases">Releases</a> · <a href="LICENSE">MIT</a></p>
  <img src="docs/images/review.jpg" width="900" alt="Ayah review with waveform, timing markers and Arabic text">
</div>

A local macOS and Windows app for aligning, listening to and reviewing Qur’an recitation.
Your audio stays on your computer. The app interface is in Russian.

## Install

| Platform | Download 0.2.0 |
| --- | --- |
| macOS · Apple Silicon | [DMG](https://github.com/Shishani58/Ayah-Studio/releases/download/v0.2.0/Ayah-Studio-0.2.0-macOS-arm64.dmg) |
| Windows · x64 | [ZIP with ready-to-run EXE](https://github.com/Shishani58/Ayah-Studio/releases/download/v0.2.0/Ayah-Studio-0.2.0-Windows-x64.zip) |

For **Apple Silicon Macs** (M1 or newer). Open the DMG, drag **Ayah Studio**
to **Applications**, eject the disk and launch the app. Python and the default model are bundled. Optional models can be downloaded in Settings.

This build is not Apple-notarized. If macOS blocks the first launch, see
[Apple’s guidance](https://support.apple.com/en-us/102445).
**Windows:** extract the entire ZIP and launch **Ayah Studio.exe**. Keep
`_internal` next to the EXE. The native window requires
[Microsoft WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).
The EXE is not code-signed. Intel Macs have not been verified.

## Features

- Three modes: pause-based fast alignment with local checks, full alignment and words.
- Bundled Quran Phonetics; optional Muaalem, FastConformer Arabic and Whisper Large-v3.
- Waveform, Arabic text, ayah loop and playback speed controls.
- Drag timing markers; manual edits survive regeneration.
- Switch projects and choose light or dark appearance.
- Search and filter 114 surahs, resume incomplete work.
- Export a SQLite timing database for Qur’an apps.

<img src="docs/images/review-dark.jpg" width="900" alt="Ayah editor in dark appearance">

<details>
<summary>Alignment models</summary>
<img src="docs/images/models.jpg" width="900" alt="Alignment models in Settings">
</details>

## Quick start

1. Select your audio folder: one file per surah, named `001.mp3` … `114.mp3`.
2. Add a project in Settings → Projects and select an alignment mode.
3. Open a surah, listen, adjust and mark ayahs as verified.
4. Export when all 114 surahs are ready.

Automatic timings are a draft. Review boundaries by ear before publishing.

## Development

[Build the app and DMG](docs/development/packaging.md) ·
[Architecture](docs/project.md) · [Project status](docs/STATUS.md)

Code is MIT-licensed. Bundled data, fonts and models retain their upstream
licenses: [acknowledgements](docs/attributions.md).
