"""CLI batch entry point.

Walks an audio directory of per-sura files with a leading sura number
(``001.mp3``, ``003_recitation.m4a``), runs the alignment pipeline on each, and merges the
resulting per-ayah start timings into an output SQLite project DB via
``data.alignment_result.write_alignment_result`` (which never clobbers
``edited``/``verified`` rows).

Light by design: heavy work lives behind ``core.pipeline`` / ``data.project_db``
and is imported lazily so ``--help`` stays instant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from surah_cutter.data.project_state import discover_audio

# Audio extensions accepted for a sura file (decode.decode handles them all).

# db_version row for generated DBs: code derived from the audio folder name,
# version bumped when the output schema/semantics change.
_DB_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ayah-studio-cli",
        description="Generate per-ayah timecodes for Qur'an recitation audio.",
    )
    p.add_argument("audio_dir", help="folder with surah audio files (001.mp3 … 114.mp3)")
    p.add_argument("-o", "--output", default="reciter.db", help="output SQLite database path")
    p.add_argument(
        "--code",
        default=None,
        help="db_version code (reciter id); defaults to the audio folder name",
    )
    p.add_argument(
        "--mode",
        default="full",
        choices=("full", "fast", "tiled", "hybrid", "fingerprint"),
        help="alignment mode (default: full)",
    )
    p.add_argument(
        "--ayah-audio-dir",
        default=None,
        help="folder with same-recording per-ayah clips for fingerprint mode",
    )
    return p


def _discover(audio_dir: Path) -> list[tuple[int, Path]]:
    """Use the same deterministic recording choice as both GUIs."""
    return list(discover_audio(audio_dir).items())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"Ayah Studio: not a directory: {audio_dir}", file=sys.stderr)
        return 2

    jobs = _discover(audio_dir)
    if not jobs:
        print(
            f"Ayah Studio: no NNN.{{mp3,m4a,ogg,flac,wav}} files in {audio_dir}",
            file=sys.stderr,
        )
        return 1

    # Derive db_version code from the folder name unless overridden.
    code = args.code or (audio_dir.resolve().name or "reciter")

    # Lazy heavy imports.
    from surah_cutter.core import pipeline
    from surah_cutter.core.cancellation import GenerationCancelled
    from surah_cutter.data.alignment_result import write_alignment_result, write_unsaved_diagnostics
    from surah_cutter.data.project_db import ProjectDB

    db = ProjectDB.open(args.output, code=code, version=_DB_VERSION)
    total = len(jobs)
    failures = 0
    try:
        for i, (surah, path) in enumerate(jobs, start=1):
            print(f"[{i}/{total}] sura {surah:03d} <- {path.name} ...", flush=True)
            try:
                trace: dict = {}
                rows = pipeline.process_surah(
                    str(path),
                    surah,
                    mode=args.mode,
                    ayah_audio_dir=args.ayah_audio_dir,
                    trace=trace,
                )
                save_result = write_alignment_result(
                    db, rows, trace=trace, requested_mode=args.mode, db_path=args.output,
                )
                written, _ = save_result
            except (Exception, GenerationCancelled) as exc:  # keep batch going, preserve cancellation
                write_unsaved_diagnostics(
                    db_path=args.output, surah=surah, requested_mode=args.mode,
                    trace=trace, outcome="cancelled" if isinstance(exc, GenerationCancelled) else "failed",
                    error=exc,
                )
                if isinstance(exc, GenerationCancelled):
                    raise
                failures += 1
                print(f"    FAILED sura {surah}: {exc}", file=sys.stderr, flush=True)
                continue
            print(f"    sura {surah:03d}: {written} ayah timings written", flush=True)
            if save_result.unresolved_ayahs:
                print(f"    partial FAST: {len(save_result.unresolved_ayahs)} unresolved ayahs "
                      "(no timestamps generated)", flush=True)
            if save_result.diagnostics_error:
                print(f"    saved_with_diagnostics_error: {save_result.diagnostics_error}; "
                      f"paths={save_result.log_paths}", file=sys.stderr, flush=True)
    finally:
        db.close()

    print(f"Ayah Studio: done ({total - failures}/{total} suras ok) -> {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
