"""CLI batch entry point.

Walks an audio directory of per-sura files named ``NNN.<ext>`` (``001.mp3`` …
``114.flac``), runs the forced-alignment pipeline on each, and merges the
resulting per-ayah start timings into an output SQLite project DB via
``data.project_db.ProjectDB.write_timings`` (``source='auto'``, which never
clobbers ``edited``/``verified`` rows).

Light by design: heavy work lives behind ``core.pipeline`` / ``data.project_db``
and is imported lazily so ``--help`` stays instant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Audio extensions accepted for a sura file (decode.decode handles them all).
_AUDIO_EXTS = (".mp3", ".m4a", ".ogg", ".flac", ".wav")

# db_version row for generated DBs: code derived from the audio folder name,
# version bumped when the output schema/semantics change.
_DB_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="surah-cutter",
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
        "--reciter",
        default=None,
        help="known-reciter id; if it matches a bundled reference, use its "
        "timings instead of running the model",
    )
    return p


def _discover(audio_dir: Path) -> list[tuple[int, Path]]:
    """Find ``NNN.<ext>`` files and map them to sura numbers (1..114).

    Returns a list of ``(surah, path)`` sorted by surah. The stem must be a
    1..114 integer (zero-padded or not); other files are ignored. If two files
    map to the same sura (e.g. ``001.mp3`` and ``1.wav``), the first by extension
    precedence (:data:`_AUDIO_EXTS` order) wins, then by name.
    """
    best: dict[int, Path] = {}
    ext_rank = {ext: i for i, ext in enumerate(_AUDIO_EXTS)}
    for path in sorted(audio_dir.iterdir() if audio_dir.is_dir() else []):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in ext_rank:
            continue
        stem = path.stem
        if not stem.isdigit():
            continue
        surah = int(stem)
        if not 1 <= surah <= 114:
            continue
        cur = best.get(surah)
        if cur is None or ext_rank[ext] < ext_rank[cur.suffix.lower()]:
            best[surah] = path
    return sorted(best.items())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"surah-cutter: not a directory: {audio_dir}", file=sys.stderr)
        return 2

    jobs = _discover(audio_dir)
    if not jobs:
        print(
            f"surah-cutter: no NNN.{{mp3,m4a,ogg,flac,wav}} files in {audio_dir}",
            file=sys.stderr,
        )
        return 1

    # Derive db_version code from the folder name unless overridden.
    code = args.code or (audio_dir.resolve().name or "reciter")

    # Lazy heavy imports.
    from surah_cutter.core import pipeline
    from surah_cutter.data.project_db import ProjectDB

    db = ProjectDB.open(args.output, code=code, version=_DB_VERSION)
    total = len(jobs)
    failures = 0
    try:
        for i, (surah, path) in enumerate(jobs, start=1):
            print(f"[{i}/{total}] sura {surah:03d} <- {path.name} ...", flush=True)
            try:
                rows = pipeline.process_surah(
                    str(path), surah, reciter=args.reciter
                )
            except Exception as exc:  # noqa: BLE001 - keep batch going, report
                failures += 1
                print(f"    FAILED sura {surah}: {exc}", file=sys.stderr, flush=True)
                continue
            written = db.write_timings(rows, source="auto")
            print(f"    sura {surah:03d}: {written} ayah timings written", flush=True)
    finally:
        db.close()

    print(f"surah-cutter: done ({total - failures}/{total} suras ok) -> {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
