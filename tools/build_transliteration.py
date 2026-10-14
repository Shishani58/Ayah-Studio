"""Rebuild and validate the phonetic word cache without changing Quran text.

Offline reconciliation: python -m tools.build_transliteration --source data/word_tr_cache.json
Source dataset rebuild (development dependencies/network): omit --source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from surah_cutter.data.transliteration import CACHE_PATH, build_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing source-derived JSON (offline)")
    parser.add_argument("--output", type=Path, default=CACHE_PATH)
    args = parser.parse_args()
    mapping = build_cache(cache_path=args.output, source_path=args.source)
    print(f"Validated {len(mapping)} ayahs, {sum(map(len, mapping.values()))} words")


if __name__ == "__main__":
    main()
