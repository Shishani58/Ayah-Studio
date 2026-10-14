"""Shared text comparisons for phonetic alignment, independent of FAST phases."""

from __future__ import annotations

import difflib
import os

SIM_CHARS = 26
SIM_PREFIX_WEIGHT = float(os.environ.get("SC_FAST_SIM_PREFIX_WEIGHT", "0.35"))
LONG_VOWELS = frozenset("āīū")


def collapse(text: str) -> str:
    """Collapse repeated characters used for melodic vowel elongation."""
    out: list[str] = []
    for char in text:
        if not out or out[-1] != char:
            out.append(char)
    return "".join(out)


def common_prefix_len_collapsed(left: str, right: str) -> int:
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    return common


def common_prefix_len(left: str, right: str) -> int:
    return common_prefix_len_collapsed(collapse(left), collapse(right))


def unit_len(text: str) -> float:
    """Recitation tempo units: long vowels weigh twice as much."""
    return float(sum(2 if char in LONG_VOWELS else 1 for char in text))


def onset_similarity(decoded: str, opening: str, *, chars: int = SIM_CHARS,
                     prefix_weight: float = SIM_PREFIX_WEIGHT) -> float:
    """Compare collapsed openings, preferring a match at the start of a read."""
    chars = max(1, int(chars))
    left, right = collapse(decoded)[:chars], collapse(opening)[:chars]
    fuzzy = difflib.SequenceMatcher(None, left, right).ratio()
    prefix = common_prefix_len_collapsed(left, right) / max(1, min(chars, len(right)))
    weight = min(1.0, max(0.0, prefix_weight))
    return fuzzy * (1.0 - weight) + prefix * weight
