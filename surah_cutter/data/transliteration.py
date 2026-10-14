"""Per-word Latin transliterations matching the alignment model's vocabulary.

The forced-alignment acoustic model
(:data:`surah_cutter.core.align.DEFAULT_MODEL_ID`,
``TBOGamer22/wav2vec2-quran-phonetics``) emits a **Latin transliteration**, not
Arabic. Its 38-char vocabulary was built from the ``word_tr`` column of the HF
dataset ``Buraaq/quran-md-words`` (per-word rows, with audio + text columns).

To forced-align the *known* Quran text we need each word's transliteration in
exactly that convention. Rather than reimplement Arabic->Latin G2P, we reuse the
dataset's own ``word_tr`` strings. The vendored cache is validated against every
canonical word before use. Rebuilding from the source dataset is a
development-only operation.

The source joins ``بَعۡدَ مَا`` in 2:181, 8:6 and 13:37. The reproducible
reconciliation below splits only those exact source strings at the canonical
boundary, preserving their transliteration characters. Every other count or
empty-target mismatch fails closed; no padding or inferred transliteration.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from surah_cutter.data import quran_text
from surah_cutter.resources import resource_path

__all__ = [
    "DATASET_REPO",
    "CACHE_PATH",
    "surah_word_tr",
    "build_cache",
    "reconcile_map",
]

#: HF dataset the model's vocabulary/transliterations come from.
DATASET_REPO = "Buraaq/quran-md-words"

#: Transliteration map, JSON keyed "sura:ayah" -> list[str] (word translits in
#: ayah order). VENDORED in the repo's data/ dir (~1 MB) and shipped, so end
#: users never download the 2 GB source dataset (nor need pyarrow/huggingface_hub
#: at runtime). Rebuilding it from the dataset (build_cache) is a dev-only task.
#: transliteration.py -> data/ -> surah_cutter/ -> <repo>/data/
CACHE_PATH = resource_path("data", "word_tr_cache.json")

# Candidate column names (lower-cased) probed at runtime; the dataset schema is
# not pre-confirmed, so we accept plausible variants and pick the first match.
_SURAH_COLS = (
    "surah_id",
    "surah",
    "sura",
    "surah_no",
    "sura_no",
    "surah_number",
    "chapter",
    "chapter_no",
)
_AYAH_COLS = (
    "ayah_id",
    "ayah",
    "aya",
    "ayah_no",
    "aya_no",
    "ayah_number",
    "verse",
    "verse_no",
    "verse_number",
)
_WORD_INDEX_COLS = (
    "word_no",
    "word_number",
    "word_index",
    "word_idx",
    "position",
    "pos",
    "word_pos",
    "word",  # sometimes a 1-based word position lives here
)
_TR_COLS = (
    "word_tr",
    "word_translit",
    "word_transliteration",
    "transliteration",
    "translit",
    "tr",
    "text_tr",
)


def surah_word_tr(
    surah: int,
    *,
    db_path: str | Path = quran_text.DEFAULT_DB_PATH,
    cache_path: str | Path = CACHE_PATH,
) -> list[str]:
    """Return per-word transliterations for ``surah``, 1:1 with ``surah_words``.

    The i-th returned string is the model-vocabulary transliteration of the
    i-th word of :func:`quran_text.surah_words(surah)` (same recitation order).
    Missing, empty or differently segmented entries raise ``ValueError``.
    Loading the shipped cache is offline; rebuilding it is explicit.
    """
    words = quran_text.surah_words(surah, db_path=db_path)
    mapping = _cached_map(cache_path, db_path)
    out: list[str] = []
    for ayah in dict.fromkeys(w.ayah for w in words):
        out.extend(mapping[f"{surah}:{ayah}"])
    return out


# --- cache plumbing -------------------------------------------------------


# Source segmentation exceptions, using 0-based positions within each ayah.
# These split the original target, without generating new Latin characters.
_JOINED_WORDS = {
    "2:181": (2, "baʿdamā"),
    "8:6": (3, "baʿda mā"),
    "13:37": (7, "baʿdamā"),
}


def reconcile_map(
    mapping: dict[str, list[str]],
    *,
    db_path: str | Path = quran_text.DEFAULT_DB_PATH,
) -> dict[str, list[str]]:
    """Reconcile known source joins and validate all canonical word targets.

    Idempotent, so both the source-derived map and an already rebuilt cache
    can be checked using the same code. Never mutates the input or Quran DB.
    """
    if not isinstance(mapping, dict):
        raise ValueError("Transliteration cache must be an ayah mapping")
    canonical: dict[str, list[str]] = {}
    for surah in quran_text.surah_ayah_counts(db_path):
        for word in quran_text.surah_words(surah, db_path=db_path):
            canonical.setdefault(f"{surah}:{word.ayah}", []).append(word.text)
    if mapping.keys() != canonical.keys():
        raise ValueError("Transliteration cache does not contain the canonical ayah set")
    result = {}
    for key, words in canonical.items():
        targets = mapping[key]
        if not isinstance(targets, list) or any(
            not isinstance(tr, str) or not tr.strip() for tr in targets
        ):
            raise ValueError(f"Empty or invalid transliteration target in {key}")
        targets = list(targets)
        if len(targets) != len(words) and key in _JOINED_WORDS:
            position, joined = _JOINED_WORDS[key]
            if (len(targets) == len(words) - 1
                    and words[position:position + 2] == ["بَعۡدَ", "مَا"]
                    and targets[position] == joined):
                targets[position:position + 1] = [joined[:5], joined[5:].strip()]
        if len(targets) != len(words):
            raise ValueError(
                f"Transliteration segmentation mismatch in {key}: "
                f"{len(targets)} targets for {len(words)} canonical words"
            )
        result[key] = targets
    return result


@lru_cache(maxsize=4)
def _cached_map(
    cache_path: str | Path = CACHE_PATH,
    db_path: str | Path = quran_text.DEFAULT_DB_PATH,
) -> dict[str, list[str]]:
    """Read and validate the shipped cache, without implicit network/rebuild."""
    with Path(cache_path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return reconcile_map(raw, db_path=db_path)


def build_cache(
    *,
    cache_path: str | Path = CACHE_PATH,
    source_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Build, reconcile and persist a fully validated transliteration map.

    ``source_path`` reuses source-derived JSON for an offline reproducible build.
    Otherwise download the dataset's text columns (lazy development dependencies).
    Invalid input is rejected before the destination cache is replaced.
    """
    if source_path is None:
        raw = _build_map_from_dataset()
    else:
        with Path(source_path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    mapping = reconcile_map(raw)
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False)
    tmp.replace(path)
    return mapping


# --- dataset ingestion ----------------------------------------------------


def _build_map_from_dataset() -> dict[str, list[str]]:
    """Download parquet shards and assemble ``"sura:ayah" -> [word_tr]``.

    Reads ONLY the columns we need (no audio bytes): each parquet shard is opened
    over ``HfFileSystem`` and pyarrow's column projection fetches only the text
    column chunks via HTTP range requests, so the ~2 GB of embedded audio is
    never transferred. Defensively resolves the schema, sorts words within each
    ayah by their word-order column (or by row order as a fallback), and returns
    the assembled map.
    """
    import pyarrow.parquet as pq  # lazy heavy dep

    shards = _dataset_parquet_paths()
    if not shards:
        raise RuntimeError(
            f"No parquet shards found for dataset {DATASET_REPO!r}; cannot build "
            "the transliteration cache."
        )

    from huggingface_hub import HfFileSystem  # lazy heavy dep

    fs = HfFileSystem()

    # Collected rows: (surah, ayah, word_pos_or_None, word_tr). word_pos may be
    # None when the dataset has no explicit word-order column; we then rely on
    # parquet row order (which is the dataset's natural per-word order).
    rows: list[tuple[int, int, int | None, str]] = []

    for shard in sorted(shards):
        with fs.open(shard) as fh:
            pf = pq.ParquetFile(fh)
            s_col, a_col, w_col, tr_col = _resolve_columns(list(pf.schema_arrow.names))
            read_cols = [s_col, a_col, tr_col]
            if w_col is not None:
                read_cols.append(w_col)
            data = pf.read(columns=read_cols).to_pydict()
        s_vals = data[s_col]
        a_vals = data[a_col]
        tr_vals = data[tr_col]
        w_vals = data[w_col] if w_col is not None else [None] * len(s_vals)
        for s, a, w, tr in zip(s_vals, a_vals, w_vals, tr_vals):
            s_i = _to_int(s)
            a_i = _to_int(a)
            if s_i is None or a_i is None:
                continue
            rows.append((s_i, a_i, _to_int(w), _clean_tr(tr)))

    return _assemble(rows)


def _assemble(
    rows: list[tuple[int, int, int | None, str]],
) -> dict[str, list[str]]:
    """Group rows by (surah, ayah) and order words within each ayah.

    Ordering: by the explicit word-position column when present, else by the
    insertion (row) order. Returns ``"sura:ayah" -> [word_tr, ...]``.
    """
    groups: dict[tuple[int, int], list[tuple[int, int | None, str]]] = {}
    for seq, (s, a, w, tr) in enumerate(rows):
        groups.setdefault((s, a), []).append((seq, w, tr))

    out: dict[str, list[str]] = {}
    for (s, a), items in groups.items():
        # If any word-position is present, sort by it (None last, then by seq);
        # otherwise sort purely by original row order.
        if any(w is not None for _seq, w, _tr in items):
            items.sort(key=lambda it: (it[1] is None, it[1] if it[1] is not None else 0, it[0]))
        else:
            items.sort(key=lambda it: it[0])
        out[f"{s}:{a}"] = [tr for _seq, _w, tr in items]
    return out


def _dataset_parquet_paths() -> list[str]:
    """List the dataset's parquet shards as ``HfFileSystem`` paths (no download).

    Returns ``hf://``-style paths (e.g.
    ``datasets/Buraaq/quran-md-words/data/train-00000-of-00005.parquet``) that
    :func:`_build_map_from_dataset` opens with column projection, so only the
    text column chunks are fetched — never the embedded audio.
    """
    from huggingface_hub import HfFileSystem  # lazy heavy dep

    fs = HfFileSystem()
    return [str(p) for p in fs.glob(f"datasets/{DATASET_REPO}/**/*.parquet")]


# --- schema resolution & cleaning ----------------------------------------


def _resolve_columns(
    names: list[str],
) -> tuple[str, str, str | None, str]:
    """Pick (surah, ayah, word_pos|None, transliteration) columns from a schema.

    Matching is case-insensitive against known candidate lists. The
    word-position column is optional (``None`` if none of the candidates exist —
    we fall back to row order). Raises ``RuntimeError`` if a required column
    cannot be located so the failure is loud rather than silently wrong.
    """
    lower = {n.lower(): n for n in names}

    def pick(cands: tuple[str, ...]) -> str | None:
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    s_col = pick(_SURAH_COLS)
    a_col = pick(_AYAH_COLS)
    tr_col = pick(_TR_COLS)
    w_col = pick(_WORD_INDEX_COLS)

    missing = [
        label
        for label, col in (("surah", s_col), ("ayah", a_col), ("transliteration", tr_col))
        if col is None
    ]
    if missing:
        raise RuntimeError(
            f"Could not locate column(s) {missing} in dataset {DATASET_REPO!r}; "
            f"available columns: {names}"
        )
    return s_col, a_col, w_col, tr_col  # type: ignore[return-value]


def _to_int(value: object) -> int | None:
    """Best-effort coercion of a schema value to ``int`` (``None`` if unusable)."""
    if value is None:
        return None
    if isinstance(value, bool):  # avoid True/False -> 1/0 surprises
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            return int(m.group())
    return None


def _clean_tr(value: object) -> str:
    """Normalise a raw transliteration cell to a stripped NFC string."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    # The model vocab uses NFC composed forms (e.g. ā, ī, ū, ḥ); normalise so
    # decomposed inputs don't slip past the char->id lookup in align.py.
    return unicodedata.normalize("NFC", text).strip()
