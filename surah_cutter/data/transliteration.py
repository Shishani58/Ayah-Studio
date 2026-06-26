"""Per-word Latin transliterations matching the alignment model's vocabulary.

The forced-alignment acoustic model
(:data:`surah_cutter.core.align.DEFAULT_MODEL_ID`,
``TBOGamer22/wav2vec2-quran-phonetics``) emits a **Latin transliteration**, not
Arabic. Its 38-char vocabulary was built from the ``word_tr`` column of the HF
dataset ``Buraaq/quran-md-words`` (per-word rows, with audio + text columns).

To forced-align the *known* Quran text we need each word's transliteration in
exactly that convention. Rather than reimplement Arabic->Latin G2P, we reuse the
dataset's own ``word_tr`` strings. This module, **lazily on first use**:

1. downloads the dataset's parquet shards via ``huggingface_hub`` (text columns
   only — we never read the audio bytes);
2. defensively locates the surah / ayah / word-order columns and the
   transliteration column (schema names are not pre-confirmed; we probe
   plausible variants at runtime);
3. builds a ``(surah, ayah, word_index_in_ayah) -> word_tr`` map and caches it to
   a small JSON under ``models/`` so subsequent runs are offline + instant;
4. exposes :func:`surah_word_tr`, aligned 1:1 with
   :func:`surah_cutter.data.quran_text.surah_words`.

Pure-ish: heavy deps (``huggingface_hub``, ``pyarrow``) are imported lazily
inside functions so ``core``/``data`` import without them. ``quran_text`` is the
only intra-package dependency and is stdlib-only.

Reconciliation (documented fallback): the canonical word order/count is the
mushaf DB (:func:`quran_text.surah_words`). We key the dataset map by
``(surah, ayah)`` plus the word's 0-based index *within its ayah*. If the
dataset's per-ayah segmentation disagrees with the mushaf word count for an
ayah, we still return one transliteration per mushaf word, padding any missing
tail positions with ``""`` (empty -> contributes no tokens, only a word
delimiter). This keeps the returned list strictly 1:1 with ``surah_words`` so
the aligner's word grouping never desynchronises.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from surah_cutter.data import quran_text

__all__ = [
    "DATASET_REPO",
    "CACHE_PATH",
    "surah_word_tr",
    "build_cache",
]

#: HF dataset the model's vocabulary/transliterations come from.
DATASET_REPO = "Buraaq/quran-md-words"

#: Transliteration map, JSON keyed "sura:ayah" -> list[str] (word translits in
#: ayah order). VENDORED in the repo's data/ dir (~1 MB) and shipped, so end
#: users never download the 2 GB source dataset (nor need pyarrow/huggingface_hub
#: at runtime). Rebuilding it from the dataset (build_cache) is a dev-only task.
#: transliteration.py -> data/ -> surah_cutter/ -> <repo>/data/
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "word_tr_cache.json"

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
    Missing entries (segmentation mismatch, unknown word) come back as ``""``;
    the caller treats ``""`` as "no acoustic tokens, just a word delimiter".

    The dataset map is built + cached on first use (network on first run only).
    """
    words = quran_text.surah_words(surah, db_path=db_path)
    mapping = _cached_map(cache_path)

    # Group mushaf words by ayah, preserving recitation order, to assign each a
    # 0-based index within its ayah (the key dimension we reconcile on).
    out: list[str] = []
    per_ayah_pos: dict[int, int] = {}
    for w in words:
        pos = per_ayah_pos.get(w.ayah, 0)
        per_ayah_pos[w.ayah] = pos + 1
        tr_list = mapping.get(f"{surah}:{w.ayah}")
        if tr_list is not None and pos < len(tr_list):
            out.append(tr_list[pos])
        else:
            out.append("")
    return out


# --- cache plumbing -------------------------------------------------------


@lru_cache(maxsize=4)
def _cached_map(cache_path: str | Path = CACHE_PATH) -> dict[str, list[str]]:
    """Load the transliteration map, building (and persisting) it if absent.

    Returns a dict ``"sura:ayah" -> [word_tr, ...]`` (ayah word order). The
    ``lru_cache`` keeps it in memory for the process after the first load.
    """
    path = Path(cache_path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            # JSON keys are strings already; values are lists of strings.
            return {str(k): list(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError, ValueError):
            # Corrupt cache -> rebuild from source.
            pass
    return build_cache(cache_path=path)


def build_cache(
    *,
    cache_path: str | Path = CACHE_PATH,
) -> dict[str, list[str]]:
    """Download the dataset, build the transliteration map, and persist it.

    Network + parquet read happen here (lazy heavy imports). Writes the map to
    ``cache_path`` as JSON and returns it. Safe to call directly to warm the
    cache offline-prep style.
    """
    mapping = _build_map_from_dataset()
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
