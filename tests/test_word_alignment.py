from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, word_align
from surah_cutter.data import transliteration
from surah_cutter.data.export_db import export_db
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.data.quran_text import Word


@pytest.fixture
def word_model(monkeypatch):
    bundle = SimpleNamespace(vocab={"a": 1, "b": 2, "|": 3}, delim_id=3, blank_id=0)
    monkeypatch.setattr(align, "_get_onnx_bundle", lambda: bundle)
    monkeypatch.setattr(align, "ALIGN_LAG_MS", 0)
    # Ayah 2 starts at original surah index 3, not transliteration index 0.
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["b", "b", "b", "a", "b"])
    words = [Word(114, 2, "first", 3), Word(114, 2, "second", 4)]
    emissions = np.full((30, 4), -10, dtype=np.float32)
    emissions[:, 0] = 0
    for a, b, token in [(3, 7, 1), (10, 12, 3), (17, 21, 2)]:
        emissions[a:b, 0] = -10
        emissions[a:b, token] = 0
    monkeypatch.setattr(align, "_emissions", lambda *args: emissions)
    return words


def test_word_alignment_uses_canonical_indices_and_local_times(word_model):
    rows = word_align.align_ayah_words(np.zeros(16000), 16000, word_model)
    assert [(i, t) for i, t, _ in rows] == [(3, 100), (4, 567)]
    assert all(0 <= c <= 1 for _, _, c in rows)


def test_incomplete_path_is_rejected_without_uniform_fallback(word_model, monkeypatch):
    monkeypatch.setattr(word_align, "forced_align", lambda *args, **kwargs:
                        (np.ones((1, 30), dtype=int), np.zeros((1, 30))))
    with pytest.raises(ValueError, match="все слова"):
        word_align.align_ayah_words(np.zeros(16000), 16000, word_model)


def test_missing_word_tokens_are_rejected(word_model, monkeypatch):
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["a"] * 4 + ["?"])
    with pytest.raises(ValueError, match="Не все слова"):
        word_align.align_ayah_words(np.zeros(16000), 16000, word_model)


def test_word_merge_preserves_manual_work_and_is_atomic(tmp_path):
    with ProjectDB.open(str(tmp_path / "p.db"), "test", 1) as db:
        rows = [(0, 100, .8), (1, 500, .9), (2, 900, .9)]
        db.write_word_timings(1, 1, rows, 0, 1000, "test")
        db.set_word_time(1, 1, 1, 550, 0, 1000)
        db.verify_word(1, 1, 2, 0, 1000)
        assert db.write_word_timings(1, 1, [(0, 120, .9), (1, 600, .9), (2, 950, .9)],
                                     0, 1000, "test") == 1
        saved = db.get_word_timings(1, 1)
        assert [r["time"] for r in saved] == [120, 550, 900]
        assert [r["status"] for r in saved] == ["auto", "edited", "verified"]
        with pytest.raises(ValueError):
            db.write_word_timings(1, 1, [(0, 600, .9), (1, 700, .9), (2, 950, .9)],
                                  0, 1000, "test")
        assert db.get_word_timings(1, 1) == saved
        with pytest.raises(ValueError, match="изменились"):
            db.write_word_timings(1, 1, rows, 10, 1000, "test")
        with pytest.raises(ValueError):
            db.verify_word(1, 1, 1, 10, 1000)
        assert db._conn.execute("SELECT old_time, new_time FROM word_edit_history").fetchall() == [(500, 550)]
        assert db.get_timings() == []


def test_export_does_not_include_word_service_tables(tmp_path):
    import sqlite3

    source, dest = tmp_path / "p.db", tmp_path / "out.db"
    with ProjectDB.open(str(source), "test", 1) as db:
        db.write_timings([(1, 1, 0)], "test")
        db.write_word_timings(1, 1, [(0, 100, .9)], 0, 1000, "test")
    export_db(str(source), str(dest), strict=False)
    with sqlite3.connect(dest) as conn:
        assert {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"timings", "db_version"}
