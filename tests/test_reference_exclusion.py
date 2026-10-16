"""External timing data must not enter alignment, scoring or the app package."""
import ast
import importlib.util
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np

from surah_cutter.core import align, align_tiled, model_store, phonetics, recheck
from surah_cutter.data import transliteration
from surah_cutter.data.onset_evidence import onset_evidence
from surah_cutter.data.quran_text import Word


def test_tiled_aligns_from_audio_and_text_without_database_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Alignment must not open an evaluation database")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["a", "b"])
    monkeypatch.setattr(align, "bundle_source", lambda _: "ctc:test:lag=0")
    matrix = np.full((100, 4), -20., dtype=np.float32)
    matrix[:, 0] = 0.
    matrix[10:15, 0], matrix[10:15, 1] = -20., 0.
    matrix[60:65, 0], matrix[60:65, 2] = -20., 0.
    emitter = SimpleNamespace(vocab={"[PAD]": 0, "a": 1, "b": 2, "|": 3}, blank_id=0,
                              emissions=lambda _pcm: matrix)
    words = [SimpleNamespace(surah=1, ayah=i + 1, index=i, text=c) for i, c in enumerate(["a", "b"])]
    trace = {}
    rows = align_tiled.align_surah_tiled(np.zeros(32000, np.float32), 16000, words,
                                         bundle=align._OnnxBundle(emitter), trace=trace,
                                         check_onsets=False)
    assert [a for a, _, _ in rows] == [1, 2]
    assert rows[0][1] < rows[1][1]
    assert trace["align_lag_ms"] == 0


def test_legacy_environment_cannot_restore_a_label_fitted_shift(monkeypatch):
    monkeypatch.setenv("SURAH_CUTTER_ALIGN_LAG_MS", "200")
    assert align.ALIGN_LAG_MS == model_store.lag_ms() == 0
    model = align._OnnxBundle(SimpleNamespace(vocab={"a": 0}, blank_id=0))
    assert model.lag_ms == 0


def test_poisoned_reference_statistics_cannot_supply_a_timing_probability():
    check = dict(ayah=2, onset_ms=2500, conf=.9, first_token_conf=.9, reliable=True, reason="ctc")
    trace = {"ayahs": [check], "reference_times": {2: 2500},
             "timing_calibration": {"probability": 1., "calibration_n": 100000}}
    evidence = onset_evidence(check, 2500, .9, trace=trace)
    assert evidence["first_sound"] == .9
    assert evidence["timing_accuracy"]["probability"] is None
    assert evidence["timing_accuracy"]["calibration"] == "unavailable"
    assert "calibration_n" not in evidence["timing_accuracy"]


def test_recheck_does_not_suppress_a_better_match_with_a_label_fitted_margin(monkeypatch):
    words = [Word(1, 1, "first", 0), Word(1, 2, "second", 1)]
    bundle = SimpleNamespace(text_kind="phonetic", blank_id=0, vocab={"[PAD]": 0, "a": 1})
    monkeypatch.setattr(align, "_emissions", lambda *args: np.tile([-99., 0.], (500, 1)))
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["a", "b"])
    monkeypatch.setattr(phonetics, "onset_similarity", lambda decoded, expected: .50 if expected == "a" else .55)
    marks = {1: 1000, 2: 5000}
    rows = recheck.recheck_surah(np.zeros(160000), 16000, words, marks, bundle=bundle)
    assert rows[0]["verdict"] == "shift"
    assert rows[1]["verdict"] == "ok"
    assert marks == {1: 1000, 2: 5000}


def test_evaluation_helpers_and_artifacts_are_outside_the_application():
    root = Path(__file__).resolve().parents[1]
    for name in ["references", "reference_prior", "timing_probability"]:
        assert importlib.util.find_spec("surah_cutter.data." + name) is None
    assert not (root / "surah_cutter/data/timing_calibration.json").exists()
    for path in (root / "surah_cutter").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tools.research"), path
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("tools.research") for alias in node.names), path
    for path in [root / "pyproject.toml", root / "surah_cutter.spec"]:
        assert '"data/references"' not in path.read_text()
    assert all(not any("benchmark" in key for key in item) for item in model_store.catalog())
    for name in ["app.js", "models.js", "index.html", "i18n.js"]:
        content = (root / "surah_cutter/webgui/static" / name).read_text()
        assert "reference-cut" not in content
        assert "timingAccuracyLimit" not in content
        assert "эталон" not in content.lower()
