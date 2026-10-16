"""Canonical ayah progress, independent of saved marks and diagnostic text."""
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import align, align_fast, align_tiled, decode, pipeline, word_align
from surah_cutter.data import quran_text
from surah_cutter.data.project_db import ProjectDB
from surah_cutter.webgui import server
from surah_cutter.webgui.state import DB_CODE, DB_VERSION


def events(job, kind):
    return [e for e in job.events.snapshot() if e and e["type"] == kind]


def test_batch_progress_weights_canonical_ayahs_and_reports_current_before_work(tmp_path, monkeypatch):
    path = tmp_path / "progress.db"
    job = server._GenJob(path, {113: tmp_path / "113.mp3", 114: tmp_path / "114.mp3"})

    def process(audio, surah, **kwargs):
        for ayah in range(1, quran_text.ayah_count(surah) + 1):
            kwargs["ayah_progress"](ayah)
        # Log text must never be mistaken for structured progress.
        kwargs["log_event"]("Аят 999: проверка")
        return [(surah, a, a * 1000) for a in range(1, quran_text.ayah_count(surah) + 1)]

    monkeypatch.setattr(pipeline, "process_surah", process)
    job._run([113, 114], "fast")
    assert events(job, "start") == [{"type": "start", "total": 2, "total_ayahs": 11}]
    progress = events(job, "ayah_progress")
    assert [e["ayahs_done"] for e in progress] == list(range(11))
    assert [(e["surah"], e["ayah_count"]) for e in events(job, "sura_start")] == [
        (113, 5), (114, 6),
    ]
    assert [(e["ayahs_done"], e["total_ayahs"]) for e in events(job, "sura")] == [
        (5, 11), (11, 11),
    ]
    assert progress[-1]["ayah"] == 6


def test_cancel_progress_never_reports_completion_or_writes_partial_sura(tmp_path, monkeypatch):
    path = tmp_path / "cancel.db"
    job = server._GenJob(path, {114: tmp_path / "114.mp3"})

    def process(audio, surah, **kwargs):
        kwargs["ayah_progress"](1)
        kwargs["ayah_progress"](2)
        job.request_cancel()
        kwargs["ayah_progress"](3)
        pytest.fail("Cancelled callback must interrupt the algorithm")

    monkeypatch.setattr(pipeline, "process_surah", process)
    job._run([114], "full")
    assert [e["ayahs_done"] for e in events(job, "ayah_progress")] == [0, 1]
    assert events(job, "cancelled") == [{"type": "cancelled", "completed": 0}]
    assert not events(job, "sura")
    with ProjectDB.open(str(path), DB_CODE, DB_VERSION) as db:
        assert db.get_timings() == []


def test_word_progress_counts_ayahs_once_even_with_boundary_pass_and_word_failure(tmp_path, monkeypatch):
    path = tmp_path / "words.db"
    job = server._GenJob(path, {114: tmp_path / "114.mp3"})

    def process(audio, surah, **kwargs):
        assert "ayah_progress" not in kwargs
        return [(surah, a, a * 1000) for a in range(1, 7)]

    def word_rows(pcm, sr, words):
        if words[0].ayah == 2:
            raise ValueError("Неуверенное начало")
        return [(w.index, i * 50, .9) for i, w in enumerate(words)]

    monkeypatch.setattr(pipeline, "process_surah", process)
    monkeypatch.setattr(decode, "decode", lambda _: np.zeros(8 * decode.DEFAULT_SR))
    monkeypatch.setattr(word_align, "align_ayah_words", word_rows)
    job._run([114], "words")
    assert [e["ayahs_done"] for e in events(job, "ayah_progress")] == list(range(6))
    result = events(job, "sura")[0]
    assert result["ayahs_done"] == result["total_ayahs"] == 6
    assert result["word_failures"] == 1


@pytest.mark.parametrize("mode, module, function", [
    ("fast", align_fast, "align_surah_fast"),
    ("full", align, "align_surah"),
    ("tiled", align_tiled, "align_surah_tiled"),
])
def test_pipeline_forwards_progress_without_changing_rows(mode, module, function, monkeypatch):
    monkeypatch.setattr(align, "default_bundle", lambda: object())
    reported = []
    monkeypatch.setattr(decode, "decode", lambda _: np.zeros(decode.DEFAULT_SR))

    def aligned(pcm, sr, words, **kwargs):
        kwargs["ayah_progress"](1)
        kwargs["ayah_progress"](2)
        return [(1, 0, .9), (2, 100, .8)]

    monkeypatch.setattr(module, function, aligned)
    rows = getattr(pipeline, f"_{mode}_align")("audio.mp3", 114, [], ayah_progress=reported.append)
    assert rows == [(114, 1, 0), (114, 2, 100)]
    assert reported == [1, 2]


def test_full_onsets_report_once_per_ayah_and_preserve_results():
    words = [SimpleNamespace(ayah=a) for a in [1, 1, 2]]
    args = (words, [(0, 1), (1, 2), (2, 3)], [(5, 8), (10, 15), (20, 25)],
            np.full(30, -.1), 20.)
    baseline = align._check_ayah_onsets(*args, pauses=[])
    reported = []
    result = align._check_ayah_onsets(*args, pauses=[], ayah_progress=reported.append)
    assert result == baseline
    assert reported == [1, 2]
