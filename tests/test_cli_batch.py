"""Batch generation continues after a single surah merge conflict."""

from surah_cutter import cli
from surah_cutter.core import pipeline
from surah_cutter.data.project_db import ProjectDB


def test_cli_continues_after_writer_conflict(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "001.mp3").touch()
    (audio / "002.mp3").touch()
    output = tmp_path / "project.db"
    with ProjectDB.open(str(output), "test", 1) as db:
        db.write_timings([(1, 1, 1000), (1, 2, 2000), (1, 3, 3000)], "initial")
        db.set_time(1, 2, 2500)

    called = []

    def process(_path, surah, *, trace, **_kwargs):
        called.append(surah)
        trace["model_source"] = "reference:test"
        if surah == 1:
            return [(1, 1, 1100), (1, 2, 2000), (1, 3, 2400)]
        return [(2, 1, 100)]

    monkeypatch.setattr(pipeline, "process_surah", process)
    assert cli.main([str(audio), "-o", str(output)]) == 1
    assert called == [1, 2]
    assert "FAILED sura 1" in capsys.readouterr().err
    with ProjectDB.open(str(output), "test", 1) as db:
        assert db.get_timings() == [
            (1, 1, 1000), (1, 2, 2500), (1, 3, 3000), (2, 1, 100),
        ]
