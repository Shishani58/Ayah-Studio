import sqlite3

from surah_cutter.data.project_state import complete_suras, discover_audio, surah_from_name


def test_discovery_uses_same_priority_for_named_and_numeric_files(tmp_path):
    for name in ("001.wav", "1.mp3", "001_alt.mp3", "002.flac", "115.mp3", "3.txt"):
        (tmp_path / name).touch()
    found = discover_audio(tmp_path)
    assert list(found) == [1, 2]
    assert found[1].name == "001_alt.mp3"
    assert surah_from_name("003_recitation.m4a") == 3
    assert surah_from_name("115.mp3") is None


def test_partial_surah_is_not_complete(tmp_path):
    db = tmp_path / "timings.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE timings (sura INTEGER, ayah INTEGER, time INTEGER)")
        conn.execute("INSERT INTO timings VALUES (1, 1, 0)")
    assert complete_suras(db, {1: 2}) == set()
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO timings VALUES (1, 2, 1000)")
    assert complete_suras(db, {1: 2}) == {1}


def test_corrupt_ayah_never_makes_surah_complete_or_crashes(tmp_path):
    db = tmp_path / "timings.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE timings (sura, ayah, time)")
        conn.executemany("INSERT INTO timings VALUES (?, ?, ?)", [
            (1, 1, 0), (1, 2.5, 1000), (2, None, 0),
        ])
    assert complete_suras(db, {1: 2, 2: 1}) == set()
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM timings WHERE sura=1 AND ayah=2.5")
        conn.execute("INSERT INTO timings VALUES (1, 2, 1000)")
    assert complete_suras(db, {1: 2, 2: 1}) == {1}
