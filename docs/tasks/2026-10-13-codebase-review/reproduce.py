"""Reproduce review findings on temporary data; never touch user projects."""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from surah_cutter.data.project_db import ProjectDB  # noqa: E402
from surah_cutter.data.quran_text import Word  # noqa: E402
from surah_cutter.webgui import settings  # noqa: E402


def run(root):
    settings.SETTINGS_PATH = root / "settings.json"
    from surah_cutter.webgui import server

    results = {}
    db_path = root / "race.db"
    with ProjectDB.open(str(db_path), "review", 1) as db:
        db.write_timings([(1, 1, 1000), (1, 2, 2000)], "initial")
    checked, edited = threading.Event(), threading.Event()

    class PausedWriter(ProjectDB):
        def _status(self, sura, ayah):
            status = super()._status(sura, ayah)
            if ayah == 1:
                checked.set()
                assert edited.wait(5)
            return status

    def writer():
        with PausedWriter.open(str(db_path), "review", 1) as db:
            db.write_timings([(1, 1, 1200), (1, 2, 2200)], "generation")

    thread = threading.Thread(target=writer)
    thread.start()
    assert checked.wait(5)
    with ProjectDB.open(str(db_path), "review", 1) as db:
        db.set_time(1, 1, 1100)
    edited.set()
    thread.join(5)
    assert not thread.is_alive()
    with ProjectDB.open(str(db_path), "review", 1) as db:
        results["manual_edit_race"] = {
            "saved_manual": 1100, "after_generation": db.get_timings()[0][2],
            "status": db.get_review(1, 1)["status"],
        }
    assert results["manual_edit_race"]["after_generation"] == 1200

    with ProjectDB.open(str(root / "merge.db"), "review", 1) as db:
        db.write_timings([(1, 1, 1000), (1, 2, 2000), (1, 3, 3000)], "initial")
        db.set_time(1, 2, 2500)
        db.write_timings([(1, 1, 1000), (1, 2, 2000), (1, 3, 2400)], "generation")
        results["nonmonotonic_merge"] = db.get_timings()
        assert db.get_timings()[1][2] > db.get_timings()[2][2]

    job = server._GenJob(root / "sse.db", {})
    job.q.put({"type": "done", "completed": 1})
    job.q.put(None)
    stream_handler = next(r.endpoint for r in server.app.routes
                          if getattr(r, "path", None) == "/api/generate/stream")
    with patch.object(server, "_JOB", job):
        responses = [stream_handler(), stream_handler()]
        # Each response wraps its own generator, but both destructively read q.
        async def consume_first():
            return [event async for event in responses[0].body_iterator]
        first = asyncio.run(consume_first())
        results["sse_second_subscriber"] = {"first": first, "remaining": job.q.qsize()}
        assert job.q.empty()

    settings.save({"mode": "full", "theme": "light"}, strict=True)
    original_load = settings.load
    both_loaded = threading.Barrier(2)
    theme_saved = threading.Event()
    original_save = settings.save

    def concurrent_load():
        data = original_load()
        both_loaded.wait(timeout=5)
        return data

    def ordered_save(data, *, strict=False):
        if data["theme"] == "dark":
            original_save(data, strict=strict)
            theme_saved.set()
        else:
            assert theme_saved.wait(5)
            original_save(data, strict=strict)

    with patch.object(settings, "load", concurrent_load), patch.object(settings, "save", ordered_save):
        threads = [threading.Thread(target=settings.update, kwargs=fields) for fields in
                   ({"theme": "dark", "strict": True}, {"mode": "words", "strict": True})]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
            assert not t.is_alive()
    results["settings_lost_update"] = original_load()
    assert results["settings_lost_update"] == {"mode": "words", "theme": "light"}

    from surah_cutter.core import align, recheck
    from surah_cutter.data import transliteration

    words = [Word(1, 1, "first", 0), Word(1, 2, "second", 1)]
    bundle = SimpleNamespace(text_kind="phonetic", blank_id=0, vocab={"<pad>": 0, "a": 1})
    emissions = np.tile([0.0, -99.0], (500, 1))
    with patch.object(align, "_emissions", return_value=emissions), \
            patch.object(transliteration, "surah_word_tr", return_value=["abc", "def"]):
        results["recheck_outside_audio"] = recheck.recheck_surah(
            np.zeros(160000, dtype=np.float32), 16000, words,
            {1: 15000, 2: 18000}, bundle=bundle,
        )
    assert all(row["verdict"] == "ok" for row in results["recheck_outside_audio"])

    from surah_cutter.core import word_align
    from surah_cutter.data import quran_text

    results["transliteration_segmentation"] = []
    for surah, ayah in ((2, 181), (8, 6), (13, 37)):
        words = [w for w in quran_text.surah_words(surah) if w.ayah == ayah]
        trs = transliteration.surah_word_tr(surah)
        record = {"surah": surah, "ayah": ayah, "canonical_words": len(words),
                  "cached_words": len(transliteration._cached_map(
                      transliteration.CACHE_PATH)[f"{surah}:{ayah}"]),
                  "last_word": words[-1].text, "last_target": trs[words[-1].index]}
        vocab = {char: i + 1 for i, char in enumerate(sorted(set("".join(trs))))}
        phonetic = align._OnnxBundle(SimpleNamespace(vocab=vocab, blank_id=0))
        with patch.object(align, "_get_onnx_bundle", return_value=phonetic):
            try:
                word_align.align_ayah_words(np.zeros(16000, np.float32), 16000, words)
            except ValueError as exc:
                record["word_alignment_error"] = str(exc)
        assert record.get("word_alignment_error") == "Не все слова представлены в словаре модели."
        results["transliteration_segmentation"].append(record)

    from surah_cutter import cli
    from surah_cutter.core import pipeline

    audio = root / "audio"
    audio.mkdir()
    (audio / "114.mp3").write_bytes(b"unused synthetic audio")
    cli_db = root / "cli.db"

    def computed(*args, trace, **kwargs):
        trace["model_source"] = "onnx-ctc:test-model:revision=immutable:lag=200"
        trace["mode"] = "full"
        return [(114, 1, 1000)]

    with patch.object(pipeline, "process_surah", computed), redirect_stdout(io.StringIO()):
        assert cli.main([str(audio), "--output", str(cli_db), "--mode", "full"]) == 0
    with ProjectDB.open(str(cli_db), "review", 1) as db:
        results["cli_provenance"] = db.get_review(114, 1)["source"]
    assert results["cli_provenance"] == "auto"
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ayah-code-review-") as directory:
        run(Path(directory))
