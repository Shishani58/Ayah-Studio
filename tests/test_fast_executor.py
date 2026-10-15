"""Behavioral contract for one FAST executor (real canonical sequence, fake acoustics)."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core.fast_adapters import Observation
from surah_cutter.core.fast_executor import (Boundary, Policy, align_surah_sequential,
                                            plan_phrase, validate_boundaries)
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.data import quran_text


class Fake:
    version = "fake-v1"
    metric = "asr_token_probability"
    model_id = "fake"
    revision = "one"
    lag_ms = 0
    timing = True

    def __init__(self, responder, words=None):
        self.responder = responder
        self.words = words or quran_text.surah_words(112)
        self.rendered = tuple((f"a{w.ayah}w{i}",) for i, w in enumerate(self.words))
        self.calls = []

    def represent(self, words):
        return self.rendered

    def phrase(self, ayah):
        return tuple(u for w, units in zip(self.words, self.rendered) if w.ayah == ayah for u in units)[:2]

    def observe(self, pcm, sr, *, total_samples):
        self.calls.append((round(float(pcm[0])), len(pcm)))
        assert len(pcm) < total_samples and len(pcm) <= sr * 6
        return self.responder(self, round(float(pcm[0])), len(pcm) / sr * 1000)


def observation(units, *, quality=.99, positions=True):
    return Observation(tuple(units), tuple(i * 100. for i in range(len(units))) if positions else None,
                       quality, unit_quality=(quality,) * len(units))


def pause(ms, width=200):
    return SimpleNamespace(accepted=True, pause=SimpleNamespace(
        start_ms=float(ms), end_ms=float(ms + width), center_ms=float(ms + width / 2)))


def run(fake, pauses=None, duration=40000, policy=Policy(sparse_anchors=False)):
    pcm = np.repeat(np.arange(duration, dtype=np.float32), 16)
    trace = {}
    from unittest.mock import patch
    with patch("surah_cutter.core.fast_executor.first_pause_hypotheses", return_value=[]):
        rows = align_surah_sequential(pcm, 16000, fake.words, adapter=fake, trace=trace,
                                      pause_decisions=pauses or [], policy=policy)
    return rows, trace


def test_distinct_pauses_sequential_no_pre_read():
    def respond(fake, start, duration):
        ayah = {1000: 1, 6000: 2, 12000: 3, 20000: 4}.get(start)
        return observation(fake.phrase(ayah) if ayah else ())
    fake = Fake(respond)
    rows, trace = run(fake, [pause(p) for p in (1000, 6000, 12000, 20000)])
    assert [r[1] for r in rows] == [1100, 6100, 12100, 20100]
    assert all(r[2] > 0 for r in rows)
    events = trace["transitions"]
    assert next(i for i, e in enumerate(events) if e["state"] == "DETECT_PAUSES") < next(i for i, e in enumerate(events) if e["state"] == "READ_LOCAL")
    first_advance = next(i for i, e in enumerate(events) if e["state"] == "ADVANCE")
    assert all(e.get("ayah") in (None, 1) for e in events[:first_advance])
    assert [e["ayah"] for e in events if e["state"] == "ADVANCE"] == [1, 2, 3, 4]


def test_missing_current_never_confirms_future_or_reads_tail():
    fake = Fake(lambda f, start, duration: observation(f.phrase(2)))
    rows, trace = run(fake, [pause(1000), pause(10000), pause(20000)])
    assert all(conf == 0 for _, _, conf in rows)
    assert not any(e["state"] == "ADVANCE" for e in trace["transitions"])
    assert trace["boundaries"][1]["review_reason"] == "blocked_by_unresolved_ayah"
    assert len(fake.calls) <= 3  # first onset, future pause, bounded block


def test_ambiguous_prefix_finishes_and_blocks_later_repeat():
    fake = Fake(lambda f, start, duration: observation(f.phrase(1)[:1]))
    fake.rendered = (("ان",),) + fake.rendered[1:]
    _, trace = run(fake, [pause(p) for p in range(1000, 30000, 1000)])
    assert len(fake.calls) <= 3
    assert trace["transitions"][-1]["state"] == "RESULT_READY"
    assert all(b["confidence"] == 0 for b in trace["boundaries"])


def test_whole_multitoken_word_and_more_than_six_words():
    groups = {1: tuple("abcdefghXYZ"), 2: tuple("abcdefghUVW")}
    ends = {1: [3, 6, 9, 11], 2: [3, 6, 9, 11]}
    phrase = plan_phrase(groups, 1, ends)
    assert len(phrase["units"]) == 9
    assert not phrase["ambiguous"]
    common = tuple(str(i) for i in range(7))
    assert len(plan_phrase({1: common + ("x",), 2: common + ("y",)}, 1)["units"]) == 8


def test_internal_repeat_is_ambiguous():
    assert plan_phrase({1: ("a", "b"), 2: ("c", "a", "b")}, 1)["ambiguous"]


def test_two_letter_complete_word_allows_expansion():
    fake = Fake(lambda f, start, duration: observation(f.phrase(1)[:1] if duration < 2000 else f.phrase(1)))
    fake.rendered = (("ان",),) + fake.rendered[1:]
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][2] > 0
    assert any(e["state"] == "EXPAND_LOCAL" for e in trace["transitions"])


def test_first_onset_no_fixed_six_seconds():
    fake = Fake(lambda f, start, duration: observation(f.phrase(1)))
    rows, _ = run(fake)
    assert rows[0][1] == 3100 and rows[0][2] > 0


def test_no_timing_cannot_confirm_first_onset():
    fake = Fake(lambda f, start, duration: observation(f.phrase(1), positions=False))
    fake.timing = False
    rows, trace = run(fake)
    assert rows[0][2] == 0
    assert trace["boundaries"][0]["review_reason"] == "unsupported_first_onset_timing"


def test_long_pause_never_dispatches_more_than_six_seconds():
    fake = Fake(lambda f, start, duration: observation(()))
    _, trace = run(fake, [pause(1000, 6000)])
    assert all(r["window_ms"][1] - r["window_ms"][0] <= 6000 for r in trace["acoustic_reads"])
    assert trace["boundaries"][0]["review_reason"] == "no_local_evidence"


def test_short_file_has_no_whole_file_exception():
    fake = Fake(lambda f, start, duration: observation(f.phrase(1)))
    _, trace = run(fake, duration=500)
    assert not fake.calls
    assert trace["denied_acoustic_reads"]


def test_unit_quality_cannot_be_compensated_by_high_mean():
    def respond(f, start, duration):
        units = f.phrase(1)
        return Observation(units, (0., 100.), .99, unit_quality=(.01, .99))
    rows, _ = run(Fake(respond), [pause(1000)])
    assert rows[0][2] == 0


def test_cache_eviction_preserves_started_budget():
    fake = Fake(lambda f, start, duration: observation(("a",)))
    reads = LocalEvidenceReads(np.zeros(160000, dtype=np.float32), 16000, fake, cache_limit_bytes=1)
    reads.read(1000, 2000, reason="test")
    reads.read(1000, 2000, reason="test")
    assert len(fake.calls) == 2 and reads.audio_ms == 2000 and reads.covered_ms == 1000
    assert reads.cache_bytes == 0 and reads.cache_evictions == 2


def test_adapter_failure_is_not_retried_and_ledger_survives():
    def fail(*args):
        raise RuntimeError("model failed")
    fake = Fake(fail)
    trace = {}
    with pytest.raises(RuntimeError, match="model failed"):
        align_surah_sequential(np.ones(160000, dtype=np.float32), 16000, fake.words,
                               adapter=fake, pause_decisions=[], trace=trace)
    assert len(fake.calls) == 1
    assert trace["acoustic_reads"][0]["outcome"] == "failed"
    assert trace["audio_ms"] > 0 and trace["transitions"][-1]["state"] == "FAILED"


@pytest.mark.parametrize("surah", [1, 9])
def test_canonical_exceptions_no_rewrite(surah):
    fake = Fake(lambda *args: observation(()), quran_text.surah_words(surah))
    rows, _ = run(fake)
    assert [r[0] for r in rows] == list(range(1, quran_text.ayah_count(surah) + 1))


def test_validator_rejects_false_proof_and_collision():
    obs = observation(("a", "b"))
    read = {"outcome": "completed", "window_ms": [1000., 3000.], "observed": asdict(obs)}
    proof = {"phrase": ("a", "b"), "complete": True, "ambiguous": False,
             "competitors_checked": True, "offset": 0, "pause_ms": [1000., 1200.]}
    valid = Boundary(1, 1100, "pause", .99, read_id=0, proof=proof)
    validate_boundaries([valid], [1], 10000, [read], groups={1: ("a", "b")}, threshold=.6)
    with pytest.raises(ValueError, match="non_distinguishing"):
        validate_boundaries([valid], [1], 10000, [read], groups={1: ("a", "b"), 2: ("a", "b")})
    with pytest.raises(ValueError, match="timeline"):
        validate_boundaries([valid, Boundary(2, 1100, "estimate", 0, "test")], [1, 2], 10000, [read])


def test_canonical_text_cannot_be_replaced():
    fake = Fake(lambda *args: observation(()))
    with pytest.raises(ValueError, match="canonical_text_mismatch"):
        align_surah_sequential(np.ones(160000, dtype=np.float32), 16000, fake.words[:-1],
                               adapter=fake, pause_decisions=[])


def test_cancellation_after_dispatch_and_before_cache():
    import threading
    from surah_cutter.core.cancellation import GenerationCancelled, cancellation_scope
    cancel = threading.Event()
    fake = Fake(lambda *args: observation(("a",)))
    reads = LocalEvidenceReads(np.zeros(160000, dtype=np.float32), 16000, fake)
    with cancellation_scope(cancel):
        reads.read(1000, 2000, reason="test")
        cancel.set()
        with pytest.raises(GenerationCancelled):
            reads.read(1000, 2000, reason="cached")
    assert len(fake.calls) == 1
    cancel.clear()
    def respond(*args):
        cancel.set()
        return observation(("a",))
    fake.responder = respond
    with cancellation_scope(cancel), pytest.raises(GenerationCancelled):
        reads.read(3000, 4000, reason="cancel-in-model")
    assert reads.calls[-1]["outcome"] == "cancelled"
    assert reads.audio_ms == 2000 and len(reads.observations) == 1


def test_anchored_short_wasl_requires_previous_tail_and_real_position():
    fake = Fake(lambda *args: observation(()))
    groups = {a: tuple(u for w, us in zip(fake.words, fake.rendered) if w.ayah == a for u in us)
              for a in range(1, 5)}
    def respond(f, start, duration):
        if start == 1000:
            return observation(f.phrase(1))
        if duration > 5000:
            return observation(groups[1][-2:] + f.phrase(2))
        return observation(())
    fake.responder = respond
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][2] > 0 and rows[1][2] > 0
    assert trace["boundaries"][1]["evidence_kind"] == "wasl"
    assert rows[2][2] == 0  # no advancing through the next unresolved ayah


def test_future_block_is_published_atomically_only_after_current_proof():
    fake = Fake(lambda *args: observation(()))
    full_first = tuple(u for w, us in zip(fake.words, fake.rendered) if w.ayah == 1 for u in us)
    def respond(f, start, duration):
        if start == 5000:
            return observation(f.phrase(2))
        if duration == 6000:
            return observation(full_first + f.phrase(2))
        return observation(())
    fake.responder = respond
    rows, trace = run(fake, [pause(5000)])
    assert rows[0][2] > 0 and rows[1][2] > 0
    confirms = [e for e in trace["transitions"] if e["state"] == "CONFIRMED"]
    assert confirms[0]["block_ayahs"] == [1, 2]
    assert trace["boundaries"][0]["read_id"] > trace["boundaries"][1]["read_id"]


def test_tempo_region_is_a_selection_hint_not_proof():
    def respond(f, start, duration):
        return observation(f.phrase({1000: 1, 6000: 2}.get(start, 4)))
    fake = Fake(respond)
    rows, trace = run(fake, [pause(p) for p in (1000, 6000, 6050, 6100, 6200, 6250, 6300, 6400, 6450, 6500, 12000)])
    assert any("forecast_ms" in e for e in trace["transitions"])
    assert rows[2][2] == 0


@pytest.mark.parametrize("experimental", [False, True])
def test_fast_pipeline_never_falls_back(monkeypatch, experimental):
    from surah_cutter.core import align, align_fast, fast_executor, pipeline, decode, model_store
    monkeypatch.setattr(model_store, "ACTIVE_ID", "quran-phonetics")
    monkeypatch.setattr(decode, "decode", lambda _: np.ones(160000, dtype=np.float32))
    monkeypatch.setattr(align, "default_bundle", lambda: object())
    calls = []
    def experimental(*args, **kwargs):
        calls.append("new")
        raise RuntimeError("new failed")
    def forbidden(*args, **kwargs):
        pytest.fail("legacy fallback")
    monkeypatch.setattr(fast_executor, "align_surah_sequential", experimental)
    monkeypatch.setattr(align_fast, "align_surah_fast_legacy", forbidden, raising=False)
    monkeypatch.setattr(pipeline, "_full_align", forbidden)
    with pytest.raises(RuntimeError, match="new failed"):
        pipeline.process_surah("test.mp3", 112, mode="fast", experimental_fast=experimental)
    assert calls == ["new"]


def test_algorithm_adapters_have_no_legacy_or_persistence_dependencies():
    import ast
    import inspect
    from surah_cutter.core import fast_adapters, fast_executor
    for module in (fast_adapters, fast_executor):
        tree = ast.parse(inspect.getsource(module))
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any("align_fast" in name or "project_db" in name or "alignment_result" in name for name in imports)
    assert not any("align_surah" in name for name in dir(fast_adapters))


def test_server_import_does_not_load_user_model_or_write_settings(monkeypatch):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", """
from surah_cutter.core import model_store
from surah_cutter.webgui import settings
model_store.ACTIVE_ID = 'import-sentinel'
def forbidden(*args, **kwargs):
    raise AssertionError('settings accessed at import')
settings.load = forbidden
settings.update = forbidden
import surah_cutter.webgui.server
assert model_store.ACTIVE_ID == 'import-sentinel'
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_phrase_uses_one_whole_distinct_word_when_sufficient():
    plan = plan_phrase({1: ("unique", "second"), 2: ("other", "second")}, 1)
    assert plan["units"] == ("unique",)


@pytest.mark.parametrize("model", ["whisper-tarteel-tiny", "whisper-tarteel-base"])
def test_actual_tarteel_representation_distinguishes_quran_cases(model):
    from surah_cutter.core.fast_adapters import TarteelAdapter
    words = quran_text.surah_words(2)
    adapter = TarteelAdapter(SimpleNamespace(emitter=SimpleNamespace(model_id=model, runtime_identity={"model_id": model})))
    rendered = adapter.represent(words)
    groups, ends = {}, {}
    for word, units in zip(words, rendered):
        groups.setdefault(word.ayah, []).extend(units)
        ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
    groups = {a: tuple(us) for a, us in groups.items()}
    for ayah in (11, 13, 68, 69, 70):
        plan = plan_phrase(groups, ayah, ends)
        assert not plan["ambiguous"]
        assert len(plan["units"]) in ends[ayah]
    assert len(plan_phrase(groups, 68, ends)["units"]) > 6


def test_changed_active_model_during_read_does_not_change_snapshot(monkeypatch):
    from surah_cutter.core import model_store
    fake = Fake(lambda *args: observation(()))
    def respond(*args):
        monkeypatch.setattr(model_store, "ACTIVE_ID", "whisper-large-v3")
        return observation(())
    fake.responder = respond
    _, trace = run(fake)
    assert trace["run_snapshot"]["model"] == "fake"
    assert trace["model_source"].startswith("fake:")
    assert trace["align_lag_ms"] == 0


def test_validator_rejects_mismatched_rows_and_impossible_duration():
    from surah_cutter.core.fast_executor import _finish_estimates
    with pytest.raises(ValueError, match="unrepresentable_timeline"):
        _finish_estimates([], [1, 2, 3], {1: ("a",), 2: ("b",), 3: ("c",)}, 2., "test")
    with pytest.raises(ValueError, match="canonical_ayah_set"):
        validate_boundaries([Boundary(2, 0, "estimate", 0, "test")], [1], 1000, [])


def test_explicit_settings_restore_runs_at_asgi_startup_only(monkeypatch):
    import asyncio
    from surah_cutter.webgui import server
    calls = []
    monkeypatch.setattr(server, "_initialize_app_settings", lambda: calls.append("restore"))
    app = server.create_app(restore_settings=False)
    assert calls == []
    async def start():
        async with app.router.lifespan_context(app):
            assert calls == ["restore"]
    asyncio.run(start())


def test_estimates_reserve_duration_for_last_confirmed_ayah():
    from surah_cutter.core.fast_executor import _finish_estimates
    rows = _finish_estimates([Boundary(1, 1000, "first_onset", .9)], [1, 2, 3],
                             {1: ("longpreviousayah" * 10,), 2: ("small",), 3: ("small",)}, 10000, "test")
    assert 8000 < rows[1].start_ms < rows[2].start_ms < 10000
    assert rows[0].start_ms == 1000 and rows[1].confidence == rows[2].confidence == 0


def test_wasl_tail_is_whole_word_and_distinguishes_repeated_ending():
    from surah_cutter.core.fast_executor import plan_tail
    groups = {1: tuple("firstun"), 2: tuple("otherun")}
    ends = {1: [5, 7], 2: [5, 7]}
    tail = plan_tail(groups, 1, ends)
    assert tail["units"] == groups[1] and not tail["ambiguous"]
    assert plan_tail({1: ("same",), 2: ("same",)}, 1, {1: [1], 2: [1]})["ambiguous"]


def test_ctc_word_seams_preserve_repeated_characters(tmp_path, monkeypatch):
    from surah_cutter.core import align
    from surah_cutter.core.fast_adapters import CTCAdapter
    from surah_cutter.data import transliteration
    model = tmp_path / "model.onnx"
    model.write_bytes(b"identity-only")
    text = "innaalladhina"
    vocab = {c: i + 1 for i, c in enumerate(dict.fromkeys(text))}
    vocab["<blank>"] = 0
    bundle = SimpleNamespace(emitter=SimpleNamespace(session=SimpleNamespace(_model_path=str(model)),
        runtime_identity={"weights_sha256": "test-only"}),
                             vocab=vocab, blank_id=0, lag_ms=0)
    adapter = CTCAdapter(bundle)
    ids = [i for c in text for i in (vocab[c], 0)]
    emissions = np.full((len(ids), len(vocab)), -100., dtype=np.float32)
    emissions[np.arange(len(ids)), ids] = 0.
    monkeypatch.setattr(align, "_emissions", lambda *args: emissions)
    heard = adapter.observe(np.zeros(16000, dtype=np.float32), 16000, total_samples=32000)
    assert "".join(heard.units) == text
    words = quran_text.surah_words(2)
    rendered = adapter.represent(words)
    original = transliteration.surah_word_tr(2)
    assert "".join(u for w, us in zip(words, rendered) if w.ayah == 6 for u in us) == "".join(
        "".join(c for c in tr if c not in ("-", "|", " ")) for w, tr in zip(words, original) if w.ayah == 6)
    assert heard.frames == len(ids)


def test_fast_pipeline_to_transaction_keeps_manual_and_identity(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from surah_cutter.core import align, candidates, decode, fast_executor, model_store, pipeline
    from surah_cutter.data.alignment_result import write_alignment_result
    from surah_cutter.data.alignment_identity import fast_resume_suras
    from surah_cutter.data.project_db import ProjectDB
    fake = Fake(lambda f, start, duration: observation(f.phrase(
        {1000: 1, 6000: 2, 12000: 3, 20000: 4}[start])) if start in (1000, 6000, 12000, 20000) else observation(()))
    monkeypatch.setattr(model_store, "ACTIVE_ID", "quran-phonetics")
    monkeypatch.setattr(align, "default_bundle", lambda: object())
    monkeypatch.setattr(fast_executor, "make_adapter", lambda _: fake)
    monkeypatch.setattr(decode, "decode", lambda _: np.repeat(np.arange(40000, dtype=np.float32), 16))
    monkeypatch.setattr(candidates, "pause_pipeline_decisions", lambda *args, **kwargs: [pause(ms) for ms in (1000, 6000, 12000, 20000)])
    path = tmp_path / "project.db"
    with ProjectDB.open(str(path), "test", 1) as db:
        db.write_timings([(112, 1, 1160)], "old", .4)
        db.set_status(112, 1, "verified")
        original_review = db.get_review(112, 1)
        trace = {}
        rows = pipeline.process_surah("fake.mp3", 112, mode="fast",
                                      project_db=db, trace=trace)
        assert rows[0] == (112, 1, 1160) and trace["generated_rows"][0] == (112, 1, 1100)
        result = write_alignment_result(db, rows, trace=trace, requested_mode="fast", db_path=path)
        assert result.outcome == "saved"
        assert db.get_review(112, 1) == original_review
        assert "fast_run_key=" + trace["fast_run_key"] in db.get_review(112, 2)["source"]
        payload = json.loads(Path(result.log_paths["run"]).read_text())
        assert payload["trace"]["boundaries"][0]["start_ms"] == 1100
        assert payload["persisted_rows"][0] == [112, 1, 1160]
    assert fast_resume_suras(path, {112: trace["fast_run_key"]}) == {112}
    assert fast_resume_suras(path, {112: "f" * 64}) == set()


def test_nonadjacent_future_cannot_be_ignored_in_favor_of_later_repetition():
    def respond(f, start, duration):
        return observation(f.phrase(3) if start == 1000 else f.phrase(1) if start == 10000 else ())
    rows, trace = run(Fake(respond), [pause(1000), pause(10000)])
    assert all(conf == 0 for _, _, conf in rows)
    assert trace["boundaries"][0]["review_reason"] == "future_outside_bounded_block"
    assert not any(e["state"] == "ADVANCE" for e in trace["transitions"])


def test_first_pause_recall_refines_vad_and_preserves_offset(monkeypatch, tmp_path):
    from surah_cutter.core import candidates
    from surah_cutter.core.fast_executor import first_pause_hypotheses
    clip = np.full(16000 * 2, .2, dtype=np.float32)
    clip[8000:14400] = .0001
    pcm = np.concatenate((np.zeros(192000, dtype=np.float32), clip))
    received = []
    def vad(local, sr, **kwargs):
        received.append(len(local))
        return [candidates.Pause(400, 1000, 600)]
    monkeypatch.setattr(candidates, "detect_silero_pauses", vad)
    from surah_cutter.core import fast_executor
    model = tmp_path / "silero.onnx"
    model.touch()
    monkeypatch.setattr(fast_executor, "_silero_model_path", lambda: model)
    result = first_pause_hypotheses(pcm, 16000, 12000, 15000)
    assert received == [32000]
    assert [(p.start_ms, p.end_ms) for p in result] == [(12500, 12900)]


def test_supplemental_first_pause_has_validated_owner(monkeypatch):
    import surah_cutter.core.fast_executor as executor
    fake = Fake(lambda f, start, duration: observation(f.phrase(1) if start == 1000 else ()))
    monkeypatch.setattr(executor, "first_pause_hypotheses", lambda *a: [pause(1000).pause])
    trace = {}
    pcm = np.repeat(np.arange(40000, dtype=np.float32), 16)
    rows = executor.align_surah_sequential(pcm, 16000, fake.words, adapter=fake,
                                           pause_decisions=[], trace=trace)
    assert rows[0][1:] == (1100, .99)
    assert trace["phase1_pauses"] == []
    hyp = trace["first_onset_hypotheses"]["supplemental_pauses"][0]
    assert hyp["pause_id"] == trace["boundaries"][0]["proof"]["pause_id"]
    assert hyp["center_ms"] == rows[0][1]
    bad = [Boundary(**b) for b in trace["boundaries"]]
    with pytest.raises(ValueError, match="unknown_pause_owner"):
        validate_boundaries(bad, [1, 2, 3, 4], 40000, trace["acoustic_reads"], pauses=[])


def test_validator_rejects_weak_wasl_tail_even_with_strong_current_phrase():
    observed = asdict(Observation(("previous", "target"), (0., 100.), .99,
                                  unit_quality=(.01, .99)))
    proof = {"phrase": ("target",), "previous_tail": ("previous",), "offset": 1,
             "complete": True, "ambiguous": False, "competitors_checked": True}
    boundary = Boundary(2, 1000, "wasl", .99, read_id=0, proof=proof)
    reads = [{"outcome": "completed", "observed": observed, "window_ms": [900, 1200]}]
    with pytest.raises(ValueError, match="weak_wasl_tail"):
        validate_boundaries([boundary], [2], 5000, reads, threshold=.6,
                            groups={1: ("previous",), 2: ("target",)}, word_ends={1: [1], 2: [1]})


def test_adjacent_block_with_weak_previous_tail_stops_for_review():
    fake = Fake(lambda *args: observation(()))
    full_second = tuple(u for w, us in zip(fake.words, fake.rendered) if w.ayah == 2 for u in us)
    tail = tuple(u for w, us in zip(fake.words, fake.rendered) if w.ayah == 1 for u in us)[-1:]
    def respond(f, start, duration):
        if start == 1000:
            return observation(f.phrase(1))
        if start == 8000:
            return observation(f.phrase(3))
        if duration == 6000:
            units = tail + full_second + f.phrase(3)
            return Observation(units, tuple(i * 100. for i in range(len(units))), .99,
                               unit_quality=(.01,) + (.99,) * (len(units)-1))
        return observation(())
    fake.responder = respond
    rows, trace = run(fake, [pause(1000), pause(8000)])
    assert rows[0][2] > 0 and all(conf == 0 for _, _, conf in rows[1:])
    assert trace["boundaries"][1]["review_reason"] in ("unresolved_adjacent_block", "unresolved_local_block")
    assert trace["transitions"][-1]["state"] == "RESULT_READY"


@pytest.mark.parametrize("experimental", [False, True])
def test_fast_pipeline_captures_bundle_before_decode(monkeypatch, experimental):
    from surah_cutter.core import align, decode, fast_executor, model_store, pipeline
    monkeypatch.setattr(model_store, "ACTIVE_ID", "quran-phonetics")
    monkeypatch.setattr(align, "default_bundle", lambda: SimpleNamespace(model=model_store.ACTIVE_ID))
    def decode_and_change_selection(path):
        model_store.ACTIVE_ID = "whisper-tarteel-base"
        return np.zeros(160000, dtype=np.float32)
    def execute(pcm, sr, words, **kwargs):
        assert kwargs["bundle"].model == "quran-phonetics"
        return [(1, 100, .9)]
    monkeypatch.setattr(decode, "decode", decode_and_change_selection)
    monkeypatch.setattr(fast_executor, "align_surah_sequential", execute)
    assert pipeline._fast_align("test.mp3", 112, quran_text.surah_words(112), experimental=experimental) == [(112, 1, 100)]


def test_vad_identity_changes_with_silero_and_optional_ten_assets(tmp_path, monkeypatch):
    from importlib import util
    from surah_cutter.core import candidates, fast_executor
    silero = tmp_path / "vad.onnx"
    silero.write_bytes(b"weights-one")
    ten = tmp_path / "ten_vad"
    ten.mkdir()
    (ten / "__init__.py").write_text("# test package")
    native = ten / "libten.dylib"
    native.write_bytes(b"embedded-weights-one")
    monkeypatch.setattr(candidates, "detect_silero_pauses", lambda *a, model_path=silero, **kw: [])
    original = util.find_spec
    monkeypatch.setattr(util, "find_spec", lambda name: SimpleNamespace(
        submodule_search_locations=[str(ten)], origin=str(ten / "__init__.py")) if name == "ten_vad" else original(name))
    before = fast_executor.dsp_model_identity()
    assert before["silero"]["provider"] == "CPUExecutionProvider"
    assert before["onnxruntime_version"]
    silero.write_bytes(b"weights-two")
    after_silero = fast_executor.dsp_model_identity()
    assert before != after_silero
    native.write_bytes(b"embedded-weights-two")
    after_ten = fast_executor.dsp_model_identity()
    assert after_ten != after_silero
    # Same audio, policy and model acquire a different key when VAD evidence changes.
    fake = Fake(lambda *a: observation(()))
    monkeypatch.setattr(fast_executor, "dsp_model_identity", lambda: before)
    _, trace1 = run(fake)
    monkeypatch.setattr(fast_executor, "dsp_model_identity", lambda: after_ten)
    _, trace2 = run(fake)
    assert trace1["fast_run_key"] != trace2["fast_run_key"]


def test_missing_optional_silero_leaves_first_hypotheses_empty(tmp_path, monkeypatch):
    from surah_cutter.core import candidates, fast_executor
    missing = tmp_path / "missing.onnx"
    def never(*args, model_path=missing, **kwargs):
        pytest.fail("missing optional VAD must not be invoked")
    monkeypatch.setattr(candidates, "detect_silero_pauses", never)
    assert fast_executor.first_pause_hypotheses(np.zeros(32000, dtype=np.float32), 16000, 0, 15000) == []


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_discriminating_quality_is_not_hidden_by_neighbor(invalid):
    observed = asdict(Observation(("one", "two"), None, .99, unit_quality=(.99, invalid)))
    proof = {"phrase": ("one", "two"), "offset": 0, "complete": True,
             "ambiguous": False, "competitors_checked": True, "pause_ms": [900, 1100]}
    boundary = Boundary(1, 1000, "pause", .9, read_id=0, proof=proof)
    reads = [{"outcome": "completed", "observed": observed, "window_ms": [900, 3000]}]
    with pytest.raises(ValueError, match="weak_discriminating_unit"):
        validate_boundaries([boundary], [1], 5000, reads, threshold=.6)


def test_supplied_pause_pool_is_part_of_resume_identity():
    fake = Fake(lambda *a: observation(()))
    _, first = run(fake, [pause(1000)])
    _, second = run(fake, [pause(1200)])
    assert first["fast_run_key"] != second["fast_run_key"]
