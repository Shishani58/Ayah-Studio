"""Regression contracts for missing evidence, bounded memory and runtime waits."""

import itertools
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import _ctc, align, decode, emissions, model_runtime, recheck
from surah_cutter.data import transliteration
from surah_cutter.data.quran_text import Word


@pytest.mark.parametrize("marks", [{1: 15000, 2: 18000}, {1: 0, 2: 1000}, {1: -1, 2: 10000}])
def test_recheck_never_confirms_blank_or_outside_audio(marks, monkeypatch):
    words = [Word(1, 1, "first", 0), Word(1, 2, "second", 1)]
    bundle = SimpleNamespace(text_kind="phonetic", blank_id=0, vocab={"<pad>": 0, "a": 1})
    monkeypatch.setattr(align, "_emissions", lambda *args: np.tile([0.0, -99.0], (500, 1)))
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["abc", "def"])
    result = recheck.recheck_surah(np.zeros(160000, np.float32), 16000, words,
                                  marks, bundle=bundle)
    assert {row["verdict"] for row in result} == {"insufficient_evidence"}
    assert recheck.summarize(result) == {
        "ok": 0, "shift": 0, "no_time": 0, "insufficient_evidence": 2,
        "checked": 0, "suspect": 0, "unchecked": 2,
    }


def test_recheck_keeps_independent_relative_word_evidence(monkeypatch):
    words = [Word(1, 1, "first", 0), Word(1, 2, "second", 1)]
    bundle = SimpleNamespace(text_kind="phonetic", blank_id=0,
                             vocab={"<pad>": 0, "a": 1, "b": 2})
    matrix = np.full((500, 3), -99.0)
    matrix[:250, 1] = 0.0
    matrix[250:, 2] = 0.0
    monkeypatch.setattr(align, "_emissions", lambda *args: matrix)
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["a", "b"])
    result = recheck.recheck_surah(np.zeros(160000), 16000, words,
                                  {1: 0, 2: 6000}, bundle=bundle)
    assert [row["verdict"] for row in result] == ["ok", "ok"]


def test_recheck_never_confirms_mixed_word_separators(monkeypatch):
    bundle = SimpleNamespace(text_kind="phonetic", blank_id=3,
                             vocab={" ": 0, "|": 1, "a": 2, "[PAD]": 3})
    matrix = np.full((200, 4), -99.0)
    matrix[np.arange(200), np.arange(200) % 2] = 0.0
    monkeypatch.setattr(align, "_emissions", lambda *args: matrix)
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["a"])
    result = recheck.recheck_surah(np.zeros(160000), 16000, [Word(1, 1, "word", 0)],
                                  {1: 1000}, bundle=bundle)
    assert result[0]["verdict"] == "insufficient_evidence"
    assert recheck.summarize(result)["checked"] == 0


@pytest.mark.parametrize("targets", [[1, 2], [1, 1], [2, 1, 2]])
def test_ctc_matches_exhaustive_legal_paths(targets):
    rng = np.random.default_rng(10 + len(targets))
    for _ in range(5):
        matrix = np.log(rng.dirichlet(np.ones(3), size=6))
        path, scores = _ctc.forced_align(matrix[None], np.array([targets]))
        valid = []
        for candidate in itertools.product(range(3), repeat=6):
            collapsed = [v for i, v in enumerate(candidate)
                         if v != 0 and (i == 0 or v != candidate[i - 1])]
            if collapsed == targets:
                valid.append(sum(matrix[i, token] for i, token in enumerate(candidate)))
        assert scores.sum() == pytest.approx(max(valid))
        np.testing.assert_array_equal(scores[0], matrix[np.arange(6), path[0]])
        assert [s.label for s in _ctc.aligned_segments(path, targets)] == targets


def test_ctc_backtrace_is_one_byte_per_frame_state(monkeypatch):
    allocated = []
    zeros = np.zeros

    def observed(shape, *args, **kwargs):
        value = zeros(shape, *args, **kwargs)
        if shape == (50, 5):
            allocated.append(value.nbytes)
        return value

    monkeypatch.setattr(np, "zeros", observed)
    _ctc.forced_align(np.log(np.full((1, 50, 3), 1 / 3)), np.array([[1, 2]]))
    assert allocated == [50 * 5]


def test_emissions_stages_only_the_current_batch(monkeypatch):
    stacked = []
    stack = np.stack

    def observed(arrays, *args, **kwargs):
        stacked.append(len(arrays))
        return stack(arrays, *args, **kwargs)

    class Session:
        def run(self, names, inputs):
            batch = inputs["input_values"]
            assert len(batch) == 1
            return [np.zeros((len(batch), batch.shape[1] // 320 - 1, 3), np.float32)]

    monkeypatch.setattr(np, "stack", observed)
    emissions.generate_emissions(Session(), np.ones(20 * 16000, np.float32),
                                 window_s=1, context_s=1)
    assert stacked == [1] * 20


def test_exact_duration_cache_invalidates_on_file_change(tmp_path, monkeypatch):
    audio = tmp_path / "001.wav"
    audio.write_bytes(b"one")
    calls = []

    def decoded(path, sr):
        calls.append(path)
        return np.zeros((1 if audio.read_bytes() == b"one" else 2) * sr)

    monkeypatch.setattr(decode, "decode", decoded)
    assert decode.duration_ms(str(audio)) == decode.duration_ms(str(audio)) == 1000
    assert len(calls) == 1
    audio.write_bytes(b"two longer")
    assert decode.duration_ms(str(audio)) == 2000
    assert len(calls) == 2


@pytest.mark.parametrize("writing", [False, True])
def test_runtime_deadline_kills_real_stalled_pipe(writing, monkeypatch):
    child = subprocess.Popen([sys.executable, "-u", "-c",
                              "import time; print('ready'); time.sleep(60)"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    emitter = model_runtime.RuntimeEmitter.__new__(model_runtime.RuntimeEmitter)
    emitter.process, emitter._lock = child, threading.RLock()
    try:
        assert child.stdout.readline().strip() == "ready"
        monkeypatch.setattr(model_runtime, "RPC_TIMEOUT_S", .15)
        with pytest.raises(ValueError, match="время ожидания"):
            if writing:
                emitter.request("probe", payload="a" * 1024 * 1024)
            else:
                emitter._receive()
        assert child.wait(timeout=2) != 0
    finally:
        emitter.close()


def test_oracle_source_names_actual_weights_and_revision():
    config = SimpleNamespace(_name_or_path="owner/oracle", _commit_hash="a" * 40)
    bundle = align._Bundle(SimpleNamespace(config=config), SimpleNamespace(), {"a": 1})
    assert align.bundle_source(bundle) == f"torch-ctc:owner/oracle:revision={'a' * 40}:lag={align.ALIGN_LAG_MS}"


@pytest.mark.parametrize("backend", ["onnx", "torch", "mms-fa", "mms-arabic",
                                     "muaalem", "fastconformer-arabic"])
def test_tiled_result_retains_actual_emitter_backend(backend, monkeypatch):
    from surah_cutter.core import align_tiled
    from surah_cutter.data.alignment_result import result_source

    vocab = {"[PAD]": 0, "a": 1, "|": 2}
    if backend == "torch":
        bundle = align._Bundle(SimpleNamespace(), SimpleNamespace(), vocab)
    else:
        emitter = SimpleNamespace(vocab=vocab, blank_id=0)
        if backend != "onnx":
            # RuntimeEmitter is wrapped by _OnnxBundle too; its identity must
            # survive the bridge without claiming an ONNX inference backend.
            emitter.model_id = backend
            emitter.text_kind = "phonetic"
        bundle = align._OnnxBundle(emitter)
    matrix = np.full((50, 3), -99.0, dtype=np.float32)
    matrix[:, 0] = 0.0
    matrix[10:20, 0], matrix[10:20, 1] = -99.0, 0.0
    monkeypatch.setattr(align, "_emissions", lambda *args: matrix)
    monkeypatch.setattr(align, "bundle_source", lambda _: f"test:{backend}:revision=immutable")
    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _: ["a"])
    trace = {}
    result = align_tiled.align_surah_tiled(
        np.zeros(16000, np.float32), 16000, [Word(1, 1, "word", 0)],
        bundle=bundle, trace=trace, check_onsets=False,
    )
    assert len(result) == 1
    assert trace["backend"] == backend
    assert f":backend={backend}:" in result_source(trace, "tiled")
