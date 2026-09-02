from types import SimpleNamespace

import numpy as np

from surah_cutter.core import align, align_tiled
from surah_cutter.data import transliteration


def test_ctc_backtrace_estimate_routes_al_baqarah_to_tiles():
    samples = 7_000 * align.EXPECTED_SR
    targets = 47_266

    estimated = align._estimated_ctc_backtrace_bytes(
        samples,
        align.EXPECTED_SR,
        targets,
    )

    assert estimated > 240 * 1024**3
    assert align._full_ctc_requires_tiling(
        samples,
        align.EXPECTED_SR,
        targets,
        max_backtrace_bytes=1024**3,
    )
    assert not align._full_ctc_requires_tiling(
        10 * align.EXPECTED_SR,
        align.EXPECTED_SR,
        100,
        max_backtrace_bytes=1024**3,
    )


def test_align_surah_routes_oversized_exact_ctc_before_emissions(monkeypatch):
    emitter = SimpleNamespace(vocab={"a": 1, "|": 2}, blank_id=0)
    bundle = align._OnnxBundle(emitter)
    words = [SimpleNamespace(surah=2, ayah=1)]
    pcm = np.zeros(align.EXPECTED_SR, dtype=np.float32)

    monkeypatch.setattr(transliteration, "surah_word_tr", lambda _surah: ["a"])
    monkeypatch.setattr(align, "_basmala_trs", lambda _transliteration: [])
    monkeypatch.setattr(align, "_full_ctc_requires_tiling", lambda *_args: True)
    monkeypatch.setattr(
        align,
        "_emissions",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unbounded emissions")),
    )

    received: dict[str, object] = {}

    def fake_tiled(_pcm, _sr, _words, *, surah, bundle):
        received.update(surah=surah, bundle=bundle)
        return [(1, 123, 0.9)]

    monkeypatch.setattr(align_tiled, "align_surah_tiled", fake_tiled)

    rows = align.align_surah(
        pcm,
        align.EXPECTED_SR,
        words,
        surah=2,
        model=bundle,
    )

    assert rows == [(1, 123, 0.9)]
    assert received == {"surah": 2, "bundle": bundle}
