"""Observe the actual backend even when a CTC runtime exposes an ASR method."""
from types import SimpleNamespace

import numpy as np

from surah_cutter.core import align
from tools.bench_fast_models import dispatch_spy


class RuntimeLike:
    def __init__(self, model_id):
        self.model_id = model_id

    def recognize_local(self, pcm, **kwargs):
        return {"samples": len(pcm)}


def test_ctc_runtime_with_recognize_method_records_emissions(monkeypatch):
    bundle = SimpleNamespace(emitter=RuntimeLike("muaalem"))
    whole = np.zeros(32000, dtype=np.float32)
    local = whole[1000:9000]
    monkeypatch.setattr(align, "_emissions", lambda selected, pcm, module: pcm)
    with dispatch_spy(bundle, whole) as calls:
        align._emissions(bundle, local, np)
    assert len(calls) == 1 and calls[0]["samples"] == len(local)


def test_whisper_runtime_records_recognition():
    emitter = RuntimeLike("whisper-tarteel-base")
    whole = np.zeros(32000, dtype=np.float32)
    local = whole[1000:9000]
    with dispatch_spy(SimpleNamespace(emitter=emitter), whole) as calls:
        emitter.recognize_local(local, total_samples=len(whole))
    assert len(calls) == 1 and calls[0]["samples"] == len(local)
