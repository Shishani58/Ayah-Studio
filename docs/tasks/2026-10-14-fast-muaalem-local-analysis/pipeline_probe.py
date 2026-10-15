"""Isolated experimental factory; normal GUI/CLI model support stays unchanged."""
import hashlib
import json
import marshal
import os
from pathlib import Path
import sys

from adapter import MuaalemLocal
from surah_cutter.core import fast_executor, model_store
from tools.bench_fast_models import main

task = Path(__file__).parent
original_factory = fast_executor.make_adapter
original_support = model_store.supports_fast


def experimental_factory(bundle):
    if getattr(bundle.emitter, "model_id", "") != "muaalem":
        return original_factory(bundle)
    adapter = MuaalemLocal(bundle)
    original_represent = adapter.represent

    def represented(words):
        try:
            return original_represent(words)
        finally:
            (task / f"mapping-{words[0].surah:03d}.json").write_text(
                json.dumps(adapter.mapping, ensure_ascii=False, indent=2) + "\n")

    adapter.represent = represented
    adapter.version += "-audit-" + hashlib.sha256(marshal.dumps(represented.__code__)).hexdigest()
    return adapter


experimental_factory.__module__ = fast_executor.__name__
fast_executor.make_adapter = experimental_factory
model_store.supports_fast = lambda selected=None: (selected or model_store.ACTIVE_ID) == "muaalem" or original_support(selected)
sys.argv[1:] = ["--models", "muaalem", "--surahs", *os.environ.get("MUA_PROBE_SURAHS", "1 32 112").split(),
                "--audio-dir", "/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri",
                "--reference", str(task.parents[2] / "data/references/abu_bakr_ash_shatri.db"),
                "--output", os.environ.get("MUA_PROBE_OUTPUT", str(task / "pipeline.json"))]
main()
