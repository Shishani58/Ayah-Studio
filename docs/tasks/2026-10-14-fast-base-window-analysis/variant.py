"""One experimental policy; no production file or setting changes."""
from dataclasses import replace
import sys

from surah_cutter.core import fast_executor
from tools.bench_fast_models import main

widths = tuple(map(int, sys.argv.pop(1).split(",")))
original = fast_executor.align_surah_sequential.__kwdefaults__["policy"]
fast_executor.align_surah_sequential.__kwdefaults__["policy"] = replace(original, speech_ms=widths)
print("experimental_speech_ms", widths, flush=True)
main()
