"""Compare finite CTC onset pre-rolls; never alter production acceptance."""
import json
from pathlib import Path

from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data.quran_text import surah_words
from tools.bench_fast_models import dispatch_spy, verify_dispatch


def main():
    release_runtimes()
    model_store.ACTIVE_ID = "quran-phonetics"
    align._ONNX_CACHE.clear()
    bundle = align.default_bundle()
    adapter = CTCAdapter(bundle)
    pcm = decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/078.mp3")
    reads = LocalEvidenceReads(pcm, 16000, adapter)
    output = []
    expected = adapter.represent(surah_words(78))[0]
    for start, end in ((5520, 7820), (5520, 9020), (5520, 10520),
                       (5720, 7820), (5720, 9020), (5720, 10520), (6320, 10520)):
        with dispatch_spy(bundle, pcm) as calls:
            obs = reads.read(start, end, reason="diagnostic_finite_onset_preroll")
        verify_dispatch(calls, pcm, {"acoustic_reads": [reads.calls[-1]]})
        output.append({"window_ms": [start, end], "expected": expected, "text": obs.text,
                       "units": obs.units, "positions_ms": obs.positions_ms,
                       "unit_quality": obs.unit_quality, "read": reads.calls[-1],
                       "actual_dispatches": calls})
        print(start, end, obs.text, flush=True)
    (Path(__file__).with_name("ctc-onset-probe.json")).write_text(
        json.dumps({"results": output, "ledger": reads.diagnostics(),
                    "runtime_identity": adapter.runtime_identity}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
