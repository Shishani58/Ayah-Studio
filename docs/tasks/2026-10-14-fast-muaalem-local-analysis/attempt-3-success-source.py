"""Independent local Muaalem head; static targets are never passed to inference."""
import hashlib
import json
import marshal
from pathlib import Path

from surah_cutter.core import align, decode, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_ctc import prefix_scores
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

task = Path(__file__).parent
model_store.ACTIVE_ID = "muaalem"
bundle = align.default_bundle()


class MuaalemLocal(CTCAdapter):
    def __init__(self, selected):
        self.bundle = selected
        self.model_id = "muaalem"
        self.runtime_identity = selected.emitter.runtime_identity
        self.revision = self.runtime_identity.get("assets", {}).get("muaalem/model.fp32.onnx")
        if not self.revision:
            raise ValueError("Loaded Muaalem weights must have an immutable asset SHA")
        self.lag_ms = selected.lag_ms
        self.vocab = dict(selected.vocab)
        self.blank_id = selected.blank_id
        self.inv = {value: key for key, value in self.vocab.items()}
        self.version = "experimental-muaalem-ctc-local-v1-" + hashlib.sha256(
            marshal.dumps(type(self).__init__.__code__) + marshal.dumps(type(self).represent.__code__)).hexdigest()
        self.mapping = None

    def represent(self, words):
        targets, spans = self.bundle.emitter.build_targets(words, intro=False)
        self.mapping = {"targets": targets, "spans": spans,
                        "words": [[w.ayah, w.index, w.text] for w in words]}
        ends = [0] + [end for start, end in spans]
        if (len(spans) != len(words) or ends[-1] != len(targets)
                or any(start != previous or end <= start
                       for previous, (start, end) in zip(ends, spans))):
            raise ValueError("Muaalem word phones are not a nonoverlapping full partition")
        return tuple(tuple(self.inv[token] for token in targets[start:end]) for start, end in spans)


adapter = MuaalemLocal(bundle)
pcm = decode.decode("/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/001.mp3")
words = quran_text.surah_words(1)
rendered = adapter.represent(words)
target = next(units for word, units in zip(words, rendered) if word.ayah == 2)
ledger = LocalEvidenceReads(pcm, 16000, adapter)
records = []
with dispatch_spy(bundle, pcm) as actual:
    for left, right in ((6420, 7920), (6420, 8520), (6120, 9120)):
        observed = ledger.read(left, right, reason="cause_probe_missing_onset_1:2")
        if observed is None:
            raise RuntimeError("local probe denied")
        ids = [adapter.vocab[u] for u in target]
        score = prefix_scores(observed.posterior, [ids], adapter.blank_id)[0]
        matches = [i for i in range(len(observed.units))
                   if observed.units[i:i + len(target)] == target]
        records.append({"window_ms": [left, right], "expected_first_word": target,
                        "greedy": observed.units, "unit_quality": observed.unit_quality,
                        "positions_ms": observed.positions_ms,
                        "first_entry_log_probability": score,
                        "greedy_occurrences": matches, "beam": observed.hypotheses})
        (task / "partial-results.json").write_text(json.dumps({
            "mapping": adapter.mapping, "model_runtime": adapter.runtime_identity,
            "local_reads": records, "ledger": ledger.diagnostics(),
            "actual_dispatches": actual, "complete": False,
        }, ensure_ascii=False, indent=2) + "\n")
trace = {"total_ms": len(pcm) / 16, **ledger.diagnostics()}
verify_dispatch(actual, pcm, trace)
report = {"model_runtime": adapter.runtime_identity, "adapter_version": adapter.version,
          "mapping": adapter.mapping, "rendered": rendered, "local_reads": records,
          "ledger": trace, "actual_dispatches": actual,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "scope": "independent phoneme output and static word mapping; not production FAST or corpus accuracy"}
(task / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(records, ensure_ascii=False, indent=1), flush=True)
release_runtimes()
