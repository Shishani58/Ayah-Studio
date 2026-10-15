"""Experimental Muaalem phoneme adapter for bounded diagnostic reads only."""
import hashlib
import marshal
from surah_cutter.core.fast_adapters import CTCAdapter

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

