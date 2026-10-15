"""Acoustic operations for experimental FAST; no surah execution or persistence."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import unicodedata

import numpy as np

from surah_cutter.core import align


@dataclass(frozen=True)
class Observation:
    units: tuple[str, ...]
    positions_ms: tuple[float, ...] | None
    quality: float
    truncated: bool = False
    text: str = ""
    unit_quality: tuple[float, ...] = ()
    frames: int = 0
    attribution: tuple[dict, ...] = ()
    hypotheses: tuple[dict, ...] = ()
    null_log_probability: float | None = None
    posterior_sha256: str | None = None
    posterior: np.ndarray | None = field(default=None, repr=False, compare=False)
    stride_ms: float = 0.


def arabic_units(text):
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({
        "ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ـ": "",
    }))
    text = "".join(c if unicodedata.category(c).startswith("L") or c.isspace()
                   else "" if unicodedata.category(c).startswith("M") else " " for c in text)
    return tuple(text.split())


def attributed_asr_units(result):
    """Map selected UTF-8 BPE bytes to normalized words, without target text.

    A token may contribute to two words; a character may span two tokens. All
    contributing selected tokens must pass the quality gate. A non-reversible
    decode is explicitly unsupported rather than assigned a window average.
    """
    tokens = result.get("selected_tokens", ())
    units = arabic_units(result["text"])
    payload, owners = bytearray(), []
    probabilities = []
    try:
        for i, token in enumerate(tokens):
            raw = bytes(token["bytes"])
            probability = math.exp(float(token["log_probability"]))
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("invalid selected probability")
            probabilities.append(probability)
            payload.extend(raw)
            owners.extend([i] * len(raw))
        decoded = payload.decode("utf-8", errors="strict")
        if arabic_units(decoded) != units:
            raise ValueError("non-reversible selected decode")
        word_owners, raw_words, current, raw_word, byte_offset = [], [], set(), [], 0
        for char in decoded:
            count = len(char.encode("utf-8"))
            contributing = owners[byte_offset:byte_offset + count]
            byte_offset += count
            normalized = unicodedata.normalize("NFKC", char)
            for c in normalized:
                if c.isspace() or not (unicodedata.category(c).startswith(("L", "M")) or c == "ـ"):
                    if current:
                        word_owners.append(sorted(current))
                        raw_words.append("".join(raw_word))
                        current, raw_word = set(), []
                elif c != "ـ":
                    current.update(contributing)
                    raw_word.append(c)
        if current:
            word_owners.append(sorted(current))
            raw_words.append("".join(raw_word))
        if len(word_owners) != len(units) or any(not ids for ids in word_owners):
            raise ValueError("ambiguous word attribution")
        attribution = tuple({"word": word, "raw_word": raw, "selected_indices": ids,
                             "token_ids": [tokens[i]["id"] for i in ids],
                             "probabilities": [probabilities[i] for i in ids]}
                            for word, raw, ids in zip(units, raw_words, word_owners))
        return units, tuple(min(probabilities[i] for i in ids) for ids in word_owners), attribution
    except (ValueError, KeyError, TypeError, OverflowError):
        return units, (0.,) * len(units), ({"unsupported": "ambiguous_token_word_attribution"},)


class CTCAdapter:
    version = "ctc-independent-words-v4-hyphen-separators"
    metric = "ctc_greedy_probability"
    timing = True

    def __init__(self, bundle):
        self.bundle = bundle
        self.lag_ms = getattr(bundle, "lag_ms", align.ALIGN_LAG_MS)
        self.model_id = "quran-phonetics"
        self.runtime_identity = bundle.emitter.runtime_identity
        if not self.runtime_identity.get("weights_sha256"):
            raise ValueError("FAST requires an identifiable installed CTC model")
        self.revision = self.runtime_identity["weights_sha256"]
        self.blank_id = bundle.blank_id
        self.vocab = dict(bundle.vocab)
        self.inv = {v: k for k, v in bundle.vocab.items()}

    def represent(self, words):
        from surah_cutter.data import transliteration
        rendered = transliteration.surah_word_tr(words[0].surah)
        if len(words) != len(rendered):
            raise ValueError("canonical_word_map_mismatch")
        # Hyphens mark transliteration segmentation, not spoken phonemes.
        # Apostrophes may encode hamza and remain; repeated letters remain.
        return tuple(tuple(c for c in text if c not in ("-", "|", " ")) for text in rendered)

    def observe(self, pcm, sr, *, total_samples):
        # pcm is exactly the guard-authorized local slice. No known text is given.
        emissions = align._emissions(self.bundle, pcm, np)
        # Optional segmentation predictions must not be mistaken for spoken
        # phonemes. Fold only these non-spoken tokens into blank, consistently
        # for greedy, beam and marginal comparisons of the same local frames.
        emissions = np.array(emissions, copy=True)
        separators = [token for char, token in self.vocab.items() if char in (" ", "-", "|")]
        emissions[:, self.blank_id] = np.logaddexp.reduce(emissions[:, [self.blank_id, *separators]], axis=-1)
        emissions[:, separators] = -np.inf
        stride = align._stride_ms(len(pcm), sr, len(emissions))
        ids = emissions.argmax(-1)
        tokens, starts, scores = [], [], []
        previous = -1
        for frame, token in enumerate(ids.tolist()):
            if token != previous and token != self.blank_id:
                tokens.append(self.inv.get(token, ""))
                starts.append(max(0., frame * stride - self.lag_ms))
                scores.append(float(np.exp(emissions[frame, token])))
            previous = token
        units, positions, quality = [], [], []
        for token, start, probability in zip(tokens, starts, scores):
            for char in token:
                if char == "|":
                    continue
                # CTC frame collapse above is sufficient: repeated characters
                # separated by a blank, including a word seam, remain distinct.
                units.append(char)
                positions.append(start)
                quality.append(probability)
        from surah_cutter.core.fast_ctc import independent_beam
        hypotheses, null = independent_beam(emissions, self.blank_id, self.inv, stride, self.lag_ms)
        return Observation(tuple(units), tuple(positions),
                           float(np.mean(scores)) if scores else 0., text="".join(tokens),
                           unit_quality=tuple(quality), frames=len(emissions),
                           hypotheses=hypotheses, null_log_probability=null,
                           posterior=emissions, stride_ms=stride,
                           posterior_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(emissions))).hexdigest())



class TarteelAdapter:
    version = "tarteel-independent-words-v4-raw-alias-ownership"
    metric = "asr_token_probability"
    timing = False  # recognize_local currently supplies no word timestamps
    lag_ms = 0

    def __init__(self, bundle):
        self.emitter = bundle.emitter
        self.model_id = self.emitter.model_id
        if self.model_id not in ("whisper-tarteel-tiny", "whisper-tarteel-base"):
            raise ValueError("unsupported_local_adapter")
        self.runtime_identity = self.emitter.runtime_identity
        self.revision = hashlib.sha256(json.dumps(self.runtime_identity, sort_keys=True).encode()).hexdigest()

    def represent(self, words):
        from surah_cutter.core.model_runtime import inference_words
        from surah_cutter.core.model_worker import whisper_alignment_text
        return tuple(arabic_units(whisper_alignment_text(text))
                     for text in inference_words(words, self.model_id))

    def proof_word_ends(self, words, rendered):
        """Whole spoken letter names can prove their original canonical owner.

        Only Quran opening letters get internal proof boundaries. This does not
        split the canonical database word, nor allow arbitrary subword proof.
        """
        from surah_cutter.core.model_worker import whisper_alignment_text
        return tuple(tuple(range(1, len(units) + 1))
                     if whisper_alignment_text(word.text) != word.text and len(units) > 1
                     else (len(units),) for word, units in zip(words, rendered))

    def observe(self, pcm, sr, *, total_samples):
        # Worker independently checks locality against the original sample count.
        result = self.emitter.recognize_local(pcm, total_samples=total_samples)
        units, quality, attribution = attributed_asr_units(result)
        # Tarteel can spell opening letters as compact orthographic symbols
        # (طه / يس) instead of names. Expand the independently decoded symbol,
        # retaining every supporting token and its quality for all named units.
        from surah_cutter.core.model_worker import whisper_alignment_text
        expanded, expanded_quality, expanded_attribution = [], [], []
        if len(attribution) == len(units):
            for word, q, ownership in zip(units, quality, attribution):
                # Decide letter names before hamza/diacritics are discarded.
                names = arabic_units(whisper_alignment_text(ownership.get("raw_word", word)))
                for unit in names:
                    expanded.append(unit)
                    expanded_quality.append(q)
                    expanded_attribution.append({**ownership, **({"spoken_alias": word, "source_units": names}
                                                                 if names != (word,) else {})})
            units, quality, attribution = tuple(expanded), tuple(expanded_quality), tuple(expanded_attribution)
        return Observation(units, None, float(result["token_confidence"]),
                           bool(result["truncated"]), result["text"],
                           unit_quality=quality, attribution=attribution)


def make_adapter(bundle):
    if getattr(bundle.emitter, "model_id", "").startswith("whisper-tarteel-"):
        return TarteelAdapter(bundle)
    if getattr(bundle, "text_kind", "phonetic") != "phonetic":
        raise ValueError("unsupported_local_adapter")
    return CTCAdapter(bundle)


def representation_sha(words, rendered):
    payload = repr([(w.surah, w.ayah, w.text, units) for w, units in zip(words, rendered)])
    return hashlib.sha256(payload.encode()).hexdigest()


def loaded_implementation_sha():
    """Fingerprint the running Python code, never a later mutable source file.

    A version label alone is easy to forget during experiments. Code objects
    identify loaded acceptance, normalization, decoding and read accounting.
    Closure/config values are captured separately in policy/runtime snapshots.
    This conservative fingerprint is Python/path specific: moving a checkout
    can invalidate resume, which safely recomputes auto rather than reusing it.
    """
    import importlib
    import marshal
    import types
    from dataclasses import is_dataclass, asdict

    code = []
    seen = set()
    def configuration(value):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {"nonfinite": str(value)}
        if is_dataclass(value) and not isinstance(value, type):
            return {"dataclass": type(value).__module__ + "." + type(value).__qualname__,
                    "fields": configuration(asdict(value))}
        if isinstance(value, (tuple, list)):
            return {type(value).__name__: [configuration(v) for v in value]}
        if isinstance(value, dict):
            return {"dict": sorted([[configuration(k), configuration(v)] for k, v in value.items()],
                                   key=lambda pair: json.dumps(pair[0], sort_keys=True))}
        return {"type": type(value).__module__ + "." + type(value).__qualname__}
    def capture(name, symbol, operation):
        if id(operation) in seen:
            return
        seen.add(id(operation))
        defaults = json.dumps({"defaults": configuration(operation.__defaults__),
                               "kwdefaults": configuration(operation.__kwdefaults__)}, sort_keys=True)
        code.append((name, symbol, operation.__code__, defaults))
        # Instrumentation wrappers retain the actual operation in a closure.
        # Hash that code too, without serializing mutable PCM/logging objects.
        for cell_name, cell in zip(operation.__code__.co_freevars, operation.__closure__ or ()):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, types.MethodType):
                value = value.__func__
            if isinstance(value, types.FunctionType):
                capture(name, symbol + ".closure." + cell_name, value)
    for name in ("fast_executor", "fast_adapters", "fast_ctc", "fast_reads", "emissions", "model_runtime"):
        module = importlib.import_module("surah_cutter.core." + name)
        knobs = {key: value for key, value in vars(module).items()
                 if key.isupper() and isinstance(value, (str, bool, int, float))}
        code.append((name, "effective_module_knobs", json.dumps(configuration(knobs), sort_keys=True)))
        for symbol, value in sorted(vars(module).items()):
            if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
                capture(name, symbol, value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                for method, operation in sorted(vars(value).items()):
                    if isinstance(operation, (staticmethod, classmethod)):
                        operation = operation.__func__
                    elif isinstance(operation, property):
                        operation = operation.fget
                    if isinstance(operation, types.FunctionType):
                        capture(name, symbol + "." + method, operation)
    align_module = importlib.import_module("surah_cutter.core.align")
    for name in ("_emissions", "_stride_ms"):
        capture("align", name, getattr(align_module, name))
    return hashlib.sha256(marshal.dumps(tuple(code))).hexdigest()
