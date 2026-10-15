"""Prototype canonical-word lower bound; phonetic variation is not owner identity."""
import hashlib
import json
import os
from pathlib import Path
import sys
import unicodedata

from adapter import MuaalemLocal
from surah_cutter.core import fast_executor, model_store
from tools.bench_fast_models import main

task = Path(__file__).parent
audit_folder = Path(os.environ.get("MUA_AUDIT_DIR", str(task)))
audit_folder.mkdir(parents=True, exist_ok=True)
original_plan = fast_executor.plan_phrase
original_noisy = fast_executor._noisy_opening
original_factory = fast_executor.make_adapter
original_support = model_store.supports_fast
registry = {}


def canonical_floor(ayah):
    if ayah not in registry["plans"]:
        groups = registry["identity"]
        plan = original_plan(groups, ayah, {a: list(range(1, len(words) + 1)) for a, words in groups.items()})
        count = len(plan["units"])
        # A declared spoken letter-name prefix may still identify one canonical
        # word. More than one canonical word requires its full final word.
        minimum = registry["ends"][ayah][count - 1] if count > 1 else 0
        registry["plans"][ayah] = (minimum, plan)
    return registry["plans"][ayah]


def identity_plan(groups, ayah, word_ends=None):
    planned = original_plan(groups, ayah, word_ends)
    if not registry or ayah not in registry["identity"]:
        return planned
    minimum, identity = canonical_floor(ayah)
    if len(planned["units"]) < minimum:
        planned["units"] = groups[ayah][:minimum]
        planned["collisions"] = [(a, i) for a, i in planned["competitors"]
                                 if groups[a][i:i + minimum] == planned["units"]]
    planned["ambiguous"] = bool(planned["collisions"]) or identity["ambiguous"]
    return planned


def identity_noisy(observation, groups, word_ends, ayah, threshold):
    proof, expand = original_noisy(observation, groups, word_ends, ayah, threshold)
    if proof and registry:
        minimum, identity = canonical_floor(ayah)
        if identity["ambiguous"] or len(proof["phrase"]) < minimum:
            return None, expand
    return proof, expand


def factory(bundle):
    if getattr(bundle.emitter, "model_id", "") != "muaalem":
        return original_factory(bundle)
    adapter = MuaalemLocal(bundle)
    represent = adapter.represent

    def represented(words):
        units = represent(words)
        identity, ends, counts = {}, {}, {}
        for word, phones in zip(words, units):
            identity.setdefault(word.ayah, []).append(unicodedata.normalize("NFKC", word.text))
            counts[word.ayah] = counts.get(word.ayah, 0) + len(phones)
            ends.setdefault(word.ayah, []).append(counts[word.ayah])
        registry.clear()
        registry.update(identity={a: tuple(text) for a, text in identity.items()}, ends=ends, plans={})
        (audit_folder / f"identity-mapping-{words[0].surah:03d}.json").write_text(json.dumps(
            {"canonical_words": registry["identity"], "canonical_phone_ends": ends,
             "phonetic_mapping": adapter.mapping}, ensure_ascii=False, indent=2) + "\n")
        return units

    adapter.represent = represented
    adapter.version += "-canonical-floor-" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return adapter


for operation in (identity_plan, identity_noisy, factory):
    operation.__module__ = fast_executor.__name__
fast_executor.plan_phrase = identity_plan
fast_executor._noisy_opening = identity_noisy
fast_executor.make_adapter = factory
model_store.supports_fast = lambda selected=None: (selected or model_store.ACTIVE_ID) == "muaalem" or original_support(selected)
sys.argv[1:] = ["--models", "muaalem", "--surahs", *os.environ.get("MUA_PROBE_SURAHS", "55").split(),
                "--audio-dir", "/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri",
                "--reference", str(task.parents[2] / "data/references/abu_bakr_ash_shatri.db"),
                "--output", os.environ.get("MUA_PROBE_OUTPUT", str(task / "identity-floor.json"))]
main()
