"""Duration-neutral identity experiment; raw madd phones remain in provenance."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import runpy
import sys

task = Path(__file__).parent.resolve()
parent = task.parent / "2026-10-14-fast-muaalem-local-analysis"
sys.path.insert(0, str(parent))
import adapter  # noqa: E402

Original = adapter.MuaalemLocal
LONG = {"ا", "ۦ", "ۥ", "ـ"}


def collapse(units, quality=None, positions=None):
    output, scores, times, owners = [], [], [], []
    for index, unit in enumerate(units):
        if output and output[-1] == unit and unit in LONG:
            owners[-1].append(index)
            if quality is not None:
                scores[-1] = min(scores[-1], quality[index])
        else:
            output.append(unit)
            owners.append([index])
            if quality is not None:
                scores.append(quality[index])
            if positions is not None:
                times.append(positions[index])
    return tuple(output), tuple(scores), tuple(times), owners


class DurationNeutral(Original):
    def __init__(self, selected):
        super().__init__(selected)
        self.version += "-duration-neutral-" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.runtime_identity["experimental_identity"] = {
            "long_vowel_runs": "one lexical unit; all raw scores retained as min",
            "posterior_prefix_enabled": False, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def represent(self, words):
        raw = super().represent(words)
        groups = {}
        for word, units in zip(words, raw):
            groups.setdefault(word.ayah, []).extend(units)
        self.tempo_groups = {ayah: tuple(units) for ayah, units in groups.items()}
        return tuple(collapse(units)[0] for units in raw)

    def observe(self, pcm, sr, *, total_samples):
        raw = super().observe(pcm, sr, total_samples=total_samples)
        units, quality, positions, owners = collapse(raw.units, raw.unit_quality, raw.positions_ms)
        hypotheses = []
        for candidate in raw.hypotheses:
            c_units, c_quality, c_positions, c_owners = collapse(
                candidate["units"], candidate["unit_quality"], candidate["positions_ms"])
            hypotheses.append({**candidate, "units": c_units, "unit_quality": c_quality,
                               "positions_ms": c_positions, "raw_unit_indices": c_owners})
        attribution = tuple({"normalized_unit": unit, "raw_unit_indices": ids,
                             "raw_units": [raw.units[i] for i in ids],
                             "raw_quality": [raw.unit_quality[i] for i in ids]}
                            for unit, ids in zip(units, owners))
        # The raw CTC alphabet scores duration-bearing repeats. Its marginal
        # probabilities cannot be reinterpreted as probabilities of this quotient.
        # Only independent decoder evidence participates in this experiment.
        return replace(raw, units=units, unit_quality=quality, positions_ms=positions,
                       hypotheses=tuple(hypotheses), attribution=attribution, posterior=None)


adapter.MuaalemLocal = DurationNeutral
os.environ.setdefault("MUA_PROBE_SURAHS", "1 55")
os.environ.setdefault("MUA_PROBE_OUTPUT", str(task / "duration-neutral-local.json"))
os.environ.setdefault("MUA_AUDIT_DIR", str(task / "duration-neutral-audit"))
runpy.run_path(str(parent / "identity_floor_probe.py"), run_name="__main__")
