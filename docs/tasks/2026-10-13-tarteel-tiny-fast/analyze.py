"""Summarize completed local FAST evidence; never run acoustic inference."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    folder = Path(__file__).parent
    items = json.loads(folder.joinpath("comparison.json").read_text())
    assert len(items) == 111 and all("errors_ms" in i for i in items)
    for item in items:
        if item["model"] == "whisper-tarteel-tiny":
            assert item["source"].startswith("local-asr-pause:whisper-tarteel-tiny:")
            assert item["trace"]["emission_passes"] == 0
            assert item["trace"]["bounded_ctc_fallback"] is None
            assert not item["sequence_violations"]
    report = {"models": {}, "paired": {}}
    for model in dict.fromkeys(i["model"] for i in items):
        subset = [i for i in items if i["model"] == model]
        errors = [e for i in subset for e in i["errors_ms"]]
        assert len(subset) == 37 and len(errors) == 564
        confident = [e for i in subset for (_, _, c), e in zip(i["rows"], i["errors_ms"]) if c > 0]
        reads = [r for i in subset for r in i["trace"]["acoustic_reads"]]
        report["models"][model] = {
            "median_ms": float(np.median(errors)), "p95_ms": float(np.percentile(errors, 95)),
            "max_ms": max(errors), "over500": sum(e > 500 for e in errors),
            "confident": len(confident), "confident_over500": sum(e > 500 for e in confident),
            "confident_max_ms": max(confident, default=None),
            "review": sum(len(i["trace"]["review_required_ayahs"]) for i in subset),
            "search_seconds": sum(i["search_seconds"] for i in subset),
            "max_read_ms": max(r["window_ms"][1] - r["window_ms"][0] for r in reads),
            "max_surah_coverage": max(i["trace"]["coverage"] for i in subset),
            "corpus_coverage": sum(i["trace"]["covered_audio_ms"] for i in subset) / sum(i["duration_ms"] for i in subset),
            "input_ratio": sum(i["trace"]["audio_ms"] for i in subset) / sum(i["duration_ms"] for i in subset),
            "denied_reads": sum(len(i["trace"]["denied_acoustic_reads"]) for i in subset),
        }
    tiny = {(i["surah"], a): e for i in items if i["model"] == "whisper-tarteel-tiny"
            for (a, _, _), e in zip(i["rows"], i["errors_ms"])}
    for model in ("quran-phonetics", "whisper-tarteel-base"):
        other = {(i["surah"], a): e for i in items if i["model"] == model
                 for (a, _, _), e in zip(i["rows"], i["errors_ms"])}
        assert other.keys() == tiny.keys()
        report["paired"][model] = {
            "tiny_corrected_over500": sum(other[k] > 500 and tiny[k] <= 500 for k in tiny),
            "tiny_new_over500": sum(other[k] <= 500 and tiny[k] > 500 for k in tiny),
            "both_over500": sum(other[k] > 500 and tiny[k] > 500 for k in tiny),
        }
    provenance = json.loads(folder.joinpath("reproducibility.json").read_text())
    for path, expected in provenance["protected_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    prior = json.loads(folder.parent.joinpath("2026-10-13-tarteel-fast/comparison.summary.json").read_text())
    prior_items = json.loads(folder.parent.joinpath("2026-10-13-tarteel-fast/comparison.json").read_text())
    prior_rows = {(i["model"], i["surah"]): i["rows"] for i in prior_items}
    assert all(i["rows"] == prior_rows[i["model"], i["surah"]]
               for i in items if i["model"] != "whisper-tarteel-tiny")
    current = json.loads(folder.joinpath("comparison.summary.json").read_text())
    assert all(v["initial_pauses_sha256"] == prior["initial_pauses"][s]["initial_pauses_sha256"]
               for s, v in current["initial_pauses"].items())
    for model in ("quran-phonetics", "whisper-tarteel-base"):
        assert current["summary"][model]["median_ms"] == prior["summary"][model]["median_ms"]
        assert current["summary"][model]["over500"] == prior["summary"][model]["over500"]
    report["protected_hashes_unchanged"] = True
    report["prior_pause_hashes_identical"] = True
    report["prior_control_rows_identical"] = True
    folder.joinpath("analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
