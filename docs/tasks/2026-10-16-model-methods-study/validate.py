"""Artifact, coverage, cache and read-only input integrity audit after inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

from common import SOURCE, TASK, SNAPSHOT, corpus, write_json
from surah_cutter.data import quran_text


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    protected = json.loads((TASK / "protected-before.json").read_text())
    changed = [name for name,sha in protected.items() if file_sha(name) != sha]
    assert not changed, changed
    changed_source = [name for name,sha in SNAPSHOT["files"].items() if file_sha(SOURCE/name) != sha]
    assert not changed_source, changed_source
    folders = {"strong20":220,"fixed30":48,"strong6":48,"strong20-fp32":12,"strong20-localnorm":24}
    failures, coverage, pcm_identity, used_keys = [], {}, {}, set()
    weights = {"int8":protected[str(Path(SNAPSHOT["root"])/"models/wav2vec2-quran-phonetics.onnx")],
               "fp32":protected[str(Path(SNAPSHOT["root"])/"models/wav2vec2-quran-phonetics.fp32.onnx")]}
    for folder, expected in folders.items():
        records = [json.loads(p.read_text()) for p in (TASK/folder).glob("*.json")]
        records = [r for r in records if "case" in r]
        assert len(records) == expected,(folder,len(records),expected)
        coverage[folder] = len(records)
        for r in records:
            case = r["case"]
            assert r["weight_sha"] == weights["fp32" if folder.endswith("fp32") else "int8"]
            previous = pcm_identity.setdefault(case["id"],r["PCM_sha"])
            assert previous == r["PCM_sha"],case["id"]
            assert r["reference_prior"] == "disabled-text-only" and r["auxiliary_models"] == "disabled"
            assert r["trace"].get("align_lag_ms") == 0
            used_keys.update(c["key"] for c in r["calls"])
            if "error" in r:
                failures.append(dict(record=case["id"],folder=folder,error=r["error"]))
                continue
            rows = r["rows"]
            assert [a for a,*_ in rows] == list(range(1,quran_text.ayah_count(case["surah"])+1))
            assert all(0 <= ms <= r["duration_ms"] for _,ms,_ in rows)
            assert all(b[1] > a[1] for a,b in zip(rows,rows[1:]))
    assert set(pcm_identity) == {c["id"] for c in corpus()}
    secondary = json.loads((TASK / "secondary-inputs.json").read_text())
    for c in secondary:
        pcm = np.load(c["input"],allow_pickle=False)
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == c["pcm_sha"]
        if c["id"] in pcm_identity:
            assert pcm_identity[c["id"]] == c["pcm_sha"]
    secondary_coverage = {}
    secondary_inputs = {c["id"]:c for c in secondary}
    for folder,methods in {"asr-nemo":["ctc","rnnt"],
            "asr-whisper":["greedy","beam5","nbest5","greedy_overlap","greedy_sdpa","token_timestamps","teacher_attention"],
            "muaalem":["madd2","madd4","madd6","connected4"],
            "nemo-forced":["forced_default_bpe","forced_acoustic_bpe"]}.items():
        ids = {c["id"] for c in secondary if c["surah"] or folder.startswith("asr-")}
        for method in methods:
            records = [json.loads(p.read_text()) for p in (TASK/folder).glob(f"*-{method}.json")]
            assert {r["case"]["id"] for r in records} == ids,(folder,method)
            secondary_coverage[folder+"/"+method] = len(records)
            failures.extend(dict(record=r["case"]["id"],folder=folder,method=method,error=r["error"])
                            for r in records if "error" in r)
    secondary_matrices = 0
    for folder in ("muaalem","nemo-forced"):
        for path in (TASK/folder).glob("*-matrix.npy"):
            key = path.name.removesuffix("-matrix.npy")
            a = np.load(path,allow_pickle=False)
            meta = json.loads(path.with_suffix(".json").read_text())
            actual = hashlib.sha256(a.tobytes()).hexdigest()
            assert actual == meta["matrix_sha"] and np.isfinite(a).all()
            assert meta.get("pcm_sha",meta.get("PCM_sha")) == secondary_inputs[key]["pcm_sha"]
            for result in (TASK/folder).glob(key+"-*.json"):
                row = json.loads(result.read_text())
                if "case" in row:
                    assert row["matrix_sha"] == actual
            secondary_matrices += 1
    broken_cache = []
    cache_count = 0
    for path in (TASK/"qpc-logits").glob("*.npy"):
        try:
            meta = json.loads(path.with_suffix(".json").read_text())
            a = np.load(path,allow_pickle=False)
            assert hashlib.sha256(a.tobytes()).hexdigest() == meta["sha256"]
            assert a.ndim == 3 and a.shape[0] == 1 and a.shape[-1] == 40 and np.isfinite(a).all()
            cache_count += 1
        except Exception as exc:
            broken_cache.append(dict(key=path.stem,used=path.stem in used_keys,error=str(exc)))
    assert not any(r["used"] for r in broken_cache),broken_cache
    root = Path(SNAPSHOT["root"])
    assets = [root/"data/word_tr_cache.json",root/"data/model_text_targets.json",
              root/"models/wav2vec2-quran-phonetics.onnx.vocab.json",
              root/"models/wav2vec2-quran-phonetics.onnx.meta.json"]
    write_json(TASK/"runtime-assets-at-end.json", {str(p):dict(sha256=file_sha(p),mtime_ns=p.stat().st_mtime_ns)
                                                for p in assets if p.is_file()})
    result = dict(protected_unchanged=len(protected),snapshot_source_unchanged=len(SNAPSHOT["files"]),
                  qpc_coverage=coverage,secondary_coverage=secondary_coverage,failures=failures,
                  checked_cache_arrays=cache_count,broken_cache=broken_cache,
                  checked_secondary_matrices=secondary_matrices,
                  source_archive_sha=file_sha(TASK/"source.zip"))
    write_json(TASK/"integrity.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="secondary_coverage"},indent=2))


if __name__ == "__main__":
    main()
