import re

from surah_cutter.core import model_store


def test_general_models_have_open_pinned_downloads_and_required_runtime_files():
    required = {
        "whisper-small": {"model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "preprocessor_config.json"},
        "whisper-medium": {"model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "preprocessor_config.json"},
        "whisper-large-v3": {"model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "preprocessor_config.json"},
        "whisper-large-v3-turbo": {"model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "preprocessor_config.json"},
        "fastconformer-arabic": {"stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo"},
        "mms-arabic": {"model.safetensors", "adapter.ara.safetensors", "vocab.json", "preprocessor_config.json"},
    }
    ids = [item["id"] for item in model_store.catalog()]
    assert len(ids) == len(set(ids))
    for model_id, paths in required.items():
        item = model_store.spec(model_id)
        assert item["gated"] is False
        assert item["language"] == "ar"
        assert re.fullmatch(r"[0-9a-f]{40}", item["revision"])
        assert paths <= {f["path"] for f in item["files"]}
        for file in item["files"]:
            assert file["size"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", file["sha256"]) if file["sha256"] else re.fullmatch(r"[0-9a-f]{40}", file["git_oid"])
        assert not any(f["path"].endswith(".bin") for f in item["files"])
        if item["runtime_family"] == "whisper":
            assert item["repo"].startswith("openai/whisper-")
            assert not item["repo"].endswith(".en")


def test_mms_download_contains_only_arabic_adapter():
    item = model_store.spec("mms-arabic")
    adapters = [f["path"] for f in item["files"] if f["path"].startswith("adapter.")]
    assert adapters == ["adapter.ara.safetensors"]
