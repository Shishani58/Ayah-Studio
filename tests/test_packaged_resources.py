from surah_cutter.data.quran_text import surah_ayah_counts
from surah_cutter.resources import external_model_path, resource_path


def test_small_resources_are_available():
    counts = surah_ayah_counts()
    assert len(counts) == 114 and sum(counts.values()) == 6236
    assert resource_path("data", "word_tr_cache.json").is_file()
    assert resource_path("data", "model_text_targets.json").is_file()
    assert resource_path("assets", "fonts", "qpc_hafs.ttf").is_file()


def test_external_model_override_is_kept_out_of_wheel(tmp_path, monkeypatch):
    monkeypatch.setenv("SURAH_CUTTER_MODELS_DIR", str(tmp_path))
    assert external_model_path("weights.onnx") == tmp_path / "weights.onnx"
