"""The phonetic dictionary must follow the canonical word boundaries exactly."""

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from surah_cutter.data import quran_text, transliteration as tr


@pytest.fixture
def mapping():
    return json.loads(tr.CACHE_PATH.read_text(encoding="utf-8"))


def test_every_canonical_word_has_a_nonempty_target(mapping):
    from surah_cutter.core.align import _build_targets

    vocab = json.loads((Path(__file__).parent / "fixtures/quran_phonetics_vocab.json")
                       .read_text(encoding="utf-8"))
    bundle = SimpleNamespace(vocab=vocab, delim_id=vocab["|"])
    assert len(mapping) == 6236
    assert sum(map(len, mapping.values())) == 77432
    total = 0
    for surah in quran_text.surah_ayah_counts():
        words = quran_text.surah_words(surah)
        targets = tr.surah_word_tr(surah)
        assert len(words) == len(targets)
        assert all(isinstance(target, str) and target.strip() for target in targets)
        _, spans = _build_targets(words, targets, bundle)
        assert len(spans) == len(words)
        assert all(start < end for start, end in spans)
        total += len(targets)
    assert total == 77432


@pytest.mark.parametrize("key,position,joined,following,last", [
    ("2:181", 2, "baʿdamā", "samiʿahu", "ʿalīmun"),
    ("8:6", 3, "baʿda mā", "tabayyana", "yanẓurūna"),
    ("13:37", 7, "baʿdamā", "jāaka", "wāqin"),
])
def test_source_join_is_split_without_changing_transliteration(
    mapping, key, position, joined, following, last,
):
    expected = mapping[key].copy()
    assert expected[position:position + 3] == ["baʿda", "mā", following]
    assert expected[-1] == last
    mapping[key][position:position + 2] = [joined]
    rebuilt = tr.reconcile_map(mapping)
    assert rebuilt[key] == expected
    assert "".join(rebuilt[key][position:position + 2]) == joined.replace(" ", "")
    assert mapping[key][position] == joined  # input remains intact
    assert tr.reconcile_map(rebuilt) == rebuilt


@pytest.mark.parametrize("damage", ["missing", "extra", "empty", "short", "invalid", "wrong_join"])
def test_bad_cache_is_rejected_instead_of_padding_or_shifting(mapping, damage):
    if damage == "missing":
        del mapping["1:1"]
    elif damage == "extra":
        mapping["1:999"] = ["a"]
    elif damage == "empty":
        mapping["1:1"][0] = " "
    elif damage == "short":
        mapping["1:1"].pop()
    elif damage == "invalid":
        mapping["1:1"] = "abcd"
    else:
        mapping["2:181"][2:4] = ["unknown"]
    with pytest.raises(ValueError):
        tr.reconcile_map(mapping)


def test_rebuild_is_offline_reproducible_and_does_not_publish_bad_data(tmp_path, mapping):
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    for key, position, joined in [("2:181", 2, "baʿdamā"), ("8:6", 3, "baʿda mā"),
                                  ("13:37", 7, "baʿdamā")]:
        mapping[key][position:position + 2] = [joined]
    source.write_text(json.dumps(mapping), encoding="utf-8")
    tr.build_cache(cache_path=output, source_path=source)
    assert output.read_bytes() == tr.CACHE_PATH.read_bytes()
    before = output.read_bytes()
    mapping["1:1"].pop()
    source.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError):
        tr.build_cache(cache_path=output, source_path=source)
    assert output.read_bytes() == before


def test_missing_runtime_cache_does_not_start_a_network_rebuild(tmp_path, monkeypatch):
    def unexpected_network():
        pytest.fail("Runtime must not rebuild the source dataset")
    monkeypatch.setattr(tr, "_build_map_from_dataset", unexpected_network)
    with pytest.raises(FileNotFoundError):
        tr.surah_word_tr(1, cache_path=tmp_path / "missing.json")
