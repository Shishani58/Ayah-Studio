"""FULL must recover early ASR EOS without skipping unconfirmed Quran words."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from surah_cutter.core import model_worker
from surah_cutter.core.model_worker import Acoustic, whisper_alignment_text, whisper_energy_pauses


def acoustic_with_transcripts(transcripts):
    acoustic = Acoustic.__new__(Acoustic)
    acoustic.torch = SimpleNamespace(inference_mode=nullcontext)
    class Processor:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_features=None, attention_mask=None)

        def batch_decode(self, *args, **kwargs):
            return [next(transcripts)]

    acoustic.processor = Processor()
    acoustic.model = SimpleNamespace(generate=lambda *args, **kwargs: None)
    acoustic.whisper_clip = lambda pcm, words, **kwargs: [(i * 1000, .9) for i in range(len(words))]
    return acoustic


def test_full_recovers_early_eos_only_after_confirming_next_known_prefix(monkeypatch):
    words = "one two three four five six seven eight".split()
    acoustic = acoustic_with_transcripts(iter([
        "one two three", "one two three", "four five six seven eight", "four five six seven eight",
    ]))
    monkeypatch.setattr(model_worker, "whisper_energy_pauses", lambda pcm: [(4000, 5000)])
    rows = acoustic.whisper_align(np.zeros(32 * 16000), words)
    assert len(rows) == len(words)
    assert [ms for ms, _ in rows] == [0, 1000, 2000, 4500, 5500, 6500, 7500, 8500]


@pytest.mark.parametrize("following", ["five six seven eight", "unrelated other text", "eight"])
def test_full_does_not_skip_an_unconfirmed_word_after_a_pause(monkeypatch, following):
    words = "one two three four five six seven eight".split()
    acoustic = acoustic_with_transcripts(iter(["one two three", "one two three", following]))
    monkeypatch.setattr(model_worker, "whisper_energy_pauses", lambda pcm: [(4000, 5000)])
    with pytest.raises(ValueError, match="продвинулся"):
        acoustic.whisper_align(np.zeros(32 * 16000), words)


def test_full_rejects_a_retry_that_changes_the_known_prefix(monkeypatch):
    acoustic = acoustic_with_transcripts(iter(["one two three", "one two"]))
    monkeypatch.setattr(model_worker, "whisper_energy_pauses", lambda pcm: [(4000, 5000)])
    with pytest.raises(ValueError, match="продвинулся"):
        acoustic.whisper_align(np.zeros(32 * 16000), "one two three four five six".split())


def test_full_does_not_treat_silence_as_confirmed_text():
    acoustic = acoustic_with_transcripts(iter(["one two three"]))
    with pytest.raises(ValueError, match="продвинулся"):
        acoustic.whisper_align(np.zeros(32 * 16000), "one two three four five six".split())


def test_full_still_requires_all_canonical_words_at_the_audio_end(monkeypatch):
    acoustic = acoustic_with_transcripts(iter(["one two three four", "three four"]))
    acoustic.whisper_clip = lambda pcm, words, **kwargs: [
        (ms, .9) for ms in ([0, 1000, 23000, 25000] if len(words) == 4 else [0, 1000])
    ]
    monkeypatch.setattr(model_worker, "whisper_energy_pauses", lambda pcm: [])
    with pytest.raises(ValueError, match="весь конец"):
        acoustic.whisper_align(np.zeros(32 * 16000), "one two three four five six".split())


@pytest.mark.parametrize(("word", "expected"), [
    ("الٓمٓ", "أَلِفْ لَامْ مِيمْ"), ("كهيعص", "كَافْ هَا يَا عَيْنْ صَادْ"),
    ("حم", "حَا مِيمْ"), ("عسق", "عَيْنْ سِينْ قَافْ"),
    ("ق", "قَافْ"), ("ن", "نُونْ"), ("أَلَمْ", "أَلَمْ"), ("عَلَيْهِمْ", "عَلَيْهِمْ"),
])
def test_opening_letters_use_spoken_names_without_changing_other_words(word, expected):
    assert whisper_alignment_text(word) == expected


def test_locator_ignores_distant_repeated_phrase():
    words = ["one", "two", "three", "four"] + [f"missing{i}" for i in range(150)] + ["ending", "phrase"]
    acoustic = acoustic_with_transcripts(iter(["one two three four ending phrase"]))
    located = acoustic.whisper_locate(np.zeros(30 * 16000), words)
    assert located[0] == 4


def test_locator_maps_multiple_orthographic_units_to_one_canonical_word():
    acoustic = acoustic_with_transcripts(iter(["يا أيها الناس اعبدوا ربكم"]))
    assert acoustic.whisper_locate(np.zeros(30 * 16000), ["يا أيها", "الناس", "اعبدوا", "ربكم"])[0] == 4


def test_locator_rejects_partial_last_canonical_word():
    acoustic = acoustic_with_transcripts(iter(["one two يا"]))
    assert acoustic.whisper_locate(np.zeros(30 * 16000), ["one", "two", "يا أيها"]) is None


def test_full_retries_unconfirmed_asr_with_shorter_audio_without_skipping_words(monkeypatch):
    words = "one two three four five six".split()
    acoustic = acoustic_with_transcripts(iter([
        "unrelated words", "other unrelated words", "one two three four five six", "four five six",
    ]))
    seen = []
    def aligned(pcm, targets, **kwargs):
        seen.append((len(pcm) / 16000, list(targets)))
        return [(i * 1000, .9) for i in range(len(targets))]
    acoustic.whisper_clip = aligned
    monkeypatch.setattr(model_worker, "whisper_energy_pauses", lambda pcm: [])
    result = acoustic.whisper_align(np.zeros(32 * 16000), words)
    assert len(result) == len(words)
    assert seen[0] == (10, words)
    assert seen[1][0] == 29


def test_standalone_dsp_detects_only_internal_contrasted_pauses():
    pcm = np.full(6 * 16000, .3, dtype=np.float32)
    pcm[:16000] = 0
    pcm[2 * 16000:3 * 16000] = 0
    pcm[4 * 16000:4 * 16000 + 1600] = 0
    pcm[5 * 16000:] = 0
    pauses = whisper_energy_pauses(pcm)
    assert len(pauses) == 1
    start, end = pauses[0]
    assert 1950 < start < 2050 and 2950 < end < 3050


def test_standalone_dsp_does_not_invent_pauses_in_silence_or_constant_noise():
    assert whisper_energy_pauses(np.zeros(5 * 16000)) == []
    assert whisper_energy_pauses(np.full(5 * 16000, .01)) == []


def test_standalone_dsp_keeps_deep_long_pause_after_quieter_speech():
    pcm = np.tile(np.array([.08, .12, .2, .3], dtype=np.float32), 3 * 16000)
    pcm[3 * 16000:4 * 16000] = .08
    pcm[4 * 16000:6 * 16000] = .001
    pcm[6 * 16000:7 * 16000] = .3
    assert any(start < 4050 and end > 5950 for start, end in whisper_energy_pauses(pcm))
