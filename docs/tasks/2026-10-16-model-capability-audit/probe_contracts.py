"""Verify feature preprocessing and demonstrate configurable phonetic targets."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from quran_transcript import MoshafAttributes, quran_phonetizer
from transformers import Wav2Vec2FeatureExtractor

OUT = Path(__file__).resolve().parent


def main():
    config = json.loads((OUT / "quran-phonetics-preprocessor_config.json").read_text())
    extractor = Wav2Vec2FeatureExtractor(**config)
    observations = json.loads((OUT / "qpc-output.json").read_text())
    parity = []
    for surah in (1, 80):
        pcm = np.load(OUT / f"pcm-{surah:03d}.npy", allow_pickle=False)
        processed = extractor(pcm, sampling_rate=16000, return_tensors="np", padding=True)
        actual = processed.input_values[0]
        expected = next(row["normalized_sha256"] for row in observations["results"]
                        if row["source"].endswith(f"{surah:03d}.mp3"))
        parity.append({"surah": surah,
                       "exact_normalized_input_sha": hashlib.sha256(actual.tobytes()).hexdigest() == expected,
                       "input_shape": list(actual.shape),
                       "mask_all_one": bool(np.all(processed.attention_mask == 1))})
    phrase = json.loads((OUT / "canonical-phrase.json").read_text())
    variants = []
    for length in (2, 4, 6):
        attrs = MoshafAttributes(rewaya="hafs", madd_monfasel_len=4,
                                madd_mottasel_len=4, madd_mottasel_waqf=4,
                                madd_aared_len=length)
        phon = quran_phonetizer(phrase["text"], attrs, sura_idx=phrase["surah"], remove_spaces=True)
        variants.append({"madd_aared_len": length, "phonemes": phon.phonemes, "count": len(phon.phonemes)})
    functions = [quran_phonetizer, Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm]
    sources = {inspect.getfile(fn): hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest()
               for fn in functions}
    result = {"official_feature_extractor_parity": parity, "canonical_phrase": phrase,
              "muaalem_representations": variants, "library_source_sha256": sources}
    (OUT / "contracts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    assert all(row["exact_normalized_input_sha"] and row["mask_all_one"] for row in parity)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
